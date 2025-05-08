import functools

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
import torch.nn.functional as F


def handle_single_or_batch(single_item_dim: int = 1, arg_index: int = 0, device=None, dtype=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = args[arg_index]

            is_numpy = isinstance(x, np.ndarray)
            is_torch  = torch.is_tensor(x)

            if not (is_numpy or is_torch):
                raise TypeError(
                    f"Expected NumPy array or Torch tensor at position {arg_index}, "
                    f"got {type(x).__name__}"
                )

            xt = (torch.as_tensor(x, device=device, dtype=dtype) if is_numpy
                  else (x.to(device=device, dtype=dtype) if (device or dtype) else x))

            is_single = xt.ndim == single_item_dim
            if is_single:
                xt = xt.unsqueeze(0)

            new_args = list(args)
            new_args[arg_index] = xt

            out = func(*new_args, **kwargs)

            if is_single:
                out = out.squeeze(0)
            return out

        return wrapper
    return decorator




def fix_camera_traj_length(trajectories, target_frames=30):
    """
    Upsample or downsample trajectories to specified number of frames.
    """
    N, T, _, _ = trajectories.shape
    resized_trajectories = np.zeros((N, target_frames, 4, 4))
    resized_trajectories[:, :, 3, 3] = 1  
    
    times = np.linspace(0, 1, T)
    target_times = np.linspace(0, 1, target_frames)
    
    for i in range(N):
        positions = trajectories[i, :, :3, 3]
        rotations = R.from_matrix(trajectories[i, :, :3, :3])
        
        resized_trajectories[i, :, :3, 3] = interp1d(times, positions, axis=0, 
                                                    kind='linear', fill_value='extrapolate')(target_times)
        resized_trajectories[i, :, :3, :3] = Slerp(times, rotations)(target_times).as_matrix()
    
    return resized_trajectories






def et_to_6dof(trajectories):
    """
    Convert camera trajectories to 6DoF: position (3), Euler angles (3)
    """
    N, T, _, _ = trajectories.shape
    result = np.zeros((N, T, 6))
    result[:, :, :3] = trajectories[:, :, :3, 3]
    for i in range(N):
        for t in range(T):
            rotation_matrix = trajectories[i, t, :3, :3]
            result[i, t, 3:6] = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
    
    return result




def axis_angle_to_quaternion(aa):
    """Convert axis-angle (B, 3) to quaternion (B, 4)"""
    angle = torch.norm(aa, dim=-1, keepdim=True) + 1e-8
    axis = aa / angle
    half_angle = angle * 0.5
    quat = torch.cat([axis * torch.sin(half_angle), torch.cos(half_angle)], dim=-1)
    return quat



def quaternion_to_axis_angle(q):
    """Convert quaternion (B, 4) to axis-angle (B, 3)"""
    q = F.normalize(q, dim=-1)
    sin_half_angle = torch.norm(q[..., :3], dim=-1, keepdim=True)
    cos_half_angle = q[..., 3:4]
    half_angle = torch.atan2(sin_half_angle, cos_half_angle)
    axis = q[..., :3] / (sin_half_angle + 1e-8)
    return axis * 2 * half_angle




def slerp_torch(quats, times, target_times):
    """
    Perform SLERP between quaternions using PyTorch.
    quats: (T, 4)
    times: (T,) original time steps in [0,1]
    target_times: (N,) target time steps in [0,1]
    Returns: (N, 4) interpolated quaternions
    """
    N = target_times.shape[0]
    T = times.shape[0]
    device = quats.device

    # Find interval indices for each target_time
    indices = torch.searchsorted(times, target_times, right=True).clamp(1, T - 1)
    t0 = times[indices - 1]
    t1 = times[indices]
    q0 = quats[indices - 1]
    q1 = quats[indices]

    # Interpolation factor
    t = (target_times - t0) / (t1 - t0 + 1e-8)
    t = t.unsqueeze(-1)

    # SLERP
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    factor0 = torch.sin((1 - t) * theta) / (sin_theta + 1e-8)
    factor1 = torch.sin(t * theta) / (sin_theta + 1e-8)

    out = factor0 * q0 + factor1 * q1
    return F.normalize(out, dim=-1)






def fix_subject_traj_length(trajectories, target_frames=30):
    """
    Resample a (T, 6) array of trajectories (3 pos + 3 rot) to a new length.
    """
    T = trajectories.shape[0]
    
    # Split position and rotation (assume axis-angle for rotation)
    pos = trajectories[:, :3]  # (T, 3)
    rot = trajectories[:, 3:]  # (T, 3) axis-angle

    # Normalize time to [0, 1]
    t_original = torch.linspace(0, 1, T, device=trajectories.device)
    t_target = torch.linspace(0, 1, target_frames, device=trajectories.device)

    # Position interpolation
    interp_pos = F.interpolate(pos.T.unsqueeze(0), size=target_frames, mode='linear', align_corners=True).squeeze(0).T

    # Rotation interpolation (via SLERP in axis-angle using quaternions)
    rot_quat = axis_angle_to_quaternion(rot)  # (T, 4)
    interp_quat = slerp_torch(rot_quat, t_original, t_target)  # (target_frames, 4)
    interp_rot = quaternion_to_axis_angle(interp_quat)  # (target_frames, 3)

    return torch.cat([interp_pos, interp_rot], dim=-1)


def resample_batch_trajectories(batch_trajectory, current_valid_len, target_len):
    batch_size = batch_trajectory.shape[0]
    max_seq_len = batch_trajectory.shape[1]
    feature_dims = batch_trajectory.shape[2:]
    device = batch_trajectory.device
    dtype = batch_trajectory.dtype
    
    resampled_batch = torch.zeros((batch_size, target_len) + feature_dims, device=device, dtype=dtype)
    padding_mask = torch.ones((batch_size, target_len), dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        valid_len = current_valid_len[i].item()
        
        if valid_len <= 0:
            padding_mask[i, :] = False
            continue
            
        valid_len = min(valid_len, max_seq_len)
        valid_trajectory = batch_trajectory[i, :valid_len]
        
        if valid_len == 1:
            resampled_batch[i] = valid_trajectory.repeat(target_len, *([1] * len(feature_dims)))
            continue
        
        flat_size = int(torch.prod(torch.tensor(feature_dims))) if feature_dims else 1
        flattened = valid_trajectory.reshape(valid_len, flat_size)
        
        transposed = flattened.T
        
        resampled_flat = F.interpolate(
            transposed.unsqueeze(0),
            size=target_len,
            mode='linear',
            align_corners=True
        ).squeeze(0)
        
        resampled_transposed = resampled_flat.T
        
        if feature_dims:
            resampled_batch[i] = resampled_transposed.reshape((target_len,) + feature_dims)
        else:
            resampled_batch[i] = resampled_transposed
    
    return resampled_batch, padding_mask
