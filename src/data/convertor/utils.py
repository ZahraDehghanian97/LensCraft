import functools

import torch
import numpy as np
import torch.nn.functional as F
from utils.pytorch3d_transform import matrix_to_quaternion, quaternion_to_matrix


def handle_single_or_batch(arg_specs=(0, 1), device=None, dtype=None):
    arg_pairs = []
    for spec in arg_specs:
        if isinstance(spec, int):
            arg_pairs.append((spec, 1))
        else:
            idx, dim = spec
            arg_pairs.append((idx, 1 if dim is None else dim))

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = list(args)
            single_flags = {}
            
            for idx, dim in arg_pairs:
                if idx >= len(args):
                    continue
                
                x = args[idx]
                
                if x is None:
                    continue

                if isinstance(x, np.ndarray):
                    xt = torch.as_tensor(x, device=device, dtype=dtype)
                elif torch.is_tensor(x):
                    xt = x.to(device=device, dtype=dtype) if (device or dtype) else x

                is_single = xt.ndim == dim
                single_flags[idx] = is_single

                if is_single:
                    xt = xt.unsqueeze(0)
                
                new_args[idx] = xt
            
            out = func(*new_args, **kwargs)
            
            if single_flags and any(single_flags.values()):
                if isinstance(out, tuple):
                    out = tuple(o.squeeze(0) if o is not None else None for o in out)
                else:
                    out = out.squeeze(0)
            return out
        return wrapper
    return decorator



def torch_interp(x, xp, fp):    
    i = torch.searchsorted(xp, x)
    i = torch.clamp(i, 1, len(xp) - 1)
    
    t = (x - xp[i - 1]) / (xp[i] - xp[i - 1] + 1e-8)
    result = fp[i - 1] + t * (fp[i] - fp[i - 1])
    
    result = torch.where(x <= xp[0], fp[0], result)
    result = torch.where(x >= xp[-1], fp[-1], result)
    
    return result


@handle_single_or_batch(arg_specs=[(0, 3), (1, 0), (3, 0)])
def resample_batch_trajectories(batch_trajectory, current_valid_len, target_len, valid_target_len=None):
    batch_size = batch_trajectory.shape[0]
    max_seq_len = batch_trajectory.shape[1]
    device = batch_trajectory.device
    dtype = batch_trajectory.dtype
    
    if valid_target_len is None:
        valid_target_len = torch.full((batch_size,), target_len, device=device, dtype=torch.long)
    
    resampled_batch = torch.zeros((batch_size, target_len, 4, 4), device=device, dtype=dtype)
    resampled_batch[:, :, 3, 3] = 1.0
    
    padding_mask = torch.zeros((batch_size, target_len), dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        valid_len = current_valid_len[i].item() if current_valid_len is not None else max_seq_len
        current_valid_target_len = valid_target_len[i].item() if valid_target_len.dim() > 0 else valid_target_len.item()
        current_valid_target_len = min(current_valid_target_len, target_len)
        if current_valid_target_len < target_len:
            padding_mask[i, current_valid_target_len:] = True
        
        valid_len = min(valid_len, max_seq_len)
        valid_trajectory = batch_trajectory[i, :valid_len]
        
        if valid_len == 1:
            resampled_batch[i, :current_valid_target_len] = valid_trajectory.repeat(current_valid_target_len, 1, 1)
            continue
        
        translations = valid_trajectory[:, :3, 3]
        
        src_times = torch.linspace(0, 1, valid_len, device=device)
        tgt_times = torch.linspace(0, 1, current_valid_target_len, device=device)
        
        interp_translations = torch.zeros((current_valid_target_len, 3), device=device, dtype=dtype)
        for dim in range(3):
            interp_translations[:, dim] = torch_interp(
                tgt_times, 
                src_times, 
                translations[:, dim]
            )
        
        rotations = valid_trajectory[:, :3, :3]
        quats = matrix_to_quaternion(rotations)
        interp_quats = torch.zeros((current_valid_target_len, 4), device=device, dtype=dtype)
        for t_idx, t in enumerate(tgt_times):
            if t <= src_times[0]:
                interp_quats[t_idx] = quats[0]
            elif t >= src_times[-1]:
                interp_quats[t_idx] = quats[-1]
            else:
                next_idx = torch.searchsorted(src_times, t)
                prev_idx = next_idx - 1
                
                q1 = quats[prev_idx]
                q2 = quats[next_idx]
                
                t1 = src_times[prev_idx]
                t2 = src_times[next_idx]
                alpha = (t - t1) / (t2 - t1)
                
                dot_product = torch.sum(q1 * q2)
                
                if dot_product < 0:
                    q2 = -q2
                    dot_product = -dot_product
                    
                if dot_product > 0.9995:
                    interp_quats[t_idx] = q1 + alpha * (q2 - q1)
                    interp_quats[t_idx] /= torch.norm(interp_quats[t_idx])
                else:
                    theta = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
                    sin_theta = torch.sin(theta)
                    
                    interp_quats[t_idx] = (
                        torch.sin((1.0 - alpha) * theta) / sin_theta * q1 +
                        torch.sin(alpha * theta) / sin_theta * q2
                    )

        interp_rotations = quaternion_to_matrix(interp_quats)
        
        resampled_batch[i, :current_valid_target_len, :3, :3] = interp_rotations
        resampled_batch[i, :current_valid_target_len, :3, 3] = interp_translations
        resampled_batch[i, :current_valid_target_len, 3, :3] = 0.0
        
    return resampled_batch, padding_mask
