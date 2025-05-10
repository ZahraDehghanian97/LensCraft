import functools

import torch
import numpy as np
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

def handle_single_or_batch(single_item_dim: int = 1, arg_index=0, device=None, dtype=None):
    if isinstance(arg_index, int):
        arg_indices = [arg_index]
    else:
        arg_indices = list(arg_index)
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = list(args)
            is_single = None
            
            for idx in arg_indices:
                if idx >= len(args):
                    continue
                
                x = args[idx]
                
                if x is None:
                    continue
                
                is_numpy = isinstance(x, np.ndarray)
                is_torch = torch.is_tensor(x)
                
                if not (is_numpy or is_torch):
                    raise TypeError(
                        f"Expected NumPy array or Torch tensor at position {idx}, "
                        f"got {type(x).__name__}"
                    )
                
                xt = (torch.as_tensor(x, device=device, dtype=dtype) if is_numpy
                      else (x.to(device=device, dtype=dtype) if (device or dtype) else x))
                
                current_is_single = xt.ndim == single_item_dim
                
                if is_single is None:
                    is_single = current_is_single
                
                if is_single:
                    xt = xt.unsqueeze(0)
                
                new_args[idx] = xt
            
            out = func(*new_args, **kwargs)
            
            if is_single:
                out = out.squeeze(0)
            return out
        return wrapper
    return decorator


def resample_batch_trajectories(batch_trajectory, current_valid_len, target_len):
    batch_size = batch_trajectory.shape[0]
    max_seq_len = batch_trajectory.shape[1]
    device = batch_trajectory.device
    dtype = batch_trajectory.dtype
    
    resampled_batch = torch.zeros((batch_size, target_len, 4, 4), device=device, dtype=dtype)
    resampled_batch[:, :, 3, 3] = 1.0
    
    padding_mask = torch.ones((batch_size, target_len), dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        valid_len = current_valid_len[i].item() if current_valid_len is not None else max_seq_len
        
        if valid_len <= 0:
            padding_mask[i, :] = False
            continue
            
        valid_len = min(valid_len, max_seq_len)
        valid_trajectory = batch_trajectory[i, :valid_len]
        
        if valid_len == 1:
            resampled_batch[i] = valid_trajectory.repeat(target_len, 1, 1)
            continue
        
        translations = valid_trajectory[:, :3, 3]
        
        src_times = torch.linspace(0, 1, valid_len, device=device)
        tgt_times = torch.linspace(0, 1, target_len, device=device)
        
        interp_translations = torch.zeros((target_len, 3), device=device, dtype=dtype)
        for dim in range(3):
            interp_translations[:, dim] = torch.interp(
                tgt_times, 
                src_times, 
                translations[:, dim]
            )
        
        rotations = valid_trajectory[:, :3, :3]
        quats = matrix_to_quaternion(rotations)
        interp_quats = torch.zeros((target_len, 4), device=device, dtype=dtype)
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
        
        resampled_batch[i, :, :3, :3] = interp_rotations
        resampled_batch[i, :, :3, 3] = interp_translations
        resampled_batch[i, :, 3, :3] = 0.0
        
    return resampled_batch, padding_mask
