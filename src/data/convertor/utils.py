import functools

import torch
import numpy as np
import torch.nn.functional as F

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
            resampled_batch[i] = resampled_transposed.squeeze(-1)
    
    return resampled_batch, padding_mask