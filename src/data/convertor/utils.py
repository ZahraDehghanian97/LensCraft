import functools

import torch
import numpy as np
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


_SLERP_NLERP_THRESHOLD = 0.9995


def _slerp(q1, q2, alpha):
    dot = torch.sum(q1 * q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > _SLERP_NLERP_THRESHOLD:
        q = q1 + alpha * (q2 - q1)
        return q / torch.norm(q)

    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta = torch.sin(theta)
    return (
        torch.sin((1.0 - alpha) * theta) / sin_theta * q1
        + torch.sin(alpha * theta) / sin_theta * q2
    )


def _interpolate_rotations(rotations, src_times, tgt_times):
    quats = matrix_to_quaternion(rotations)
    out = torch.zeros((len(tgt_times), 4), device=quats.device, dtype=quats.dtype)

    for t_idx, t in enumerate(tgt_times):
        if t <= src_times[0]:
            out[t_idx] = quats[0]
        elif t >= src_times[-1]:
            out[t_idx] = quats[-1]
        else:
            next_idx = torch.searchsorted(src_times, t)
            prev_idx = next_idx - 1
            span = src_times[next_idx] - src_times[prev_idx]
            alpha = (t - src_times[prev_idx]) / span
            out[t_idx] = _slerp(quats[prev_idx], quats[next_idx], alpha)

    return quaternion_to_matrix(out)


def _interpolate_translations(translations, src_times, tgt_times):
    """Linearly resample each xyz channel of [src_len, 3] onto `tgt_times`."""
    out = torch.zeros(
        (len(tgt_times), 3), device=translations.device, dtype=translations.dtype
    )
    for dim in range(3):
        out[:, dim] = torch_interp(tgt_times, src_times, translations[:, dim])
    return out


def _resample_one_trajectory(valid_trajectory, num_target):
    valid_len = valid_trajectory.shape[0]
    device, dtype = valid_trajectory.device, valid_trajectory.dtype

    out = torch.zeros((num_target, 4, 4), device=device, dtype=dtype)
    out[:, 3, 3] = 1.0

    if valid_len == 1:
        out[:] = valid_trajectory.repeat(num_target, 1, 1)
        return out

    src_times = torch.linspace(0, 1, valid_len, device=device)
    tgt_times = torch.linspace(0, 1, num_target, device=device)

    out[:, :3, :3] = _interpolate_rotations(
        valid_trajectory[:, :3, :3], src_times, tgt_times
    )
    out[:, :3, 3] = _interpolate_translations(
        valid_trajectory[:, :3, 3], src_times, tgt_times
    )
    return out


@handle_single_or_batch(arg_specs=[(0, 3), (1, 0), (3, 0)])
def resample_batch_trajectories(
    batch_trajectory, current_valid_len, target_len, valid_target_len=None
):
    batch_size, max_seq_len = batch_trajectory.shape[:2]
    device = batch_trajectory.device

    if valid_target_len is None:
        valid_target_len = torch.full(
            (batch_size,), target_len, device=device, dtype=torch.long
        )

    resampled_batch = torch.zeros(
        (batch_size, target_len, 4, 4), device=device, dtype=batch_trajectory.dtype
    )
    resampled_batch[:, :, 3, 3] = 1.0
    padding_mask = torch.zeros((batch_size, target_len), dtype=torch.bool, device=device)

    for i in range(batch_size):
        valid_len = (
            current_valid_len[i].item() if current_valid_len is not None else max_seq_len
        )
        valid_len = min(valid_len, max_seq_len)

        num_target = (
            valid_target_len[i].item()
            if valid_target_len.dim() > 0
            else valid_target_len.item()
        )
        num_target = min(num_target, target_len)
        if num_target < target_len:
            padding_mask[i, num_target:] = True

        valid_trajectory = batch_trajectory[i, :valid_len]
        resampled_batch[i, :num_target] = _resample_one_trajectory(
            valid_trajectory, num_target
        )

    return resampled_batch, padding_mask
