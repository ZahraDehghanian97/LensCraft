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


def _batched_slerp(q0, q1, alpha):
    """Spherical-linear interpolation between quaternion fields ``q0``/``q1``.

    ``q0``/``q1`` are ``[..., 4]`` and ``alpha`` is broadcastable to ``[...]``.
    Returns ``[..., 4]``. Numerically equivalent (per element) to the previous
    scalar ``_slerp``, but fully vectorised — no Python loop, no host syncs.
    """
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    # Take the shortest path: flip q1 where the dot product is negative.
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()

    alpha = alpha.unsqueeze(-1)

    # Near-parallel quaternions: fall back to normalised linear interpolation.
    nlerp = q0 + alpha * (q1 - q0)
    nlerp = nlerp / (torch.norm(nlerp, dim=-1, keepdim=True) + 1e-8)

    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta = torch.sin(theta) + 1e-8
    slerp = (
        torch.sin((1.0 - alpha) * theta) / sin_theta * q0
        + torch.sin(alpha * theta) / sin_theta * q1
    )

    return torch.where(dot > _SLERP_NLERP_THRESHOLD, nlerp, slerp)


@handle_single_or_batch(arg_specs=[(0, 3), (1, 0), (3, 0)])
def resample_batch_trajectories(
    batch_trajectory, current_valid_len, target_len, valid_target_len=None
):
    """Resample each ``[valid_len, 4, 4]`` trajectory in the batch onto a regular
    ``num_target``-length grid (slerp on rotations, linear on translations).

    Fully vectorised across both the batch and the time axis. Because the source
    grid is ``linspace(0, 1, valid_len)`` (uniform), the bracketing source frames
    and interpolation weight are closed-form, so no per-sample ``searchsorted`` /
    Python loop is needed.
    """
    batch_size, max_seq_len = batch_trajectory.shape[:2]
    device = batch_trajectory.device

    if current_valid_len is None:
        valid_len = torch.full((batch_size,), max_seq_len, device=device, dtype=torch.long)
    else:
        valid_len = current_valid_len.to(device=device, dtype=torch.long)
    valid_len = valid_len.clamp(min=1, max=max_seq_len)

    if valid_target_len is None:
        num_target = torch.full((batch_size,), target_len, device=device, dtype=torch.long)
    else:
        num_target = valid_target_len.to(device=device, dtype=torch.long)
    num_target = num_target.clamp(min=1, max=target_len)

    # Per (sample, target step) bracketing indices into the source frames.
    steps = torch.arange(target_len, device=device).unsqueeze(0)            # [1, T]
    valid_mask = steps < num_target.unsqueeze(1)                            # [B, T]

    # Normalised target time in [0, 1]; guard the single-target-frame case.
    denom = (num_target - 1).clamp(min=1).float().unsqueeze(1)             # [B, 1]
    t = (steps.float() / denom).clamp(max=1.0)                             # [B, T]

    pos = t * (valid_len - 1).float().unsqueeze(1)                         # [B, T]
    prev_idx = pos.floor().long()
    next_idx = prev_idx + 1
    max_idx = (valid_len - 1).unsqueeze(1)
    prev_idx = prev_idx.clamp(min=0).minimum(max_idx)
    next_idx = next_idx.clamp(min=0).minimum(max_idx)
    alpha = pos - prev_idx.float()                                         # [B, T]

    translations = batch_trajectory[..., :3, 3]                           # [B, S, 3]
    quats = matrix_to_quaternion(batch_trajectory[..., :3, :3])           # [B, S, 4]

    trans_prev = torch.gather(translations, 1, prev_idx.unsqueeze(-1).expand(-1, -1, 3))
    trans_next = torch.gather(translations, 1, next_idx.unsqueeze(-1).expand(-1, -1, 3))
    trans_out = trans_prev + alpha.unsqueeze(-1) * (trans_next - trans_prev)

    q_prev = torch.gather(quats, 1, prev_idx.unsqueeze(-1).expand(-1, -1, 4))
    q_next = torch.gather(quats, 1, next_idx.unsqueeze(-1).expand(-1, -1, 4))
    rot_out = quaternion_to_matrix(_batched_slerp(q_prev, q_next, alpha))  # [B, T, 3, 3]

    resampled_batch = torch.zeros(
        (batch_size, target_len, 4, 4), device=device, dtype=batch_trajectory.dtype
    )
    resampled_batch[..., :3, :3] = rot_out
    resampled_batch[..., :3, 3] = trans_out
    # Zero out padded steps, then restore the homogeneous 1 on every row.
    resampled_batch = resampled_batch * valid_mask[..., None, None].to(resampled_batch.dtype)
    resampled_batch[..., 3, 3] = 1.0

    return resampled_batch, ~valid_mask
