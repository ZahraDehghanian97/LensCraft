"""Additive pose-validation metrics for each conditioning mode."""

import torch

from utils.pytorch3d_transform import euler_angles_to_matrix


METRIC_NAMES = (
    "position_error_normalized",
    "rotation_error_deg",
    "known_position_error_normalized",
    "known_rotation_error_deg",
    "hidden_position_error_normalized",
    "hidden_rotation_error_deg",
    "step_error_normalized",
    "boundary_step_error_normalized",
)


def pose_metric_totals(generated, target, padding_mask=None, known_mask=None):
    """Return float64 ``[sum, count]`` tensors for frame/step-weighted means.

    Poses are ``[batch, frames, 6]`` with normalized XYZ positions and Euler XYZ
    rotations in radians. True padding is excluded from all metrics; true known
    entries select constrained frames. A step requires two adjacent valid frames,
    and a boundary step additionally crosses between known and hidden frames.
    No division is performed so callers can aggregate unequal batches/ranks.
    Invalid valid poses raise instead of making a failed model look better.
    """
    if (not torch.is_tensor(generated) or not torch.is_tensor(target)
            or generated.ndim != 3 or generated.shape[-1] != 6
            or generated.shape != target.shape):
        raise ValueError("generated and target must have matching [batch, frames, 6] shapes")
    if generated.device != target.device:
        raise ValueError("generated and target must be on the same device")
    if generated.is_complex() or target.is_complex():
        raise ValueError("generated and target must contain real pose values")

    def mask_or_default(mask, name):
        if mask is None:
            return torch.zeros(generated.shape[:2], dtype=torch.bool, device=generated.device)
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                or mask.shape != generated.shape[:2]):
            raise ValueError(f"{name} must be a boolean [batch, frames] tensor")
        return mask.to(device=generated.device)

    valid = ~mask_or_default(padding_mask, "padding_mask")
    known = mask_or_default(known_mask, "known_mask") & valid
    generated = generated.detach().to(dtype=torch.float64)
    target = target.detach().to(dtype=torch.float64)
    for name, poses in (("generated", generated), ("target", target)):
        if (valid & ~torch.isfinite(poses).all(dim=-1)).any():
            raise ValueError(f"{name} contains non-finite pose values at valid frames")

    # Sanitize before differences or trigonometry; multiplying a padded NaN by
    # zero after computing an error would still contaminate the accumulator.
    generated = torch.where(valid[..., None], generated, torch.zeros_like(generated))
    target = torch.where(valid[..., None], target, torch.zeros_like(target))
    position_error = torch.linalg.vector_norm(generated[..., :3] - target[..., :3], dim=-1)
    generated_rotation = euler_angles_to_matrix(generated[..., 3:], "XYZ")
    target_rotation = euler_angles_to_matrix(target[..., 3:], "XYZ")
    relative = target_rotation.transpose(-1, -2) @ generated_rotation
    skew = torch.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
    ), dim=-1)
    sine = torch.linalg.vector_norm(skew, dim=-1) / 2
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).clamp(-1, 1)
    rotation_error = torch.rad2deg(torch.atan2(sine, cosine))

    position_residual = generated[..., :3] - target[..., :3]
    step_error = torch.linalg.vector_norm(
        position_residual[:, 1:] - position_residual[:, :-1], dim=-1
    )
    valid_step = valid[:, 1:] & valid[:, :-1]
    boundary_step = valid_step & (known[:, 1:] ^ known[:, :-1])

    def total(values, eligible):
        return torch.stack((values.masked_select(eligible).sum(), eligible.sum(dtype=torch.float64)))

    hidden = valid & ~known
    return {
        "position_error_normalized": total(position_error, valid),
        "rotation_error_deg": total(rotation_error, valid),
        "known_position_error_normalized": total(position_error, known),
        "known_rotation_error_deg": total(rotation_error, known),
        "hidden_position_error_normalized": total(position_error, hidden),
        "hidden_rotation_error_deg": total(rotation_error, hidden),
        "step_error_normalized": total(step_error, valid_step),
        "boundary_step_error_normalized": total(step_error, boundary_step),
    }
