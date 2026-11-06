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
    "relative_rotation_error_deg",
    "rotation_step_error_deg",
    "generated_rotation_path_deg",
    "target_rotation_path_deg",
    "rotation_path_error_deg",
)


def rotation_distance_deg(left, right):
    """Stable SO(3) geodesic distance, including equal and half-turn rotations."""
    relative = left.transpose(-1, -2) @ right
    skew = torch.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
    ), dim=-1)
    sine = torch.linalg.vector_norm(skew, dim=-1) / 2
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.atan2(sine, cosine))


def conditioning_pose_score(means, motion_score_weight=0.0):
    """Absolute pose score plus optional motion error; all terms minimize.

    Motion compares SO(3) rotations relative to each clip's own first pose,
    adjacent rotation increments and total angular distance. Path error is
    averaged over clips rather than frames, so static predictions cannot hide
    their missing motion behind small per-frame increments. A singleton clip
    supplies absolute pose metrics but cannot supply motion evidence.
    """
    score = means['position_error_normalized'] + means['rotation_error_deg'] / 180.
    if motion_score_weight:
        names = ('step_error_normalized', 'relative_rotation_error_deg',
                 'rotation_step_error_deg', 'rotation_path_error_deg')
        if not all(name in means for name in names):
            raise RuntimeError('Motion-aware validation requires at least one valid adjacent frame pair')
        motion = means['step_error_normalized'] + sum(means[name] / 180. for name in names[1:]) / 3.
        score = score + motion_score_weight * motion
    return score


def deterministic_caption_shuffle(caption, presence_mask, seed):
    """Derange batch captions and their presence masks without touching RNG.

    Returns None for singleton batches: a caption compared to itself is not a
    shuffled-caption diagnostic. The tokens use [fields, batch, features].
    """
    if caption.ndim != 3:
        raise ValueError('Caption shuffle requires [fields, batch, features] tokens')
    size = caption.shape[1]
    if presence_mask is not None and (
        presence_mask.dtype != torch.bool or presence_mask.shape != (size, caption.shape[0])
    ):
        raise ValueError('Caption presence_mask must be boolean [batch, fields]')
    if size < 2:
        return None
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(size, generator=generator)
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)
    shuffled = caption.index_select(1, permutation.to(caption.device))
    shuffled_mask = (None if presence_mask is None else
                     presence_mask.index_select(0, permutation.to(presence_mask.device)))
    return shuffled, shuffled_mask


def pose_metric_totals(generated, target, padding_mask=None, known_mask=None):
    """Return float64 ``[sum, count]`` tensors for frame/step-weighted means.

    Poses are ``[batch, frames, 6]`` with normalized XYZ positions and Euler XYZ
    rotations in radians. True padding is excluded from all metrics; true known
    entries select constrained frames. A step requires two adjacent valid frames,
    and a boundary step additionally crosses between known and hidden frames.
    Angular path statistics are clip-weighted and require a valid adjacent pair;
    all other rotation metrics are frame- or step-weighted. Relative rotations
    use each clip's first valid pose, excluding that trivial zero-error frame.
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
    rotation_error = rotation_distance_deg(target_rotation, generated_rotation)

    position_residual = generated[..., :3] - target[..., :3]
    step_error = torch.linalg.vector_norm(
        position_residual[:, 1:] - position_residual[:, :-1], dim=-1
    )
    valid_step = valid[:, 1:] & valid[:, :-1]
    boundary_step = valid_step & (known[:, 1:] ^ known[:, :-1])
    generated_step = generated_rotation[:, :-1].transpose(-1, -2) @ generated_rotation[:, 1:]
    target_step = target_rotation[:, :-1].transpose(-1, -2) @ target_rotation[:, 1:]
    rotation_step_error = rotation_distance_deg(target_step, generated_step)
    generated_path = rotation_distance_deg(generated_rotation[:, :-1], generated_rotation[:, 1:])
    target_path = rotation_distance_deg(target_rotation[:, :-1], target_rotation[:, 1:])
    generated_path = generated_path.masked_fill(~valid_step, 0).sum(dim=1)
    target_path = target_path.masked_fill(~valid_step, 0).sum(dim=1)
    valid_clip = valid_step.any(dim=1)

    # Empty frame dimensions are allowed by the existing additive contract.
    relative_valid = valid.clone()
    relative_error = torch.zeros_like(rotation_error)
    if generated.shape[1]:
        first = valid.to(torch.int64).argmax(dim=1)
        rows = torch.arange(len(generated), device=generated.device)
        generated_relative = generated_rotation[rows, first, None].transpose(-1, -2) @ generated_rotation
        target_relative = target_rotation[rows, first, None].transpose(-1, -2) @ target_rotation
        relative_error = rotation_distance_deg(target_relative, generated_relative)
        relative_valid[rows, first] = False
    relative_valid &= valid_clip[:, None]

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
        "relative_rotation_error_deg": total(relative_error, relative_valid),
        "rotation_step_error_deg": total(rotation_step_error, valid_step),
        "generated_rotation_path_deg": total(generated_path, valid_clip),
        "target_rotation_path_deg": total(target_path, valid_clip),
        "rotation_path_error_deg": total((generated_path - target_path).abs(), valid_clip),
    }


def rotation_baseline_totals(target, padding_mask=None):
    """Rotation-only baselines in SIM coordinates, explicitly labeling leakage.

    SIM zero is a fixed identity orientation independent of the clip. The
    static-first-pose baseline reads the reference first orientation and is an
    oracle diagnostic, never a fair text-only predictor. Neither baseline's
    position metrics are returned because positions are not being predicted.
    """
    fixed = torch.zeros_like(target)
    oracle = torch.zeros_like(target)
    if target.shape[1]:
        valid = (torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
                 if padding_mask is None else ~padding_mask.to(target.device))
        first = valid.to(torch.int64).argmax(dim=1)
        rows = torch.arange(len(target), device=target.device)
        first_rotation = target[rows, first, 3:]
        first_rotation = first_rotation.masked_fill(~valid.any(dim=1)[:, None], 0)
        oracle[..., 3:] = first_rotation[:, None]
    result = {}
    for name, prediction in (('sim_zero', fixed), ('static_first_pose_oracle', oracle)):
        result[name] = {
            key: value for key, value in pose_metric_totals(prediction, target, padding_mask).items()
            if 'rotation' in key and not key.startswith(('known_', 'hidden_'))
        }
    return result
