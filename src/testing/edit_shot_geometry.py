"""Independent shot-edit diagnostics using the simulator's exact ROI contract.

World poses are ``[x, y, z, rx, ry, rz]`` with Three.js intrinsic XYZ Euler
angles in radians. Camera-to-world rotations look along local -Z, with +Y up.
Subject dimensions describe a centered, oriented box. Attention-box positions
are subject-local offsets; without an authored attention box the simulator
uses half the body dimensions and offset ``(0, body_height / 4, 0)``.

The ROI interpolation and NDC occupancy targets below mirror
``simulation/rule-based/setup/roi.ts`` and ``dataset/finalSetup.ts``. These
measure extrinsic framing under the source clip's lens calibration, not zoom.
They are continuous target errors, not a calibrated shot-class success test.
"""

from itertools import product

import numpy as np

from testing.metrics.modules.geometry import _xyz_matrix


# (body interpolation weight, target maximum NDC box dimension).
_SHOT_ROI = {
    "extremeCloseUp": (0.0, 2.0),
    "closeUp": (0.0, 1.0),
    "mediumCloseUp": (0.25, 1.0),
    "mediumShot": (0.5, 1.0),
    "fullShot": (1.0, 1.0),
    "longShot": (1.0, 0.75),
    "veryLongShot": (1.0, 0.5),
    "extremeLongShot": (1.0, 0.3),
}

_METRICS = (
    "initial_shot_target_log_occupancy_error",
    "reconstruction_initial_shot_target_log_occupancy_error",
    "initial_shot_target_log_occupancy_improvement",
    "initial_shot_target_log_occupancy_improved_rate",
    "initial_shot_geometry_unavailable_rate",
    "initial_shot_geometry_invalid_rate",
)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _metadata(value, shape, width, name):
    array = _array(value)
    if array.shape == (shape[0], width):
        array = array[:, None, :]
    try:
        return np.broadcast_to(array, shape + (width,))
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to {shape + (width,)}, got {array.shape}"
        ) from error


def _occupancy(camera, center, dimensions, rotation, intrinsics):
    """Maximum projected NDC dimension, with simulator geometric depth guard."""
    signs = np.asarray(list(product((-1.0, 1.0), repeat=3)))
    corners = np.einsum(
        "bij,bkj->bki", rotation, dimensions[:, None, :] * signs / 2,
    ) + center[:, None, :]
    local = np.einsum(
        "bki,bij->bkj", corners - camera[:, None, :3],
        _xyz_matrix(camera[:, 3:]),
    )
    depth = -local[..., 2]
    # Match projectBoundingBox: geometric visibility must also work for tiny
    # scenes; a renderer's fixed 0.1 near clip changes the shot-size contract.
    coordinate_scale = np.max(np.abs(np.concatenate(
        (dimensions, center, camera[:, :3]), axis=-1,
    )), axis=-1)
    depth_epsilon = 64 * np.finfo(np.float64).eps * coordinate_scale
    in_front = np.all(depth > depth_epsilon[:, None], axis=-1)
    focal, aspect = intrinsics[:, 0], intrinsics[:, 1]
    # Three.js filmGauge=35: the long sensor edge is 35 mm in portrait too.
    tan_half_vertical = 35 / np.maximum(aspect, 1) / (2 * focal)
    scale = np.stack((tan_half_vertical * aspect, tan_half_vertical), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ndc = local[..., :2] / (depth[..., None] * scale[:, None, :])
        size = np.max(ndc, axis=1) - np.min(ndc, axis=1)
        occupancy = np.max(size, axis=-1)
    usable = in_front & np.isfinite(occupancy) & (occupancy > 0)
    return occupancy, usable


def shot_size_geometry_totals(
    generated,
    reconstruction,
    *,
    subject_trajectory,
    subject_dimensions,
    intrinsics,
    valid_mask,
    shot_size,
    attention_dimensions=None,
    attention_position=None,
):
    """Return sums/counts for the first valid frame of each world-space clip.

    Camera inputs are matching ``[B,T,6]`` arrays. Subject poses, dimensions,
    and intrinsics broadcast to ``[B,T,6/3/2]``; ``[B,D]`` also means constant
    within each clip. Optional attention arrays are local dimensions/offsets,
    normally ``[B,3]``. Supply both or neither to use the simulator default.

    Error is ``abs(log(max(NDC_bbox_width, NDC_bbox_height) / target))``.
    Improvement is reconstruction error minus generated error; its rate only
    asks whether this continuous error decreased, with no success threshold.
    Each error has its own eligibility count; improvement requires both poses.
    Missing/nonfinite metadata contributes to unavailable rate. Nonpositive
    dimensions/calibration, invalid camera poses, and boxes touching/behind the
    camera contribute to invalid rate. Both status rates count clips, and fully
    padded clips contribute to no metric. Missing geometry never scores zero.
    """
    shot_size = getattr(shot_size, "value", shot_size)
    if shot_size not in _SHOT_ROI:
        raise ValueError(f"Unknown shot_size: {shot_size!r}")
    if (attention_dimensions is None) != (attention_position is None):
        raise ValueError("Supply both attention_dimensions and attention_position")
    generated, reconstruction = _array(generated), _array(reconstruction)
    if (generated.ndim != 3 or generated.shape[-1] != 6
            or generated.shape != reconstruction.shape):
        raise ValueError("generated and reconstruction need matching [B,T,6] shapes")
    shape = generated.shape[:2]
    if valid_mask is None:
        valid = np.ones(shape, dtype=bool)
    else:
        if hasattr(valid_mask, "detach"):
            valid_mask = valid_mask.detach().cpu().numpy()
        valid = np.asarray(valid_mask)
        if valid.shape != shape or valid.dtype != np.bool_:
            raise ValueError(f"valid_mask must be boolean with shape {shape}")
    totals = {name: (0.0, 0) for name in _METRICS}
    samples = np.flatnonzero(valid.any(axis=1))
    if not len(samples):
        return totals
    frame = valid[samples].argmax(axis=1)
    if any(value is None for value in (subject_trajectory, subject_dimensions, intrinsics)):
        totals["initial_shot_geometry_unavailable_rate"] = (float(len(samples)), len(samples))
        return totals

    def first(value, width, name):
        return _metadata(value, shape, width, name)[samples, frame]

    subject = first(subject_trajectory, 6, "subject_trajectory")
    body = first(subject_dimensions, 3, "subject_dimensions")
    lens = first(intrinsics, 2, "intrinsics")
    if attention_dimensions is None:
        attention = body * 0.5
        offset = np.zeros_like(body)
        offset[:, 1] = body[:, 1] * 0.25
    else:
        attention = first(attention_dimensions, 3, "attention_dimensions")
        offset = first(attention_position, 3, "attention_position")
    available = np.isfinite(np.concatenate((subject, body, lens, attention, offset), axis=-1)).all(axis=-1)
    totals["initial_shot_geometry_unavailable_rate"] = (
        float((~available).sum()), len(samples),
    )
    available_indices = np.flatnonzero(available)
    if not len(available_indices):
        return totals
    subject, body, lens, attention, offset = (
        value[available_indices] for value in (subject, body, lens, attention, offset)
    )
    edited = generated[samples, frame][available_indices]
    before = reconstruction[samples, frame][available_indices]
    geometry_valid = (body > 0).all(axis=-1) & (attention > 0).all(axis=-1) & (lens > 0).all(axis=-1)
    weight, target = _SHOT_ROI[shot_size]
    rotation = _xyz_matrix(subject[:, 3:])
    dimensions = weight * body + (1 - weight) * attention
    center = subject[:, :3] + (1 - weight) * np.einsum("bij,bj->bi", rotation, offset)
    # Neutralize invalid rows before projection; their eligibility remains off.
    safe_lens = np.where(lens > 0, lens, 1.0)
    errors, usable = [], []
    for camera in (edited, before):
        finite = np.isfinite(camera).all(axis=-1)
        occupancy, projected = _occupancy(
            np.where(np.isfinite(camera), camera, 0), center, dimensions, rotation, safe_lens,
        )
        eligible = geometry_valid & finite & projected
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            error = np.abs(np.log(occupancy / target))
        eligible &= np.isfinite(error)
        errors.append(error)
        usable.append(eligible)
    after_error, before_error = errors
    after_valid, before_valid = usable

    def add(name, values, eligible):
        selected = values[eligible]
        totals[name] = (float(np.sum(selected)), int(selected.size))

    add(_METRICS[0], after_error, after_valid)
    add(_METRICS[1], before_error, before_valid)
    paired = after_valid & before_valid
    improvement = before_error - after_error
    add(_METRICS[2], improvement, paired)
    add(_METRICS[3], improvement > 0, paired)
    totals["initial_shot_geometry_invalid_rate"] = (
        float((~(after_valid & before_valid)).sum()), len(available_indices),
    )
    return totals
