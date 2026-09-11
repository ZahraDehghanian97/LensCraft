"""Evaluator-independent camera and framing metrics in simulation coordinates.

Poses are denormalized ``[..., x, y, z, rx, ry, rz]``: rotations are intrinsic
XYZ Euler angles in radians, camera-to-world, looking along local -Z with +Y up.
Subject dimensions are local (width, height, depth), centered on its pose, as in
the simulator's VBox. Projection uses all eight rotated box corners. Coordinates
and sizes are fractions of image width/height (origin bottom-left, no clipping).

No projection calibration is assumed. Supply per-frame (focal length in mm,
aspect ratio), using the simulator's 35 mm film gauge, OR explicitly supply a
vertical FOV and aspect ratio. Shared intrinsics measure the generated extrinsic
trajectory under the reference calibration; they do not evaluate optical zoom.
"""

from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np


_POSE_METRICS = (
    "position_error", "rotation_error_deg",
    "keyframe_position_error", "keyframe_rotation_error_deg",
    "hidden_position_error", "hidden_rotation_error_deg",
    "invalid_generated_pose_rate",
)
_PROJECTION_METRICS = (
    "bbox_size_error", "bbox_center_error",
    "subject_bbox_width", "subject_bbox_height",
    "subject_bbox_center_x", "subject_bbox_center_y",
    "out_of_frame_rate", "behind_camera_rate", "near_plane_violation_rate",
)


def _array(value, dtype=np.float64):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _frame_array(value, shape, trailing, name):
    array = _array(value)
    if array.shape == (shape[0], trailing):
        array = array[:, None, :]
    try:
        return np.broadcast_to(array, shape + (trailing,))
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to {shape + (trailing,)}, got {array.shape}"
        ) from error


def _mask(value, shape, name):
    array = _array(value, bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _xyz_matrix(angles):
    """Rx @ Ry @ Rz, matching Three.js Euler('XYZ') and dataset conversion."""
    x, y, z = np.moveaxis(angles, -1, 0)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    return np.stack((
        cy * cz, -cy * sz, sy,
        cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy,
        sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy,
    ), axis=-1).reshape(angles.shape[:-1] + (3, 3))


def _project_box(camera, corners, tan_half_vertical_fov, aspect, near_clip):
    # A row vector transformed by R_cw is equivalent to R_cw.T @ column.
    local = np.einsum(
        "...ki,...ij->...kj", corners - camera[..., None, :3],
        _xyz_matrix(camera[..., 3:]),
    )
    depth = -local[..., 2]
    in_front = np.all(depth > near_clip, axis=-1)
    behind = np.any(depth <= 0, axis=-1)
    safe_depth = np.where(depth > near_clip, depth, np.nan)
    scale = np.stack((tan_half_vertical_fov * aspect, tan_half_vertical_fov), -1)
    image_xy = 0.5 + local[..., :2] / (2 * safe_depth[..., None] * scale[..., None, :])
    lower, upper = np.min(image_xy, axis=-2), np.max(image_xy, axis=-2)
    size, center = upper - lower, (lower + upper) / 2
    outside = (~in_front) | np.any((lower < 0) | (upper > 1), axis=-1)
    return size, center, in_front, outside, behind


def geometry_metric_totals(
    generated,
    target,
    *,
    valid_mask=None,
    known_mask=None,
    subject_trajectory=None,
    subject_dimensions=None,
    intrinsics=None,
    vertical_fov_deg=None,
    aspect_ratio=None,
    near_clip: float = 0.1,
    film_gauge_mm: float = 35.0,
) -> Dict[str, Tuple[float, int]]:
    """Return sums and frame counts for unbiased aggregation over batches.

    Camera tensors have shape (B,T,6). ``valid_mask`` and ``known_mask`` are
    (B,T) booleans: True means a real frame and a constrained keyframe,
    respectively. Without known_mask, constrained/hidden metrics are absent
    (count zero), rather than treating an unknown conditioning regime as K=0.
    Positions use the dataset's world units; angular errors are SO(3) geodesics
    in degrees. All means are weighted by eligible frames, not by batches.

    Subject poses, dimensions, and intrinsics broadcast to (B,T,6), (B,T,3),
    and (B,T,2); (B,D) also denotes a time-constant value for each sample.
    Invalid/missing calibration is excluded, with zero counts when absent.
    Bbox error requires both boxes fully in front of the near plane. Generated
    boxes crossing/behind that plane still count as out-of-frame violations;
    ``behind_camera_rate`` marks any corner at/behind the camera plane.
    Missing geometry never becomes a perfect zero score.
    """
    generated, target = _array(generated), _array(target)
    if generated.ndim != 3 or generated.shape[-1] != 6 or generated.shape != target.shape:
        raise ValueError("generated and target must have matching (B,T,6) shapes")
    if not np.isfinite(near_clip) or near_clip <= 0:
        raise ValueError("near_clip must be finite and positive")
    if not np.isfinite(film_gauge_mm) or film_gauge_mm <= 0:
        raise ValueError("film_gauge_mm must be finite and positive")
    if intrinsics is not None and (vertical_fov_deg is not None or aspect_ratio is not None):
        raise ValueError("Supply intrinsics OR an explicit vertical FOV/aspect, not both")
    if (vertical_fov_deg is None) != (aspect_ratio is None):
        raise ValueError("Explicit projection requires both vertical_fov_deg and aspect_ratio")

    shape = generated.shape[:2]
    valid = np.ones(shape, dtype=bool) if valid_mask is None else _mask(valid_mask, shape, "valid_mask")
    totals = {name: (0.0, 0) for name in _POSE_METRICS + _PROJECTION_METRICS}

    def add(name, values, eligible):
        selected = np.asarray(values)[eligible]
        totals[name] = (float(np.sum(selected)), int(selected.size))

    generated_finite = np.isfinite(generated).all(axis=-1)
    pose_valid = valid & generated_finite & np.isfinite(target).all(axis=-1)
    add("invalid_generated_pose_rate", ~generated_finite, valid)
    # Invalid padded poses are neutralized before trigonometry/projection.
    generated = np.where(np.isfinite(generated), generated, 0.0)
    target = np.where(np.isfinite(target), target, 0.0)
    position_error = np.linalg.norm(generated[..., :3] - target[..., :3], axis=-1)
    gen_rotation, ref_rotation = _xyz_matrix(generated[..., 3:]), _xyz_matrix(target[..., 3:])
    relative = np.swapaxes(ref_rotation, -1, -2) @ gen_rotation
    # atan2 is stable at both identical and 180-degree rotations, unlike acos.
    sine = np.linalg.norm(np.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
    ), -1), axis=-1) / 2
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    angular_error = np.rad2deg(np.arctan2(sine, np.clip(cosine, -1, 1)))
    add("position_error", position_error, pose_valid)
    add("rotation_error_deg", angular_error, pose_valid)
    if known_mask is not None:
        known = _mask(known_mask, shape, "known_mask")
        for prefix, subset in (("keyframe", known), ("hidden", ~known)):
            add(f"{prefix}_position_error", position_error, pose_valid & subset)
            add(f"{prefix}_rotation_error_deg", angular_error, pose_valid & subset)

    if subject_trajectory is None or subject_dimensions is None:
        return totals
    if intrinsics is None and vertical_fov_deg is None:
        return totals

    subject = _frame_array(subject_trajectory, shape, 6, "subject_trajectory")
    dimensions = _frame_array(subject_dimensions, shape, 3, "subject_dimensions")
    projection_valid = (
        valid & generated_finite & np.isfinite(subject).all(axis=-1)
        & np.isfinite(dimensions).all(axis=-1) & (dimensions > 0).all(axis=-1)
    )
    if intrinsics is not None:
        calibration = _frame_array(intrinsics, shape, 2, "intrinsics")
        focal, aspect = calibration[..., 0], calibration[..., 1]
        calibration_valid = np.isfinite(calibration).all(axis=-1) & (calibration > 0).all(axis=-1)
        safe_focal = np.where(calibration_valid, focal, 1.0)
        aspect = np.where(calibration_valid, aspect, 1.0)
        tan_half_fov = film_gauge_mm / np.maximum(aspect, 1.0) / (2 * safe_focal)
    else:
        def frame_scalar(value, name):
            array = _array(value)
            if array.shape == (shape[0],):
                array = array[:, None]
            try:
                return np.broadcast_to(array, shape)
            except ValueError as error:
                raise ValueError(f"{name} must broadcast to {shape}") from error

        fov, aspect = frame_scalar(vertical_fov_deg, "vertical_fov_deg"), frame_scalar(aspect_ratio, "aspect_ratio")
        calibration_valid = np.isfinite(fov) & (fov > 0) & (fov < 180) & np.isfinite(aspect) & (aspect > 0)
        tan_half_fov = np.tan(np.deg2rad(np.where(calibration_valid, fov, 90.0)) / 2)
        aspect = np.where(calibration_valid, aspect, 1.0)
    projection_valid &= calibration_valid
    # Neutralize unavailable subject geometry before projection, too.
    subject = np.where(np.isfinite(subject), subject, 0.0)
    dimensions = np.where(np.isfinite(dimensions), dimensions, 0.0)
    signs = np.asarray(list(product((-1.0, 1.0), repeat=3)))
    corners = dimensions[..., None, :] * signs / 2
    corners = np.einsum("...ij,...kj->...ki", _xyz_matrix(subject[..., 3:]), corners)
    corners += subject[..., None, :3]
    gen_size, gen_center, gen_front, gen_outside, gen_behind = _project_box(
        generated, corners, tan_half_fov, aspect, near_clip,
    )
    ref_size, ref_center, ref_front, _, _ = _project_box(
        target, corners, tan_half_fov, aspect, near_clip,
    )
    paired = projection_valid & pose_valid & gen_front & ref_front
    add("bbox_size_error", np.linalg.norm(gen_size - ref_size, axis=-1), paired)
    add("bbox_center_error", np.linalg.norm(gen_center - ref_center, axis=-1), paired)
    for name, values in (
        ("subject_bbox_width", gen_size[..., 0]),
        ("subject_bbox_height", gen_size[..., 1]),
        ("subject_bbox_center_x", gen_center[..., 0]),
        ("subject_bbox_center_y", gen_center[..., 1]),
    ):
        add(name, values, projection_valid & gen_front)
    add("out_of_frame_rate", gen_outside, projection_valid)
    add("behind_camera_rate", gen_behind, projection_valid)
    add("near_plane_violation_rate", ~gen_front, projection_valid)
    return totals


class GeometryMetrics:
    """Small accumulator independent of learned evaluator/model weights."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.totals = {name: (0.0, 0) for name in _POSE_METRICS + _PROJECTION_METRICS}

    def update(self, generated, target, **kwargs):
        for name, (total, count) in geometry_metric_totals(generated, target, **kwargs).items():
            previous_total, previous_count = self.totals[name]
            self.totals[name] = (previous_total + total, previous_count + count)

    def compute(self) -> Dict[str, Optional[float]]:
        result = {}
        for name, (total, count) in self.totals.items():
            result[name] = total / count if count else None
            result[f"{name}_count"] = count
        return result
