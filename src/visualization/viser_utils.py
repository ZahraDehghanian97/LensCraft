from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from utils.pytorch3d_transform import matrix_to_quaternion

_NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "gt": (60, 200, 90),
    "orig": (60, 200, 90),
    "reconstruction": (70, 130, 240),
    "prompt_generation": (240, 150, 40),
    "key_framing": (165, 95, 220),
    "key_framing+prompt": (40, 200, 200),
    "hybrid_generation": (40, 200, 200),
    "source_trajectory": (235, 95, 165),
}
_CYCLE_COLORS: List[Tuple[int, int, int]] = [
    (70, 130, 240), (240, 150, 40), (165, 95, 220),
    (40, 200, 200), (235, 95, 165), (210, 200, 60),
]

ROUNDTRIP_COLOR: Tuple[int, int, int] = (235, 70, 70)
SUBJECT_COLOR: Tuple[int, int, int] = (150, 150, 150)


def color_for(name: str, index: int = 0) -> Tuple[int, int, int]:
    """Pick a stable color for a named trajectory, falling back to a cycle."""
    key = name.lower()
    for known, color in _NAMED_COLORS.items():
        if known in key:
            return color
    return _CYCLE_COLORS[index % len(_CYCLE_COLORS)]


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


_OPENGL_TO_VISER = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _rotations_for_viser(rot: np.ndarray, cam_convention: str) -> np.ndarray:
    if cam_convention == "opencv":
        return rot
    if cam_convention == "opengl":
        return rot @ _OPENGL_TO_VISER
    raise ValueError(
        f"Unknown cam_convention '{cam_convention}' (use 'opencv' or 'opengl')."
    )


def matrix_to_wxyz(rotation) -> np.ndarray:
    """``[..., 3, 3]`` rotation -> ``[..., 4]`` (w, x, y, z) quaternion."""
    r = np.ascontiguousarray(to_numpy(rotation).astype(np.float32))
    squeeze = r.ndim == 2
    if squeeze:
        r = r[None]
    q = matrix_to_quaternion(torch.from_numpy(r)).numpy()
    return q[0] if squeeze else q


def transforms_to_wxyz_position(
    transforms, cam_convention: str = "opencv"
) -> Tuple[np.ndarray, np.ndarray]:
    """``[T, 4, 4]`` -> (wxyz ``[T, 4]``, position ``[T, 3]``) for viser."""
    mats = to_numpy(transforms).astype(np.float32)
    rot = _rotations_for_viser(mats[..., :3, :3], cam_convention)
    pos = np.nan_to_num(mats[..., :3, 3])
    wxyz = matrix_to_wxyz(rot)
    return wxyz, pos


def trajectory_diagonal(transforms) -> float:
    pos = to_numpy(transforms)[..., :3, 3].reshape(-1, 3)
    if pos.shape[0] == 0:
        return 1.0
    extent = pos.max(axis=0) - pos.min(axis=0)
    diag = float(np.linalg.norm(extent))
    return diag if diag > 1e-6 else 1.0


def auto_frustum_scale(all_transforms: Sequence) -> float:
    diags = [trajectory_diagonal(t) for t in all_transforms if t is not None]
    return 0.05 * (max(diags) if diags else 1.0)


def subject_dims(volume) -> np.ndarray:
    if volume is None:
        return np.array([0.5, 1.7, 0.3], dtype=np.float32)
    v = to_numpy(volume).reshape(-1)[:3].astype(np.float32)
    return np.maximum(v, 1e-3)


def add_path(server, name, transforms, color, line_width: float = 3.0):
    pos = np.nan_to_num(to_numpy(transforms)[..., :3, 3].reshape(-1, 3)).astype(np.float32)
    if pos.shape[0] < 2:
        return None
    return server.scene.add_spline_catmull_rom(
        name, positions=pos, color=color, line_width=line_width
    )


def add_frustums(
    server, base_name, transforms, color, fov, aspect, scale,
    stride: int = 1, cam_convention: str = "opencv", line_width: float = 1.5,
):
    wxyz, pos = transforms_to_wxyz_position(transforms, cam_convention)
    handles = []
    for i in range(0, pos.shape[0], max(1, int(stride))):
        handles.append(
            server.scene.add_camera_frustum(
                f"{base_name}/f{i:04d}",
                fov=fov, aspect=aspect, scale=scale, color=color,
                wxyz=tuple(float(v) for v in wxyz[i]),
                position=tuple(float(v) for v in pos[i]),
                line_width=line_width,
            )
        )
    return handles


def add_subject_box(server, name, subject_transform_frame, dims, color=SUBJECT_COLOR):
    mat = to_numpy(subject_transform_frame).astype(np.float32)
    wxyz = matrix_to_wxyz(mat[:3, :3])
    pos = np.nan_to_num(mat[:3, 3])
    return server.scene.add_box(
        name, color=color,
        dimensions=tuple(float(v) for v in np.maximum(dims, 1e-3)),
        wxyz=tuple(float(v) for v in wxyz),
        position=tuple(float(v) for v in pos),
    )


def pose_error(transforms_a, transforms_b) -> Dict[str, float]:
    a = to_numpy(transforms_a).reshape(-1, 4, 4)
    b = to_numpy(transforms_b).reshape(-1, 4, 4)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return {"frames": 0.0, "pos_mean": 0.0, "pos_max": 0.0,
                "rot_mean_deg": 0.0, "rot_max_deg": 0.0}
    a, b = a[:n], b[:n]
    pos_err = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=-1)
    rel = np.matmul(np.transpose(a[:, :3, :3], (0, 2, 1)), b[:, :3, :3])
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = np.degrees(np.arccos(cos))
    return {
        "frames": float(n),
        "pos_mean": float(np.mean(pos_err)),
        "pos_max": float(np.max(pos_err)),
        "rot_mean_deg": float(np.mean(rot_err)),
        "rot_max_deg": float(np.max(rot_err)),
    }
