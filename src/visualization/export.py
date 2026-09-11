"""Headless, publication-quality comparisons of camera trajectories.

Poses are OpenCV camera-to-world matrices in a world with Y pointing up.
Every method in a row uses exactly the same bounds and orthographic view.
Paths join the recorded positions directly; they are never interpolated.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import textwrap
from typing import Any, Sequence

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


GOLD = "#c89524"
INK = "#253044"
MUTED = "#687487"
SUBJECT = "#a4afbc"
_COLORS = ("#386bb5", "#b06490", "#238b86", "#dd8549", "#8872b2", "#a28551")
_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
          (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
          (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def method_label(name: str) -> str:
    """Readable labels, while preserving arbitrary user-supplied method names."""
    base, separator, variant = name.partition(":")
    label = {"gt": "Reference", "ground truth": "Reference", "ground_truth": "Reference",
             "lens_craft": "LensCraft", "lenscraft": "LensCraft", "ccdm": "CCDM",
             "et": "E.T.", "gendop": "GenDoP"}.get(base.lower(), base)
    return f"{label} · {variant.replace('_', ' ')}" if separator else label


def method_color(name: str, index: int = 0) -> str:
    base = name.partition(":")[0].lower()
    # A saved/custom method keeps its color when columns are selected/reordered.
    fallback = int.from_bytes(hashlib.sha256(base.encode()).digest()[:2], "big")
    return {"gt": "#6a788c", "reference": "#6a788c", "ground truth": "#6a788c",
            "ground_truth": "#6a788c", "lens_craft": "#386bb5", "lenscraft": "#386bb5",
            "ccdm": "#238b86", "et": "#b06490", "gendop": "#dd8549"}.get(
                base, _COLORS[fallback % len(_COLORS)])


def _as_samples(samples: Any) -> list[Any]:
    result = [samples] if hasattr(samples, "trajectories") else list(samples)
    if not result:
        raise ValueError("Select at least one sample to export.")
    return result


def _methods(samples: Sequence[Any], methods: Sequence[str] | None) -> list[str]:
    available = list(dict.fromkeys(name for sample in samples for name in sample.trajectories))
    selected = list(dict.fromkeys(methods)) if methods is not None else available
    if not selected:
        raise ValueError("Select at least one trajectory to export.")
    unknown = set(selected).difference(available)
    if unknown:
        raise ValueError("Unknown trajectory names: " + ", ".join(sorted(unknown)))
    return selected


def _poses(value: Any, label: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.shape == (4, 4):
        poses = poses[None]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError(f"{label} must have shape (T, 4, 4) with at least one pose.")
    if not np.isfinite(poses).all():
        raise ValueError(f"{label} contains a non-finite pose.")
    return poses


def keyframe_constraints(sample: Any, method: str | None = None) -> tuple[list[int], np.ndarray | None, str | None]:
    """Return actual conditioning poses, never points sampled from a prediction.

    ``metadata.keyframe_source`` can name a reference trajectory. Explicit
    ``metadata.keyframe_poses`` is also accepted for constraints without a full
    reference path. Without either, only known reference names are considered.
    For an output method, its ``metadata.runs[method]`` record is authoritative:
    prompt-only outputs receive no constraints, while historical keyframe runs
    retain their own input poses. The default and reference columns show the
    sample's current global inputs.
    """
    metadata = getattr(sample, "metadata", {}) or {}
    indices = getattr(sample, "keyframes", [])
    reference_names = {"gt", "reference", "ground truth", "ground_truth"}
    if method is not None and method.lower() not in reference_names:
        if method not in sample.trajectories:
            return [], None, None
        run = metadata.get("runs", {}).get(method)
        if isinstance(run, dict) and "keyframes" in run:
            indices = run["keyframes"]
            # A newer global input must not replace an older run's poses.
            source_metadata = {"keyframe_source": metadata["keyframe_source"]} if "keyframe_source" in metadata else {}
            metadata = {**source_metadata, **run}
        else:
            mode = run.get("mode", method.partition(":")[2]) if isinstance(run, dict) else method.partition(":")[2]
            illustrative = metadata.get("demo") and metadata.get("keyframes_illustrative")
            if mode not in {"key_framing", "key_framing+prompt"} and not illustrative:
                return [], None, None
    indices = [int(i) for i in indices]
    if not indices:
        return [], None, None
    if any(i < 0 for i in indices) or len(indices) != len(set(indices)):
        raise ValueError("Keyframe indices must be unique, non-negative frame numbers.")
    explicit = metadata.get("keyframe_poses")
    if explicit is not None:
        poses = _poses(explicit, "Keyframe constraints")
        if len(poses) != len(indices):
            raise ValueError("The number of keyframe poses must match the keyframe indices.")
        return indices, poses, str(metadata.get("keyframe_source", "explicit input poses"))
    source = metadata.get("keyframe_source")
    if source is None:
        source = next((name for name in sample.trajectories
                       if name.lower() in reference_names), None)
    if source is None or source not in sample.trajectories:
        return indices, None, None
    poses = _poses(sample.trajectories[source], str(source))
    if max(indices) >= len(poses):
        raise ValueError(f"Keyframe {max(indices)} is outside the {len(poses)}-frame source '{source}'.")
    return indices, poses[indices], str(source)


def _subject_corners(sample: Any) -> np.ndarray | None:
    if getattr(sample, "subject", None) is None or getattr(sample, "volume", None) is None:
        return None
    poses = _poses(sample.subject, "Subject")
    volume = np.asarray(sample.volume, dtype=float)
    if volume.size == 3:
        volume = np.broadcast_to(volume.reshape(1, 3), (len(poses), 3))
    elif volume.shape != (len(poses), 3):
        raise ValueError("Subject volume must contain three dimensions, or one set per subject pose.")
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError("Subject dimensions must be finite and non-negative.")
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    local = signs[None] * volume[:, None] / 2
    return np.einsum("tij,tkj->tki", poses[:, :3, :3], local) + poses[:, None, :3, 3]


def scene_bounds(sample: Any, methods: Sequence[str] | None = None,
                 *, show_keyframes: bool = True) -> np.ndarray:
    """Shared world-space limits as ``[[xmin,xmax], [ymin,ymax], [zmin,zmax]]``."""
    selected = list(methods) if methods is not None else list(sample.trajectories)
    points = [_poses(sample.trajectories[name], name)[:, :3, 3]
              for name in selected if name in sample.trajectories]
    if getattr(sample, "subject", None) is not None:
        points.append(_poses(sample.subject, "Subject")[:, :3, 3])
        corners = _subject_corners(sample)
        if corners is not None:
            points.append(corners.reshape(-1, 3))
    if show_keyframes:
        for method in [None, *selected]:
            _, constraints, _ = keyframe_constraints(sample, method)
            if constraints is not None:
                points.append(constraints[:, :3, 3])
    if not points:
        return np.array([[-1., 1.], [-1., 1.], [-1., 1.]])
    cloud = np.concatenate(points)
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
    span = np.ptp(cloud, axis=0)
    reference_span = max(float(span.max()), 1.)
    half_span = np.maximum(span / 2 + reference_span * .10, reference_span * .15)
    return np.column_stack((center - half_span, center + half_span))


def _xyz(points: np.ndarray) -> np.ndarray:
    # Matplotlib's vertical coordinate is Z; the dataset's is Y.
    return np.asarray(points)[..., [0, 2, 1]]


def _draw_frustums(ax: Any, poses: np.ndarray, scale: float, color: str,
                    *, alpha: float = .82, linewidth: float = .8) -> None:
    # OpenCV cameras look along +Z; image down is +Y.
    local = np.array([[0., 0., 0.], [-.65, -.4, 1.], [.65, -.4, 1.],
                      [.65, .4, 1.], [-.65, .4, 1.]]) * scale
    edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    world = np.einsum("tij,kj->tki", poses[:, :3, :3], local) + poses[:, None, :3, 3]
    lines = [_xyz(pose[[a, b]]) for pose in world for a, b in edges]
    ax.add_collection3d(Line3DCollection(lines, colors=color, linewidths=linewidth, alpha=alpha))


def _draw_context(ax: Any, sample: Any, bounds: np.ndarray) -> None:
    subject = getattr(sample, "subject", None)
    corners = _subject_corners(sample)
    ground = float(bounds[1, 0] + .05 * np.ptp(bounds[1]))
    if corners is not None:
        ground = float(corners[..., 1].min())
    # A restrained ground grid supplies scale and depth without tick labels.
    xlim, zlim = bounds[0], bounds[2]
    grid = [np.array([[x, zlim[0], ground], [x, zlim[1], ground]])
            for x in np.linspace(*xlim, 7)]
    grid += [np.array([[xlim[0], z, ground], [xlim[1], z, ground]])
             for z in np.linspace(*zlim, 7)]
    ax.add_collection3d(Line3DCollection(grid, colors="#e5e9ee", linewidths=.45, alpha=.8))
    if subject is None:
        return
    positions = _poses(subject, "Subject")[:, :3, 3]
    if len(positions) > 1 and np.ptp(positions, axis=0).max() > 1e-8:
        ax.plot(*_xyz(positions).T, color=SUBJECT, lw=1., ls=(0, (3, 3)), alpha=.9)
    if corners is None:
        ax.scatter(*_xyz(positions[[0]]).T, color=SUBJECT, s=22, marker="s", depthshade=False)
        return
    moving = np.ptp(corners, axis=0).max() > 1e-8
    indices = np.unique(np.linspace(0, len(corners) - 1, min(3, len(corners)), dtype=int)) if moving else [0]
    for i, frame in enumerate(indices):
        box = _xyz(corners[frame])
        opacity = .25 if i == 0 else .1
        ax.add_collection3d(Poly3DCollection([box[list(face)] for face in _FACES],
                                            facecolors=SUBJECT, alpha=opacity, edgecolors="none"))
        ax.add_collection3d(Line3DCollection([box[[a, b]] for a, b in _EDGES],
                                            colors=SUBJECT, linewidths=.65, alpha=.75 if i == 0 else .4))


def _draw_panel(ax: Any, sample: Any, method: str | None, color: str,
                bounds: np.ndarray, *, show_keyframes: bool, elev: float,
                azim: float, camera_count: int) -> dict[str, Any]:
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[2])
    ax.set_zlim(*bounds[1])
    ax.set_box_aspect(np.ptp(bounds, axis=1)[[0, 2, 1]], zoom=1.28)
    ax.set_axis_off()
    _draw_context(ax, sample, bounds)
    scale = float(np.ptp(bounds, axis=1).max()) * .045
    record: dict[str, Any] = {"method": method, "bounds_world_xyz": bounds.tolist()}
    if method is not None:
        trajectory = sample.trajectories.get(method)
        if trajectory is None:
            ax.text2D(.5, .48, "No output", transform=ax.transAxes, ha="center", color=MUTED, fontsize=10)
        else:
            poses = _poses(trajectory, method)
            positions = _xyz(poses[:, :3, 3])
            line, = ax.plot(*positions.T, color=color, lw=2., solid_capstyle="round")
            line.set_gid(f"trajectory:{sample.sample_id}:{method}")
            ax.scatter(*positions[0], marker="o", s=34, facecolors="white", edgecolors=color,
                       linewidths=1.5, depthshade=False, zorder=10)
            ax.scatter(*positions[-1], marker="D", s=26, color=color, edgecolors="white",
                       linewidths=.7, depthshade=False, zorder=11)
            camera_indices = np.unique(np.linspace(0, len(poses) - 1, min(camera_count, len(poses)), dtype=int))
            if len(camera_indices):
                _draw_frustums(ax, poses[camera_indices], scale, color)
            record.update(frame_count=len(poses), camera_indices=camera_indices.tolist(),
                          pose_sha256=hashlib.sha256(poses.astype("<f8").tobytes()).hexdigest())
    keyframes, constraints, source = keyframe_constraints(sample, method) if show_keyframes or method is None else ([], None, None)
    if constraints is not None:
        _draw_frustums(ax, constraints, scale * 1.2, GOLD, alpha=1., linewidth=1.25)
        ax.scatter(*_xyz(constraints[:, :3, 3]).T, s=65, marker="o", facecolors="none",
                   edgecolors=GOLD, linewidths=1.4, depthshade=False, zorder=20)
        if method is None:
            for frame, point in zip(keyframes, _xyz(constraints[:, :3, 3])):
                ax.text(*point, f"  {frame}", fontsize=8, color="#987014", zorder=21)
    elif method is None:
        message = "No input keyframes" if not keyframes else "Keyframe poses unavailable"
        ax.text2D(.5, .48, message, transform=ax.transAxes, ha="center", color=MUTED, fontsize=10)
    record.update(keyframe_indices=keyframes, keyframe_source=source)
    return record


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_figure(samples: Any, *, methods: Sequence[str] | None = None,
                 show_keyframes: bool = True, include_input: bool = True,
                 elev: float = 24., azim: float = -58., camera_count: int = 6,
                 title: str | None = None) -> Figure:
    """Build an editable Matplotlib figure, with one sample per row.

    The optional input column displays only actual conditioning poses and the
    subject. Gold markers in each conditioned output column remain at its
    actual input poses, so failures to satisfy constraints remain visible.
    Prompt-only columns have no input markers. Frame labels are zero based,
    matching the dataset indices.
    """
    samples = _as_samples(samples)
    methods = _methods(samples, methods)
    if not isinstance(camera_count, (int, np.integer)) or camera_count < 0:
        raise ValueError("camera_count must be a non-negative integer.")
    if not math.isfinite(elev) or not math.isfinite(azim):
        raise ValueError("View angles must be finite.")
    columns: list[str | None] = ([None] if include_input else []) + methods
    width = max(4., 3.35 * len(columns))
    prompt_width = max(28, int((width - .8) * 14))
    prompt_lines = [textwrap.wrap(" ".join(str(sample.prompt).split()), width=prompt_width,
                                 break_long_words=False, break_on_hyphens=False) or ["No text prompt"]
                    for sample in samples]
    row_heights = [3.25 + .18 * len(lines) for lines in prompt_lines]
    top = .66 if title else .18
    bottom = .60
    height = top + sum(row_heights) + bottom
    fig = Figure(figsize=(width, height), facecolor="white")
    FigureCanvasAgg(fig)
    if title:
        fig.text(.035, 1 - .26 / height, title, ha="left", va="top", fontsize=15,
                 fontweight="bold", color=INK, fontfamily="DejaVu Sans")
    records = []
    cursor = height - top
    for row, (sample, lines, row_height) in enumerate(zip(samples, prompt_lines, row_heights)):
        row_top = cursor
        label = f"{row + 1:02d}   {sample.dataset} / {sample.sample_id}"
        if (getattr(sample, "metadata", {}) or {}).get("demo"):
            label += "   ·   Illustrative demo"
        fig.text(.035, (row_top - .10) / height, label, va="top", fontsize=8., color=MUTED)
        fig.text(.035, (row_top - .33) / height, "\n".join(lines), va="top", fontsize=10.5,
                 linespacing=1.35, color=INK)
        panel_top = row_top - .42 - .18 * len(lines)
        panel_bottom = row_top - row_height + .09
        bounds = scene_bounds(sample, methods, show_keyframes=show_keyframes or include_input)
        panels = []
        for column, method in enumerate(columns):
            left = .025 + column * .95 / len(columns)
            panel_width = .95 / len(columns)
            color = GOLD if method is None else method_color(method, methods.index(method))
            heading = "Input keyframes" if method is None else method_label(method)
            heading = textwrap.fill(heading, width=max(20, int(panel_width * width * 10)),
                                    break_long_words=False, break_on_hyphens=False)
            fig.text(left + .012, (panel_top - .06) / height, heading, va="top",
                     fontsize=10.5, fontweight="bold", color=color)
            if method is None:
                indices, poses, _ = keyframe_constraints(sample)
                qualifier = "illustrative frames" if (getattr(sample, "metadata", {}) or {}).get("keyframes_illustrative") else "constrained frames"
                note = f"{len(indices)} {qualifier}" if poses is not None else "Text conditioning only" if not indices else "Source poses missing"
                fig.text(left + .012, (panel_top - .28) / height, note, va="top", fontsize=7.5, color=MUTED)
            ax = fig.add_axes([left, panel_bottom / height, panel_width,
                               (panel_top - .30 - panel_bottom) / height], projection="3d")
            panels.append(_draw_panel(ax, sample, method, color, bounds, show_keyframes=show_keyframes,
                                      elev=elev, azim=azim, camera_count=camera_count))
        records.append({"sample_id": str(sample.sample_id), "dataset": str(sample.dataset),
                        "prompt": str(sample.prompt), "metadata": _json_safe(getattr(sample, "metadata", {})),
                        "panels": panels})
        cursor -= row_height
        if row < len(samples) - 1:
            fig.add_artist(Line2D([.035, .965], [cursor / height] * 2, transform=fig.transFigure,
                                   color="#e8ebef", lw=.7))
    handles = [Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", lw=0,
                      markersize=5, label="Start"),
               Line2D([], [], color=MUTED, marker="D", lw=0, markersize=4, label="End")]
    constraint_columns = ([None] if include_input else []) + (methods if show_keyframes else [])
    if any(keyframe_constraints(sample, method)[1] is not None
           for sample in samples for method in constraint_columns):
        handles.append(Line2D([], [], color=GOLD, marker="o", markerfacecolor="none", lw=0,
                              markersize=5, label="Input constraint"))
    if any(getattr(sample, "subject", None) is not None for sample in samples):
        handles.append(Line2D([], [], color=SUBJECT, marker="s", ls="--", lw=.9, markersize=5, label="Subject"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .23 / height),
               ncol=len(handles), frameon=False, fontsize=8, labelcolor=MUTED,
               handlelength=1.3, columnspacing=1.8)
    fig.text(.5, .12 / height, "Shared scale and view within each row · camera-to-world poses · input labels use frame indices",
             ha="center", va="center", fontsize=6.7, color=MUTED)
    fig._lenscraft_metadata = {  # type: ignore[attr-defined]
        "schema_version": 1, "coordinate_convention": "OpenCV camera-to-world; world +Y up",
        "projection": "orthographic", "view": {"elev": elev, "azim": azim},
        "methods": methods, "show_keyframes": show_keyframes, "include_input": include_input,
        "camera_count": camera_count, "title": title, "path_interpolation": "none",
        "matplotlib_version": matplotlib.__version__, "samples": records,
    }
    return fig


def render_preview(samples: Any, *, dpi: int = 100, **options: Any) -> np.ndarray:
    """Render an RGB uint8 preview without a display server or pyplot."""
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    fig = build_figure(samples, **options)
    fig.set_dpi(dpi)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def export_figure(samples: Any, path: str | Path, *, methods: Sequence[str] | None = None,
                  show_keyframes: bool = True, include_input: bool = True,
                  elev: float = 24., azim: float = -58., dpi: int = 300,
                  formats: Sequence[str] = ("pdf", "svg", "png"), camera_count: int = 6,
                  title: str | None = None) -> dict[str, Path]:
    """Save vector PDF/SVG and a high-resolution PNG with a JSON provenance file.

    A path with a supported suffix exports that format; a suffixless path
    exports all requested ``formats``. Returned keys are format names and
    ``metadata``. This function never requires Torch, Viser, or a GPU.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    destination = Path(path).expanduser()
    suffix = destination.suffix.lower().lstrip(".")
    if suffix:
        if suffix not in {"pdf", "svg", "png"}:
            raise ValueError("Output suffix must be .pdf, .svg, .png, or omitted.")
        formats = (suffix,)
        destination = destination.with_suffix("")
    else:
        formats = tuple(dict.fromkeys(str(value).lower().lstrip(".") for value in formats))
    if not formats or not set(formats).issubset({"pdf", "svg", "png"}):
        raise ValueError("Choose one or more export formats from pdf, svg, png.")
    figure = build_figure(samples, methods=methods, show_keyframes=show_keyframes,
                          include_input=include_input, elev=elev, azim=azim,
                          camera_count=camera_count, title=title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    # Font embedding keeps PDF text editable and glyphs portable. SVG text is
    # outlined for consistent rendering on machines without the same fonts.
    with matplotlib.rc_context({"pdf.fonttype": 42, "svg.fonttype": "path"}):
        for fmt in formats:
            output = destination.with_suffix(f".{fmt}")
            figure.savefig(output, format=fmt, dpi=dpi, facecolor="white")
            written[fmt] = output
    metadata = dict(figure._lenscraft_metadata)  # type: ignore[attr-defined]
    metadata.update(dpi=dpi, figure_inches=figure.get_size_inches().tolist(),
                    files={key: str(value) for key, value in written.items()})
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False,
                                         allow_nan=False) + "\n", encoding="utf-8")
    written["metadata"] = metadata_path
    return written
