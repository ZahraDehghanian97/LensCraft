from __future__ import annotations

import logging
import math
import os
import textwrap
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import hydra
import lightning as L
import numpy as np
import torch
from dotenv import load_dotenv
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from data.dataset_type import resolve_dataset_type
from data.sim_format import to_simulation_format
from inferencing.process import inference_batch
from testing.baseline_modes import (
    BASELINE_ITEMS,
    NO_NORM_ITEM,
    NORM_ITEM,
    NORM_LENSCRAFT_INIT_ITEM,
)
from testing.process import _generate_baseline_variants
from models.factory import (
    load_model as _load_model,
    model_type_from_cfg as _model_type_from_cfg,
)
from utils.device import move_batch_to_device, resolve_device as _resolve_device
from visualization.viser_utils import (
    add_grid as _add_grid,
    safe_name as _safe,
    sim_to_standard as _sim_to_standard,
    volume_at as _vol_at,
    add_frustums,
    add_path,
    add_subject_box,
    auto_frustum_scale,
    matrix_to_wxyz,
    subject_dims,
    to_numpy,
    trajectory_diagonal,
    transforms_to_wxyz_position,
)

load_dotenv()
logger = logging.getLogger(__name__)

MODEL_LABEL = {
    "GT": "GT",
    "lens_craft": "LensCraft",
    "ccdm": "CCDM",
    "et": "E.T.",
    "gendop": "GenDoP",
}
MODEL_COLOR = {
    "GT": (60, 200, 90),
    "lens_craft": (240, 150, 40),
    "ccdm": (70, 130, 240),
    "et": (165, 95, 220),
    "gendop": (235, 95, 165),
}
COLOR_WORD = {
    "GT": "green",
    "lens_craft": "orange",
    "ccdm": "blue",
    "et": "purple",
    "gendop": "pink",
}
BASELINE_MODE_LABEL = {
    NO_NORM_ITEM: "plain",
    NORM_ITEM: "normalized",
    NORM_LENSCRAFT_INIT_ITEM: "normalized + LensCraft first",
}
BASELINE_MODE_COLOR_SCALE = {
    NO_NORM_ITEM: 1.25,
    NORM_ITEM: 1.0,
    NORM_LENSCRAFT_INIT_ITEM: 0.72,
}


@dataclass
class CompareSample:
    prompt: str
    columns: Dict[str, np.ndarray]          # "GT" first, then methods -> [T, 4, 4]
    subject: Optional[np.ndarray] = None    # [T, 4, 4]
    volume: Optional[np.ndarray] = None     # [3] (or [1, 3])


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _offset(transforms: np.ndarray, off: np.ndarray) -> np.ndarray:
    out = transforms.copy()
    out[..., :3, 3] = out[..., :3, 3] + off
    return out


def _baseline_column_name(model_type: str, mode: str) -> str:
    return f"{model_type}:{mode}"


def _split_column_name(name: str) -> tuple[str, Optional[str]]:
    model_type, separator, mode = name.partition(":")
    return (model_type, mode) if separator else (name, None)


def _column_label(name: str) -> str:
    model_type, mode = _split_column_name(name)
    label = MODEL_LABEL.get(model_type, model_type)
    if mode is not None:
        label = f"{label} — {BASELINE_MODE_LABEL.get(mode, mode)}"
    return label


def _column_color(name: str) -> tuple[int, int, int]:
    model_type, mode = _split_column_name(name)
    color = MODEL_COLOR.get(model_type, (210, 200, 60))
    scale = BASELINE_MODE_COLOR_SCALE.get(mode, 1.0)
    return tuple(min(255, round(channel * scale)) for channel in color)


def _column_color_word(name: str) -> str:
    model_type, mode = _split_column_name(name)
    color_word = COLOR_WORD.get(model_type, str(_column_color(name)))
    if mode == NO_NORM_ITEM:
        return f"light {color_word}"
    if mode == NORM_LENSCRAFT_INIT_ITEM:
        return f"dark {color_word}"
    return color_word


def _compose_model_cfg(model_name: str) -> DictConfig:
    """Compose this config with ``training/model=<name>`` so each baseline gets
    its own inference config (checkpoint path, seq_length, ...), exactly as if
    src/test.py had been launched with that override.

    Note: CLI overrides of the primary run are NOT forwarded here; baseline
    knobs come from their env vars (CCDM_CHECKPOINT_PATH, ...), as usual.
    """
    with initialize(version_base=None, config_path="../config"):
        return compose(config_name="compare", overrides=[f"training/model={model_name}"])


def fetch_batch(cfg: DictConfig, device: torch.device):
    dataset_type = resolve_dataset_type(cfg.data.dataset.config["_target_"])
    n = max(1, int(cfg.n_samples))

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=n,
        num_workers=0,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()
    batch = move_batch_to_device(next(iter(data_module.test_dataloader())), device)
    return batch, dataset_type


def _ground_truth(batch, dataset_type):
    sim_cam, sim_subj, sim_vol, _ = to_simulation_format(batch, dataset_type)
    gt_cam, gt_subj, gt_vol = _sim_to_standard(sim_cam, sim_subj, sim_vol, True)
    return (
        to_numpy(gt_cam),
        to_numpy(gt_subj) if gt_subj is not None else None,
        to_numpy(gt_vol) if gt_vol is not None else None,
    )


def _generate_for_model(
    cfg,
    model_cfg,
    batch,
    dataset_type,
    device,
    ref_model=None,
) -> Dict[str, np.ndarray]:
    """Run one method and return its comparison columns in standard poses."""
    model_type = _model_type_from_cfg(model_cfg)
    seq_length = int(model_cfg.training.model.data_format.seq_length)
    owns_model = model_type != "lens_craft" or ref_model is None
    model = _load_model(model_cfg, model_type, device) if owns_model else ref_model
    try:
        with torch.no_grad():
            if model_type == "lens_craft":
                results, *_ = inference_batch(
                    model, batch, device, dataset_type, model_type, seq_length
                )
                key = str(cfg.lens_craft_mode)
                if key not in results:
                    key = next(iter(results))
                cam_std = _sim_to_standard(results[key], None, None, denormalize=True)[
                    0
                ]
                return {model_type: to_numpy(cam_std)}

            if ref_model is None:
                raise ValueError(
                    f"{MODEL_LABEL.get(model_type, model_type)} requires the "
                    "LensCraft reference model to visualize all baseline modes."
                )

            sim_camera, sim_subject, *_ = to_simulation_format(
                batch, dataset_type
            )
            variants = _generate_baseline_variants(
                ref_model,
                model,
                batch,
                list(BASELINE_ITEMS),
                dataset_type,
                model_type,
                seq_length,
                sim_camera,
                sim_subject,
                len(batch["text_prompts"]),
            )
            columns = {}
            for mode in BASELINE_ITEMS:
                if mode not in variants:
                    continue
                # Plain baseline output is already in raw world coordinates;
                # the two normalized variants are in simulation-normalized space.
                cam_std = _sim_to_standard(
                    variants[mode],
                    None,
                    None,
                    denormalize=(mode != NO_NORM_ITEM),
                )[0]
                columns[_baseline_column_name(model_type, mode)] = to_numpy(cam_std)
            return columns
    finally:
        if owns_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# figure export (PIL)
# --------------------------------------------------------------------------- #
def _to_pil(img: np.ndarray):
    from PIL import Image

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr)


def _font(size: int):
    from PIL import ImageFont

    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_w(draw, text, font) -> float:
    try:
        return draw.textlength(text, font=font)
    except AttributeError:  # very old Pillow
        return draw.textsize(text, font=font)[0]


def stitch_row(panels, labels, prompt, out_path, pad=12,
               bg=(255, 255, 255), fg=(20, 20, 20)) -> None:
    """[img, img, ...] + labels + prompt caption -> one Figure-3-style row."""
    from PIL import Image, ImageDraw

    imgs = [_to_pil(p) for p in panels]
    w, h = imgs[0].size
    label_font = _font(max(16, h // 24))
    caption_font = _font(max(14, h // 30))
    label_size = getattr(label_font, "size", 14)
    caption_size = getattr(caption_font, "size", 12)

    n = len(imgs)
    total_w = n * w + (n + 1) * pad
    wrap_width = max(20, int(total_w / (caption_size * 0.62)))
    lines = textwrap.wrap(prompt or "", width=wrap_width) or [""]

    label_h = int(label_size * 1.9)
    caption_h = int(len(lines) * caption_size * 1.5) + pad
    canvas = Image.new("RGB", (total_w, pad + h + label_h + caption_h), bg)
    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(zip(imgs, labels)):
        x = pad + i * (w + pad)
        canvas.paste(img if img.size == (w, h) else img.resize((w, h)), (x, pad))
        tw = _text_w(draw, label, label_font)
        draw.text((x + (w - tw) / 2.0, pad + h + 0.25 * label_size),
                  label, fill=fg, font=label_font)

    y = pad + h + label_h
    for line in lines:
        tw = _text_w(draw, line, caption_font)
        draw.text(((total_w - tw) / 2.0, y), line, fill=fg, font=caption_font)
        y += int(caption_size * 1.5)

    canvas.save(out_path)


def stack_rows(row_paths, out_path, pad=16, bg=(255, 255, 255)) -> None:
    """Stack per-sample rows into one Figure-3-style grid."""
    from PIL import Image

    rows = [Image.open(p).convert("RGB") for p in row_paths]
    width = max(r.size[0] for r in rows)
    height = sum(r.size[1] for r in rows) + pad * (len(rows) + 1)
    canvas = Image.new("RGB", (width + 2 * pad, height), bg)
    y = pad
    for r in rows:
        canvas.paste(r, (pad + (width - r.size[0]) // 2, y))
        y += r.size[1] + pad
    canvas.save(out_path)


# --------------------------------------------------------------------------- #
# viser
# --------------------------------------------------------------------------- #
def launch(cfg: DictConfig, samples: List[CompareSample], column_names: List[str]) -> None:
    import viser

    server = viser.ViserServer(host=cfg.viser.host, port=cfg.viser.port)

    all_cams = [cam for s in samples for cam in s.columns.values()]
    base_scale = (
        float(cfg.frustum_scale)
        if float(cfg.frustum_scale) > 0
        else auto_frustum_scale(all_cams)
    )
    max_diag = max(trajectory_diagonal(c) for c in all_cams)
    spacing = float(cfg.grid_spacing) if float(cfg.grid_spacing) > 0 else 1.6 * max_diag
    max_t = max(c.shape[0] for c in all_cams)
    up_idx = {"x": 0, "y": 1, "z": 2}.get(str(cfg.up_axis), 1)
    out_dir = os.path.abspath(str(cfg.export.dir))

    space_axes = server.scene.add_frame(
        "/world", axes_length=base_scale * 4.0, axes_radius=base_scale * 0.1
    )
    if bool(cfg.show_grid):
        _add_grid(server, str(cfg.up_axis), base_scale)

    legend = "  |  ".join(
        f"{_column_label(n)} = {_column_color_word(n)}"
        for n in column_names
    )
    info_md = server.gui.add_markdown("")

    initial_layout = str(cfg.layout) if str(cfg.layout) in ("grid", "overlay") else "grid"
    sample_dd = server.gui.add_dropdown(
        "sample", options=[str(i) for i in range(len(samples))], initial_value="0"
    )
    layout_dd = server.gui.add_dropdown(
        "layout", options=["grid", "overlay"], initial_value=initial_layout
    )
    scale_sl = server.gui.add_slider(
        "frustum scale", min=base_scale * 0.1, max=base_scale * 5.0,
        step=base_scale * 0.05, initial_value=base_scale,
    )
    stride_sl = server.gui.add_slider(
        "frustum stride", min=1, max=max(1, max_t), step=1,
        initial_value=min(max(1, int(cfg.frustum_stride)), max(1, max_t)),
    )
    frame_sl = server.gui.add_slider(
        "frame", min=0, max=max(0, max_t - 1), step=1, initial_value=0
    )
    show_path = server.gui.add_checkbox("paths", True)
    show_frustums = server.gui.add_checkbox("frustums", True)
    show_subject = server.gui.add_checkbox("subject", True)
    show_space_axes = server.gui.add_checkbox("space axes", True)
    animate = server.gui.add_checkbox("animate", True)
    with server.gui.add_folder("methods"):
        col_cbs = {
            n: server.gui.add_checkbox(_column_label(n), True)
            for n in column_names
        }
    shot_btn = server.gui.add_button("capture this sample")
    shot_all_btn = server.gui.add_button("capture all samples")

    state = {"static": [], "current": [], "cur_arrays": [], "subjects": []}
    # Playback runs here while browser GUI callbacks run on Viser worker threads.
    scene_lock = threading.RLock()

    def clear() -> None:
        handles = state["static"] + state["current"] + [h for h, _ in state["subjects"]]
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        state["static"], state["current"] = [], []
        state["cur_arrays"], state["subjects"] = [], []

    def offsets_for(layout: str) -> Dict[str, np.ndarray]:
        if layout == "overlay":
            return {n: np.zeros(3, dtype=np.float32) for n in column_names}
        offs = {}
        for i, n in enumerate(column_names):
            off = np.zeros(3, dtype=np.float32)
            off[0] = i * spacing
            offs[n] = off
        return offs

    def _render(only: Optional[str] = None, force_origin: bool = False,
                sample: Optional[CompareSample] = None) -> None:
        clear()
        s = sample if sample is not None else samples[int(sample_dd.value)]
        info_md.content = f"**prompt:** {s.prompt}\n\n{legend}"

        layout = "overlay" if force_origin else str(layout_dd.value)
        offs = offsets_for(layout)
        names = (
            [only]
            if only is not None
            else [n for n in column_names if col_cbs[n].value and n in s.columns]
        )
        fov = math.radians(float(cfg.fov_deg))
        aspect = float(cfg.aspect)
        scale = float(scale_sl.value)
        stride = int(stride_sl.value)
        frame = int(frame_sl.value)
        convention = str(cfg.cam_convention)

        for name in names:
            cam = s.columns.get(name)
            if cam is None:
                continue
            off = np.zeros(3, dtype=np.float32) if force_origin else offs[name]
            cam_o = _offset(cam, off)
            color = _column_color(name)
            base = f"/cmp/{_safe(name)}"

            if show_path.value:
                handle = add_path(server, f"{base}/path", cam_o, color)
                if handle is not None:
                    state["static"].append(handle)
            if show_frustums.value:
                state["static"] += add_frustums(
                    server, f"{base}/frusta", cam_o, color,
                    fov, aspect, scale, stride=stride, cam_convention=convention,
                )

            wxyz, pos = transforms_to_wxyz_position(cam_o, convention)
            fi = min(frame, pos.shape[0] - 1)
            cur = server.scene.add_camera_frustum(
                f"{base}/current", fov=fov, aspect=aspect, scale=scale * 1.6,
                color=color,
                wxyz=tuple(float(v) for v in wxyz[fi]),
                position=tuple(float(v) for v in pos[fi]),
                line_width=2.5,
            )
            state["current"].append(cur)
            state["cur_arrays"].append((wxyz, pos, cur))

            if show_subject.value and s.subject is not None:
                subj_o = _offset(s.subject, off)
                fj = min(frame, subj_o.shape[0] - 1)
                sh = add_subject_box(
                    server, f"{base}/subject", subj_o[fj], subject_dims(s.volume)
                )
                state["subjects"].append((sh, subj_o))

            if layout == "grid" and only is None and not force_origin:
                pts = cam_o[..., :3, 3].reshape(-1, 3)
                label_pos = pts.mean(axis=0)
                label_pos[up_idx] = pts[:, up_idx].max() + 0.12 * max_diag
                try:
                    state["static"].append(
                        server.scene.add_label(
                            f"{base}/label",
                            text=_column_label(name),
                            position=tuple(float(v) for v in label_pos),
                        )
                    )
                except Exception:
                    pass  # labels are cosmetic; keep going on old viser versions

    def render(only: Optional[str] = None, force_origin: bool = False,
               sample: Optional[CompareSample] = None) -> None:
        with scene_lock:
            _render(only=only, force_origin=force_origin, sample=sample)

    def update_frame() -> None:
        with scene_lock:
            frame = int(frame_sl.value)
            for wxyz, pos, handle in state["cur_arrays"]:
                fi = min(frame, pos.shape[0] - 1)
                handle.position = tuple(float(v) for v in pos[fi])
                handle.wxyz = tuple(float(v) for v in wxyz[fi])
            for handle, subj in state["subjects"]:
                fj = min(frame, subj.shape[0] - 1)
                mat = subj[fj]
                handle.position = tuple(
                    float(v) for v in np.nan_to_num(mat[:3, 3])
                )
                handle.wxyz = tuple(
                    float(v) for v in matrix_to_wxyz(mat[:3, :3])
                )

    def update_space_axes() -> None:
        with scene_lock:
            space_axes.visible = bool(show_space_axes.value)

    def _grab(client) -> np.ndarray:
        height, width = int(cfg.export.height), int(cfg.export.width)
        try:
            img = client.get_render(height=height, width=width)
        except TypeError:
            img = client.get_render(height, width)
        return np.asarray(img)

    def capture_sample(sid: int) -> Optional[str]:
        clients = server.get_clients()
        client = next(iter(clients.values()), None)
        if client is None:
            logger.warning("No browser connected -- open the viser page before capturing.")
            return None

        s = samples[sid]
        names = [n for n in column_names if col_cbs[n].value and n in s.columns]
        panels, labels = [], []
        resume_animation = bool(animate.value)
        animate.value = False
        try:
            for name in names:
                # One method at a time, re-centered at the origin, so every
                # panel shares the exact same (current) camera framing.
                render(only=name, force_origin=True, sample=s)
                getattr(server, "flush", lambda: None)()
                time.sleep(float(cfg.export.settle_s))
                panels.append(_grab(client))
                labels.append(_column_label(name))
        except AttributeError as exc:
            logger.error(
                "This viser version has no client.get_render (%s). "
                "Upgrade with: pip install -U viser", exc,
            )
            render()
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Capture failed: %s", exc)
            render()
            return None
        finally:
            animate.value = resume_animation

        render()  # restore the interactive view
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"sample_{sid:03d}.png")
        try:
            stitch_row(panels, labels, s.prompt, path)
        except ImportError:
            logger.error("Pillow is required for figure export: pip install pillow")
            return None
        logger.info("Saved %s", path)
        return path

    def capture_all() -> None:
        resume_animation = bool(animate.value)
        animate.value = False
        try:
            paths = [
                p for p in (capture_sample(i) for i in range(len(samples))) if p
            ]
        finally:
            animate.value = resume_animation
        if len(paths) > 1:
            grid_path = os.path.join(out_dir, "comparison_grid.png")
            try:
                stack_rows(paths, grid_path)
                logger.info("Saved %s", grid_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not build the combined grid: %s", exc)

    controls = (
        sample_dd, layout_dd, scale_sl, stride_sl,
        show_path, show_frustums, show_subject, *col_cbs.values(),
    )
    for ctrl in controls:
        ctrl.on_update(lambda _: render())
    frame_sl.on_update(lambda _: update_frame())
    show_space_axes.on_update(lambda _: update_space_axes())
    shot_btn.on_click(lambda _: capture_sample(int(sample_dd.value)))
    shot_all_btn.on_click(lambda _: capture_all())

    render()
    logger.info("viser comparison server at http://%s:%s", cfg.viser.host, cfg.viser.port)
    logger.info("Figure exports will be written to %s", out_dir)

    period = 1.0 / 15.0
    while True:
        if animate.value and max_t > 1:
            with scene_lock:
                frame_sl.value = (int(frame_sl.value) + 1) % max_t
        time.sleep(period)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="../config", config_name="compare")
def main(cfg: DictConfig) -> None:
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    L.seed_everything(int(cfg.get("seed", 42)), workers=True)
    device = _resolve_device(cfg)

    batch, dataset_type = fetch_batch(cfg, device)
    batch_size = len(batch["text_prompts"])
    # Needed by inference_batch's "source_trajectory" mode (LensCraft path).
    batch["random_prompt_index"] = (
        np.random.randint(0, batch_size, size=batch_size).tolist()
    )

    gt_cam, gt_subj, gt_vol = _ground_truth(batch, dataset_type)

    model_names = [str(m) for m in cfg.models]
    baseline_models = {"ccdm", "et", "gendop"}
    needs_ref_model = any(name in baseline_models for name in model_names)
    ref_model = None
    if needs_ref_model:
        try:
            ref_model = _load_model(cfg, "lens_craft", device)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load the LensCraft reference model (%s: %s)",
                type(exc).__name__,
                exc,
            )

    outputs: Dict[str, np.ndarray] = {}
    try:
        for name in model_names:
            try:
                model_cfg = cfg if name == "lens_craft" else _compose_model_cfg(name)
                model_outputs = _generate_for_model(
                    cfg,
                    model_cfg,
                    batch,
                    dataset_type,
                    device,
                    ref_model=ref_model,
                )
                outputs.update(model_outputs)
                logger.info(
                    "Generated %d trajectories for each %s column: %s",
                    batch_size,
                    name,
                    list(model_outputs),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s (%s: %s)", name, type(exc).__name__, exc)
    finally:
        if ref_model is not None:
            del ref_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not outputs:
        raise RuntimeError(
            "No model could be loaded. Check TEST_CHECKPOINT_PATH (LensCraft) "
            "and the baseline checkpoint environment variables."
        )

    prompts = batch.get("text_prompts") or [""] * batch_size
    column_names = ["GT"] + list(outputs)

    samples: List[CompareSample] = []
    for i in range(batch_size):
        columns = {"GT": gt_cam[i]}
        for name in column_names[1:]:
            columns[name] = outputs[name][i]
        samples.append(
            CompareSample(
                prompt=str(prompts[i]),
                columns=columns,
                subject=gt_subj[i] if gt_subj is not None else None,
                volume=_vol_at(gt_vol, i),
            )
        )

    logger.info(
        "Prepared %d sample(s) x %d method(s): %s",
        len(samples), len(column_names), column_names,
    )
    launch(cfg, samples, column_names)


if __name__ == "__main__":
    main()
