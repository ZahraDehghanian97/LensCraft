from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from data.dataset_type import resolve_dataset_type
from data.convertor.convertor import convert_to_target
from data.convertor.constant import default_convertors, default_normalizers
from data.simulation.dataset import SimulationDataset
from utils.device import move_batch_to_device
from visualization.viser_utils import (
    ROUNDTRIP_COLOR,
    add_frustums,
    add_path,
    add_subject_box,
    auto_frustum_scale,
    color_for,
    matrix_to_wxyz,
    pose_error,
    subject_dims,
    to_numpy,
    transforms_to_wxyz_position,
)

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class VisTrajectory:
    name: str
    color: tuple
    cam: np.ndarray
    subject: Optional[np.ndarray] = None
    volume: Optional[np.ndarray] = None


def _convertor_key(name: str) -> str:
    return "simulation" if name in ("lens_craft", "simulation") else name


def _vol_at(vol_np: Optional[np.ndarray], i: int) -> Optional[np.ndarray]:
    if vol_np is None:
        return None
    if vol_np.ndim == 1:
        return vol_np
    return vol_np[i] if i < vol_np.shape[0] else vol_np[0]


def _safe(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_") or "x"


def _resolve_device(cfg: DictConfig) -> torch.device:
    if cfg.get("device"):
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_type_from_cfg(cfg: DictConfig) -> str:
    data_format_type = cfg.training.model.data_format.get("type", "simulation")
    return "lens_craft" if data_format_type == "simulation" else data_format_type


def _load_model(cfg: DictConfig, model_type: str, device: torch.device):
    from utils.load_lens_craft import load_lens_craft_model

    if model_type == "lens_craft":
        return load_lens_craft_model(
            model_module=cfg.training.model.module,
            model_inference=cfg.training.model.inference,
            device=device,
        )
    if model_type == "ccdm":
        from models.baselines.ccdm_adapter import CCDMAdapter

        return CCDMAdapter(cfg.training.model.inference, device)
    if model_type == "et":
        from models.baselines.et_adapter import ETAdapter

        return ETAdapter(cfg.training.model.inference, device)
    if model_type == "gendop":
        from models.baselines.gendop_adapter import GenDoPAdapter

        return GenDoPAdapter(cfg.training.model.inference, device)
    raise ValueError(f"Unsupported model type: {model_type}")


def fetch_batch(cfg: DictConfig, device: torch.device):
    dataset_type = resolve_dataset_type(cfg.data.dataset.config["_target_"])
    normalize = bool(cfg.data.dataset.config.get("normalize", True))
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
    return batch, dataset_type, normalize



def _dataset_to_standard(cam, subject, volume, key: str, normalize: bool):
    cam = cam.clone()
    subject = subject.clone() if subject is not None else None
    volume = volume.clone() if torch.is_tensor(volume) else volume
    if normalize:
        cam, subject, volume = default_normalizers[key](cam, subject, volume, False)
    return default_convertors[key].to_standard(cam, subject, volume)


def _sim_to_standard(cam, subject, volume, denormalize: bool):
    SimulationDataset.get_normalization_parameters()
    cam = cam.clone()
    subject = subject.clone() if subject is not None else None
    volume = volume.clone() if torch.is_tensor(volume) else volume
    if denormalize:
        cam, subject, volume = SimulationDataset.normalize_item(
            cam, subject, volume, False
        )
    return default_convertors["simulation"].to_standard(cam, subject, volume)


def _relative_to_subject(cam, subject):
    cam = cam.clone()
    subject = subject.clone()
    offset = subject[..., :3, 3]
    cam[..., :3, 3] = cam[..., :3, 3] - offset
    subject[..., :3, 3] = 0.0
    return cam, subject


def build_dataset(cfg, batch, dataset_type, normalize):
    key = _convertor_key(dataset_type)
    cam_std, subj_std, vol_std = _dataset_to_standard(
        batch["camera_trajectory"],
        batch.get("subject_trajectory"),
        batch.get("subject_volume"),
        key,
        normalize,
    )
    cam_np = to_numpy(cam_std)
    subj_np = to_numpy(subj_std) if subj_std is not None else None
    vol_np = to_numpy(vol_std) if vol_std is not None else None

    samples: Dict[int, List[VisTrajectory]] = {}
    for i in range(cam_np.shape[0]):
        samples[i] = [
            VisTrajectory(
                name="GT",
                color=color_for("gt"),
                cam=cam_np[i],
                subject=subj_np[i] if subj_np is not None else None,
                volume=_vol_at(vol_np, i),
            )
        ]
    info = (
        f"mode: dataset\n"
        f"dataset: {dataset_type}\n"
        f"samples: {cam_np.shape[0]}\n"
        f"frames/sample: {cam_np.shape[1]}\n"
        f"green = ground-truth camera, grey box = subject"
    )
    return samples, info


def build_output(cfg, batch, dataset_type, device):
    from inferencing.process import inference_batch

    model_type = _model_type_from_cfg(cfg)
    model = _load_model(cfg, model_type, device)
    seq_length = cfg.training.model.data_format.seq_length

    results, sim_camera, sim_subject, sim_volume, sim_padding, _ = inference_batch(
        model, batch, device, dataset_type, model_type, seq_length
    )

    gt_cam, gt_subj, gt_vol = _sim_to_standard(sim_camera, sim_subject, sim_volume, True)
    gen_denorm = model_type == "lens_craft"
    gen_std = {
        name: _sim_to_standard(traj, None, None, gen_denorm)[0]
        for name, traj in results.items()
    }

    gt_cam_np = to_numpy(gt_cam)
    gt_subj_np = to_numpy(gt_subj) if gt_subj is not None else None
    gt_vol_np = to_numpy(gt_vol) if gt_vol is not None else None
    gen_np = {name: to_numpy(t) for name, t in gen_std.items()}

    samples: Dict[int, List[VisTrajectory]] = {}
    for i in range(gt_cam_np.shape[0]):
        trajs = [
            VisTrajectory(
                name="GT",
                color=color_for("gt"),
                cam=gt_cam_np[i],
                subject=gt_subj_np[i] if gt_subj_np is not None else None,
                volume=_vol_at(gt_vol_np, i),
            )
        ]
        for j, (name, arr) in enumerate(gen_np.items()):
            trajs.append(
                VisTrajectory(name=name, color=color_for(name, j), cam=arr[i])
            )
        samples[i] = trajs

    info = (
        f"mode: output\n"
        f"dataset: {dataset_type}\n"
        f"model: {model_type}\n"
        f"samples: {gt_cam_np.shape[0]}\n"
        f"green = GT; other colors = generations:\n  "
        + ", ".join(gen_np.keys())
    )
    return samples, info


def build_convertor(cfg, batch, dataset_type, normalize):
    src_key = _convertor_key(dataset_type)
    tgt_key = _convertor_key(str(cfg.convert_target))

    cam = batch["camera_trajectory"]
    subject = batch.get("subject_trajectory")
    volume = batch.get("subject_volume")
    padding = batch.get("padding_mask")
    target_len = cam.shape[1]

    orig_cam, orig_subj, orig_vol = _dataset_to_standard(
        cam, subject, volume, src_key, normalize
    )

    rt_traj, rt_subj_fmt, rt_vol_fmt, _ = convert_to_target(
        src_key, tgt_key, cam, subject, volume, padding, target_len,
        need_denormal=normalize, need_normal=True,
    )
    rt_cam, rt_subj, rt_vol = _dataset_to_standard(
        rt_traj, rt_subj_fmt, rt_vol_fmt, tgt_key, True
    )

    if bool(cfg.relative_to_subject):
        orig_cam, orig_subj = _relative_to_subject(orig_cam, orig_subj)
        rt_cam, rt_subj = _relative_to_subject(rt_cam, rt_subj)

    err = pose_error(orig_cam, rt_cam)

    oc = to_numpy(orig_cam)
    os_ = to_numpy(orig_subj) if orig_subj is not None else None
    ov = to_numpy(orig_vol) if orig_vol is not None else None
    rc = to_numpy(rt_cam)

    samples: Dict[int, List[VisTrajectory]] = {}
    for i in range(oc.shape[0]):
        samples[i] = [
            VisTrajectory(
                name=f"orig ({src_key})",
                color=color_for("orig"),
                cam=oc[i],
                subject=os_[i] if os_ is not None else None,
                volume=_vol_at(ov, i),
            ),
            VisTrajectory(
                name=f"roundtrip ({src_key}->{tgt_key})",
                color=ROUNDTRIP_COLOR,
                cam=rc[i],
            ),
        ]

    notes = []
    if tgt_key == "ccdm":
        notes.append(
            "CCDM is subject-relative: a world-space overlay is offset by the "
            "subject position. Pass relative_to_subject=true to compare shapes."
        )
    if src_key in ("et", "ccdm"):
        notes.append(
            "Source is padded/variable-length: per-frame error includes "
            "resampling differences -- trust the visual overlay over the numbers."
        )
    info = (
        f"mode: convertor\n"
        f"round-trip: {src_key} -> {tgt_key} -> standard\n"
        f"relative_to_subject: {bool(cfg.relative_to_subject)}\n"
        f"samples: {oc.shape[0]}, frames: {oc.shape[1]}\n"
        f"green = original, red = round-trip\n"
        f"pose error over {int(err['frames'])} frames:\n"
        f"  position  mean={err['pos_mean']:.4f}  max={err['pos_max']:.4f}\n"
        f"  rotation  mean={err['rot_mean_deg']:.3f} deg  max={err['rot_max_deg']:.3f} deg"
    )
    if notes:
        info += "\n" + "\n".join(f"note: {n}" for n in notes)

    logger.info(
        "Convertor round-trip %s->%s: pos mean=%.4f max=%.4f | rot mean=%.3f max=%.3f deg",
        src_key, tgt_key, err["pos_mean"], err["pos_max"],
        err["rot_mean_deg"], err["rot_max_deg"],
    )
    return samples, info


def _add_grid(server, up_axis: str, scale: float) -> None:
    plane = {"y": "xz", "z": "xy", "x": "yz"}.get(up_axis, "xz")
    size = max(scale * 40.0, 1.0)
    try:
        server.scene.add_grid("/grid", width=size, height=size, plane=plane)
    except TypeError:
        try:
            server.scene.add_grid("/grid", width=size, height=size)
        except Exception:
            pass
    except Exception:
        pass


def _add_info(server, info_text: str) -> None:
    if not info_text:
        return
    try:
        server.gui.add_markdown(f"```\n{info_text}\n```")
    except Exception:
        pass


def launch(cfg, samples: Dict[int, List[VisTrajectory]], info_text: str) -> None:
    import viser

    server = viser.ViserServer(host=cfg.viser.host, port=cfg.viser.port)

    sample_ids = sorted(samples.keys())
    all_cams = [tr.cam for sid in sample_ids for tr in samples[sid]]
    base_scale = (
        float(cfg.frustum_scale)
        if float(cfg.frustum_scale) > 0
        else auto_frustum_scale(all_cams)
    )
    max_t = max((tr.cam.shape[0] for sid in sample_ids for tr in samples[sid]), default=1)

    server.scene.add_frame(
        "/world", axes_length=base_scale * 4.0, axes_radius=base_scale * 0.1
    )
    if bool(cfg.show_grid):
        _add_grid(server, str(cfg.up_axis), base_scale)
    _add_info(server, info_text)

    options = [str(s) for s in sample_ids] or ["0"]
    sample_dd = server.gui.add_dropdown(
        "sample", options=options, initial_value=options[0]
    )
    scale_sl = server.gui.add_slider(
        "frustum scale", min=base_scale * 0.1, max=base_scale * 5.0,
        step=base_scale * 0.05, initial_value=base_scale,
    )
    stride_sl = server.gui.add_slider(
        "frustum stride", min=1, max=max(1, max_t), step=1,
        initial_value=min(max(1, int(cfg.frustum_stride)), max(1, max_t)),
    )
    show_frustums = server.gui.add_checkbox("frustums", True)
    show_path = server.gui.add_checkbox("path", True)
    show_subject = server.gui.add_checkbox("subject", True)
    animate = server.gui.add_checkbox("animate", False)
    frame_sl = server.gui.add_slider(
        "frame", min=0, max=max(0, max_t - 1), step=1, initial_value=0
    )

    state = {
        "static": [], "current": [], "cur_arrays": [],
        "subject": None, "subject_src": None,
    }

    def current_trajs() -> List[VisTrajectory]:
        value = sample_dd.value
        sid = int(value) if value.isdigit() else (sample_ids[0] if sample_ids else 0)
        return samples.get(sid, [])

    def clear() -> None:
        for handle in state["static"] + state["current"]:
            try:
                handle.remove()
            except Exception:
                pass
        if state["subject"] is not None:
            try:
                state["subject"].remove()
            except Exception:
                pass
        state["static"] = []
        state["current"] = []
        state["cur_arrays"] = []
        state["subject"] = None
        state["subject_src"] = None

    def render() -> None:
        clear()
        trajs = current_trajs()
        fov = math.radians(float(cfg.fov_deg))
        aspect = float(cfg.aspect)
        scale = float(scale_sl.value)
        stride = int(stride_sl.value)
        frame = int(frame_sl.value)
        convention = str(cfg.cam_convention)

        for idx, tr in enumerate(trajs):
            base = f"/traj/{idx}_{_safe(tr.name)}"
            if show_path.value:
                handle = add_path(server, f"{base}/path", tr.cam, tr.color)
                if handle is not None:
                    state["static"].append(handle)
            if show_frustums.value:
                state["static"] += add_frustums(
                    server, f"{base}/frusta", tr.cam, tr.color,
                    fov, aspect, scale, stride=stride, cam_convention=convention,
                )

            wxyz, pos = transforms_to_wxyz_position(tr.cam, convention)
            state["cur_arrays"].append((wxyz, pos))
            fi = min(frame, pos.shape[0] - 1)
            state["current"].append(
                server.scene.add_camera_frustum(
                    f"{base}/current", fov=fov, aspect=aspect, scale=scale * 1.6,
                    color=tr.color,
                    wxyz=tuple(float(v) for v in wxyz[fi]),
                    position=tuple(float(v) for v in pos[fi]),
                    line_width=2.5,
                )
            )

        if show_subject.value:
            for tr in trajs:
                if tr.subject is not None:
                    fi = min(frame, tr.subject.shape[0] - 1)
                    state["subject"] = add_subject_box(
                        server, "/subject", tr.subject[fi], subject_dims(tr.volume)
                    )
                    state["subject_src"] = tr
                    break

    def update_frame() -> None:
        frame = int(frame_sl.value)
        for (wxyz, pos), handle in zip(state["cur_arrays"], state["current"]):
            fi = min(frame, pos.shape[0] - 1)
            handle.position = tuple(float(v) for v in pos[fi])
            handle.wxyz = tuple(float(v) for v in wxyz[fi])

        src = state["subject_src"]
        if state["subject"] is not None and src is not None and src.subject is not None:
            fi = min(frame, src.subject.shape[0] - 1)
            mat = src.subject[fi]
            state["subject"].position = tuple(
                float(v) for v in np.nan_to_num(mat[:3, 3])
            )
            state["subject"].wxyz = tuple(float(v) for v in matrix_to_wxyz(mat[:3, :3]))

    for ctrl in (sample_dd, scale_sl, stride_sl, show_frustums, show_path, show_subject):
        ctrl.on_update(lambda _: render())
    frame_sl.on_update(lambda _: update_frame())

    render()
    logger.info("viser server running at http://%s:%s", cfg.viser.host, cfg.viser.port)

    period = 1.0 / 15.0
    while True:
        if animate.value and max_t > 1:
            frame_sl.value = (int(frame_sl.value) + 1) % max_t
            update_frame()
        time.sleep(period)


@hydra.main(version_base=None, config_path="../config", config_name="viser")
def main(cfg: DictConfig) -> None:
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    device = _resolve_device(cfg)
    mode = str(cfg.mode)
    batch, dataset_type, normalize = fetch_batch(cfg, device)

    if mode == "dataset":
        samples, info = build_dataset(cfg, batch, dataset_type, normalize)
    elif mode == "output":
        samples, info = build_output(cfg, batch, dataset_type, device)
    elif mode == "convertor":
        samples, info = build_convertor(cfg, batch, dataset_type, normalize)
    else:
        raise ValueError(
            f"Unknown mode '{mode}' (expected 'dataset', 'output' or 'convertor')."
        )

    logger.info("Prepared %d sample(s) for mode '%s'", len(samples), mode)
    launch(cfg, samples, info)


if __name__ == "__main__":
    main()
