from __future__ import annotations

import logging
import time

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from data.convertor.convertor import convert_to_target
from models.ccdm_adapter import CCDMAdapter
from models.et_adapter import ETAdapter
from models.gendop_adapter import GenDoPAdapter
from testing.process import to_cuda
from utils.load_lens_craft import load_lens_craft_model

load_dotenv()
logger = logging.getLogger(__name__)


def time_generation(gen_fn, device, n_warmup=10, n_runs=100):
    """Returns (mean_seconds, std_seconds) for gen_fn(), a no-arg closure over ONE fixed batch."""
    for _ in range(n_warmup):
        gen_fn()
    is_cuda = (getattr(device, "type", str(device)) == "cuda")
    if is_cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_runs):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen_fn()
        if is_cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts)
    return float(ts.mean()), float(ts.std(ddof=1))


def _resolve_dataset_type(target: str) -> str:
    if "CCDMDataset" in target:
        return "ccdm"
    if "ETDataset" in target:
        return "et"
    return "simulation"


@hydra.main(version_base=None, config_path="../config", config_name="test")
def main(cfg: DictConfig) -> None:
    device = (
        torch.device(cfg.device)
        if cfg.get("device")
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    data_format_type = cfg.training.model.data_format.get("type", "simulation")
    model_type = "lens_craft" if data_format_type == "simulation" else data_format_type

    if model_type in ("ccdm", "gendop"):
        default_warmup, default_runs = 3, 8
    else:
        default_warmup, default_runs = 10, 50
    n_warmup = int(cfg.get("eff_warmup", default_warmup))
    n_runs = int(cfg.get("eff_runs", default_runs))

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()
    dataset_type = _resolve_dataset_type(cfg.data.dataset.config["_target_"])
    seq_length = cfg.training.model.data_format.seq_length

    batch = next(iter(data_module.test_dataloader()))
    batch = to_cuda(batch, device)
    batch_size = len(batch["text_prompts"])

    if dataset_type != "simulation":
        (
            sim_camera_trajectory,
            sim_subject_trajectory,
            sim_subject_volume,
            sim_padding_mask,
        ) = convert_to_target(
            dataset_type, "simulation",
            batch["camera_trajectory"], batch["subject_trajectory"],
            batch["subject_volume"], batch["padding_mask"], 30,
        )
    else:
        sim_camera_trajectory = batch["camera_trajectory"]
        sim_subject_trajectory = batch["subject_trajectory"]
        sim_subject_volume = batch["subject_volume"]
        sim_padding_mask = batch["padding_mask"]

    if model_type == "lens_craft":
        ref_model = load_lens_craft_model(
            model_module=cfg.training.model.module,
            model_inference=cfg.training.model.inference,
            device=device,
        )
        caption_embedding = batch.get("cinematography_prompt", None)

        gen_fn = lambda: ref_model.generate_camera_trajectory(
            subject_trajectory=sim_subject_trajectory,
            subject_volume=sim_subject_volume,
            camera_trajectory=sim_camera_trajectory,
            padding_mask=sim_padding_mask,
            memory_teacher_forcing_ratio=1.0,
            caption_embedding=caption_embedding,
        )

        flop_model = ref_model
        flop_inputs = (sim_camera_trajectory, sim_subject_trajectory, sim_subject_volume)
    else:
        (
            camera_trajectory,
            subject_trajectory,
            subject_volume,
            padding_mask,
        ) = convert_to_target(
            dataset_type, model_type,
            batch["camera_trajectory"], batch["subject_trajectory"],
            batch["subject_volume"], batch["padding_mask"], seq_length,
            torch.full((batch_size,), 30, device=device),
        )

        if model_type == "ccdm":
            model = CCDMAdapter(cfg.training.model.inference, device)
        elif model_type == "et":
            model = ETAdapter(cfg.training.model.inference, device)
        elif model_type == "gendop":
            model = GenDoPAdapter(cfg.training.model.inference, device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        gen_fn = lambda: model.generate_using_text(
            batch["text_prompts"],
            subject_trajectory,
            camera_trajectory,
            padding_mask,
        )
        flop_model = None  # baselines are multi-step/autoregressive; see caveats above
        flop_inputs = None

    logger.info("Timing %s (batch_size=%d, warmup=%d, runs=%d) ...",
                model_type, batch_size, n_warmup, n_runs)
    mean_s, std_s = time_generation(gen_fn, device, n_warmup=n_warmup, n_runs=n_runs)
    per_traj_mean = mean_s / batch_size
    logger.info("Inference time (batch): %.6f +/- %.6f s", mean_s, std_s)
    logger.info("Inference time (per trajectory): %.6f +/- %.6f s",
                per_traj_mean, std_s / batch_size)

    gflops = None
    if flop_model is not None:
        try:
            from fvcore.nn import FlopCountAnalysis

            flop_model.eval()
            with torch.no_grad():
                flops = FlopCountAnalysis(flop_model, flop_inputs)
                flops.unsupported_ops_warnings(False)
                flops.uncalled_modules_warnings(False)
                gflops = flops.total() / 1e9 / batch_size
            logger.info(
                "GFLOPs per trajectory (counts model.forward(), NOT the "
                "single-step generate path; note this in the T4 caption): %.4f",
                gflops,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FLOPs count failed for %s: %s", model_type, exc)
    else:
        logger.info(
            "FLOPs for %s not auto-counted: multiply the per-step/per-token forward "
            "FLOPs by the denoising-step count (CCDM N_T) or generated-token count "
            "(GenDoP). State that count in the T4 caption.", model_type,
        )

    try:
        import json, os
        eff = {
            "model_type": model_type,
            "inference_time_batch_s": mean_s,
            "inference_time_batch_std_s": std_s,
            "inference_time_per_traj_s": per_traj_mean,
            "gflops_per_traj": gflops,   # None for baselines (not auto-counted)
            "batch_size": batch_size, "n_warmup": n_warmup, "n_runs": n_runs,
        }
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, f"efficiency_{model_type}.json"), "w") as fh:
            json.dump(eff, fh, indent=2)
        logger.info("Wrote efficiency JSON to efficiency_%s.json", model_type)
    except Exception as exc:
        logger.warning("Failed to write efficiency JSON: %s", exc)


if __name__ == "__main__":
    main()
