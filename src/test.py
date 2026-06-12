from __future__ import annotations

import logging
import os
from typing import Literal

import hydra
import torch
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.datamodule import CameraTrajectoryDataModule
from models.ccdm_adapter import CCDMAdapter
from models.et_adapter import ETAdapter
from models.gendop_adapter import GenDoPAdapter

from testing.metrics.callback import MetricCallback
from testing.metrics.clatr_extractor import CLaTrFeatureExtractor
from testing.metrics.native_clatr_extractor import NativeCLaTrFeatureExtractor
from testing.process import test_batch
from utils.load_lens_craft import load_lens_craft_model
from visualization.utils import (
    tSNE_visualize_embeddings,
    tSNE_visualize_embeddings_by_class_type,
)

load_dotenv()
logger = logging.getLogger(__name__)

DatasetType = Literal["ccdm", "et", "simulation"]

def _build_native_clatr_extractor(
    cfg: DictConfig, device: torch.device
) -> NativeCLaTrFeatureExtractor:
    """Native CLaTr trained on LensCraft data (default)."""
    checkpoint_path = (
        cfg.get("clatr_native_checkpoint_path", None)
        or os.environ.get("CLATR_NATIVE_CHECKPOINT_PATH")
    )
    if checkpoint_path in (None, "None", ""):
        raise EnvironmentError(
            "Native CLaTr is the default evaluation backend but no checkpoint "
            "was provided. Train one with `python src/train_clatr.py` and "
            "either point `clatr_native_checkpoint_path` at it in your test "
            "config, set the CLATR_NATIVE_CHECKPOINT_PATH environment "
            "variable, or fall back to the legacy E.T. CLaTr by passing "
            "`clatr_backend=et` to the test script."
        )

    logger.info("Using native CLaTr backend (checkpoint: %s)", checkpoint_path)
    return NativeCLaTrFeatureExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
        clip_model_name=cfg.clip.model_name
        if cfg.clip.model_name.startswith("openai/")
        else f"openai/{cfg.clip.model_name}",
    )


def _build_et_clatr_extractor(
    cfg: DictConfig, device: torch.device
) -> CLaTrFeatureExtractor:
    """Legacy E.T.-trained CLaTr backend (kept for parity with prior runs)."""
    director_project_dir = os.environ.get("DIRECTOR_PROJECT_DIR")
    if director_project_dir is None:
        raise EnvironmentError(
            "DIRECTOR_PROJECT_DIR must be set to point at "
            "third_parties/DIRECTOR so the legacy E.T. CLaTr can be loaded."
        )

    project_config = os.path.join(director_project_dir, "configs", "config.yaml")
    et_data_dir = os.environ.get("ET_DATA_DIR")
    if et_data_dir is None:
        raise EnvironmentError(
            "ET_DATA_DIR must be set so Hydra can instantiate the E.T. CLaTr "
            "config (no ET data is actually read)."
        )

    default_ckpt = os.path.join(
        os.path.dirname(director_project_dir), "checkpoints", "clatr-e100.ckpt"
    )
    checkpoint_path = (
        cfg.get("clatr_checkpoint_path", None)
        or os.environ.get("CLATR_CHECKPOINT_PATH")
        or default_ckpt
    )

    logger.info("Using E.T. CLaTr backend (checkpoint: %s)", checkpoint_path)
    return CLaTrFeatureExtractor(
        project_config_dir=project_config,
        dataset_dir=et_data_dir,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def _build_clatr_extractor(cfg: DictConfig, device: torch.device):
    backend = str(cfg.get("clatr_backend", "native")).lower()
    if backend == "native":
        return _build_native_clatr_extractor(cfg, device)
    if backend == "et":
        return _build_et_clatr_extractor(cfg, device)
    raise ValueError(
        f"Unknown clatr_backend '{backend}'. Expected 'native' or 'et'."
    )


def _resolve_dataset_type(target: str) -> DatasetType:
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

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()
    dataset_type: DatasetType = _resolve_dataset_type(cfg.data.dataset.config["_target_"])

    trajectories_dir = os.path.join(cfg.cache_dir, "generated_trajectory")
    os.makedirs(trajectories_dir, exist_ok=True)
    if model_type == "et":
        trajectory_save_path = os.path.join(
            trajectories_dir,
            f"dataset_{dataset_type}_model_{model_type}_{cfg.training.model.inference.et_type}.pth",
        )
    else:
        trajectory_save_path = os.path.join(
            trajectories_dir, f"dataset_{dataset_type}_model_{model_type}.pth"
        )

    model = None
    if model_type == "lens_craft":
        model = load_lens_craft_model(
            model_module=cfg.training.model.module,
            model_inference=cfg.training.model.inference,
            device=device,
        )
        ref_model = model
    else:
        ref_model = load_lens_craft_model(
            model_module=cfg.ref_model.module,
            model_inference=cfg.ref_model.inference,
            device=device,
        )
        if not os.path.exists(trajectory_save_path):
            if model_type == "ccdm":
                model = CCDMAdapter(cfg.training.model.inference, device)
            elif model_type == "et":
                model = ETAdapter(cfg.training.model.inference, device)
            elif model_type == "gendop":
                model = GenDoPAdapter(cfg.training.model.inference, device)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

    clip_embeddings = None
    if model_type == "lens_craft" and cfg.get("caption_top1_metric", False):
        from data.simulation.init_embeddings import initialize_all_clip_embeddings

        clip_embeddings = initialize_all_clip_embeddings(
            cache_file=cfg.training.model.inference.get(
                "clip_embeddings_cache", "clip_embeddings_cache.pkl"
            )
        )

    metric_callback = MetricCallback(num_cams=1, device=device, clip_embeddings=clip_embeddings)
    clatr_extractor = _build_clatr_extractor(cfg, device)

    test_dataloader = data_module.test_dataloader()

    if model_type in ("ccdm", "et", "gendop"):
        metric_items = ["prompt_generation"]
    else:
        metric_items = (
            [
                "reconstruction",
                "key_framing",
                "prompt_generation",
                "key_framing+prompt",
                "hybrid_generation",
            ]
            if dataset_type in ("simulation", "et")
            else ["reconstruction", "key_framing"]
        )

    if os.path.exists(trajectory_save_path) and model_type in ("ccdm", "et", "gendop"):
        logger.info("Loading pre-generated trajectories from %s", trajectory_save_path)
        generated_trajectories = torch.load(trajectory_save_path)
        logger.info("Loaded %d pre-generated trajectories", len(generated_trajectories))

        with torch.no_grad():
            for batch, generated_trajectory in tqdm(
                zip(test_dataloader, generated_trajectories),
                total=min(len(test_dataloader), len(generated_trajectories)),
            ):
                test_batch(
                    ref_model,
                    model,
                    batch,
                    metric_callback,
                    device,
                    metric_items,
                    dataset_type=dataset_type,
                    model_type=model_type,
                    seq_length=cfg.training.model.data_format.seq_length,
                    pre_generated_trajectory=generated_trajectory.to(device),
                    clatr_extractor=clatr_extractor,
                )
    else:
        all_generated_trajectories = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                generated_trajectory_data = test_batch(
                    ref_model,
                    model,
                    batch,
                    metric_callback,
                    device,
                    metric_items,
                    dataset_type=dataset_type,
                    model_type=model_type,
                    seq_length=cfg.training.model.data_format.seq_length,
                    clatr_extractor=clatr_extractor,
                )
                if generated_trajectory_data is not None:
                    all_generated_trajectories.append(generated_trajectory_data)

    metrics = {
        item: metric_callback.compute_clatr_metrics(item)
        for item in metric_items
        if item in metric_callback.active_metrics
    }
    logger.info("Final Metrics (CLaTr backend: %s): %s", cfg.get("clatr_backend", "native"), metrics)

    if cfg.tsne:
        metric_features = {item: {"GT": None, "GEN": None} for item in metric_items}
        for metric_item in metric_items:
            if metric_item in metric_callback.metrics:
                prdc = metric_callback.metrics[metric_item].get("clatr_prdc")
                if prdc is not None:
                    if getattr(prdc, "real_features", None) is not None:
                        metric_features[metric_item]["GT"] = prdc.real_features
                    if getattr(prdc, "fake_features", None) is not None:
                        metric_features[metric_item]["GEN"] = prdc.fake_features

        save_dir = os.path.dirname(os.path.dirname(cfg.ref_model.inference.config))
        features_save_dir = os.path.join(save_dir, "features")
        os.makedirs(features_save_dir, exist_ok=True)
        features_save_path = os.path.join(
            features_save_dir, f"dataset_{dataset_type}_model_{model_type}.pth"
        )
        torch.save(metric_features, features_save_path)

        os.makedirs(cfg.output_dir, exist_ok=True)
        for metric_item, features in metric_features.items():
            if features["GT"] is not None and features["GEN"] is not None:
                logger.info("Creating t-SNE visualization for %s", metric_item)
                save_path = os.path.join(
                    cfg.output_dir, f"embeddings_tSNE_{metric_item}.png"
                )
                tSNE_visualize_embeddings(
                    features,
                    title=f"Embedding Visualization using t-SNE ({metric_item})",
                    save_path=save_path,
                )

        if (
            model_type == "lens_craft"
            and dataset_type == "simulation"
            and "prompt_generation" in metric_items
        ):
            movement_types: list[str] = []
            if dataset_type == "simulation":
                for batch in test_dataloader:
                    for prompt_params in batch["cinematography_prompt_parameters"]:
                        movement_types.append(prompt_params[4][1])
            logger.info("Extracted %d movement types from the test set", len(movement_types))

            if (
                metric_features["prompt_generation"]["GT"] is not None
                and metric_features["prompt_generation"]["GEN"] is not None
                and movement_types
            ):
                tSNE_visualize_embeddings_by_class_type(
                    caption_embeddings=metric_features["prompt_generation"]["GT"],
                    encoder_embeddings=metric_features["prompt_generation"]["GEN"],
                    class_types=movement_types,
                    title="Embedding Visualization using t-SNE (Coloured by Movement Type)",
                    save_path=os.path.join(
                        cfg.output_dir, "embeddings_tSNE_by_movement_type.png"
                    ),
                )


if __name__ == "__main__":
    main()
