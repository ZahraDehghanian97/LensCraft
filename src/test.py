from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from data.datamodule import CameraTrajectoryDataModule
from data.dataset_type import DatasetType, resolve_dataset_type
from data.simulation.utils import (
    cinematography_struct_size,
    simulation_struct_size,
)
from models.factory import load_model, model_type_from_cfg as _model_type_from_cfg

from testing.metrics.callback import MetricCallback
from testing.metrics.clatr_extractor import CLaTrFeatureExtractor
from testing.metrics.native_clatr_extractor import NativeCLaTrFeatureExtractor
from testing.baseline_modes import (
    BASELINE_ITEMS,
    NORM_ITEM,
)
from testing.process import test_batch
from testing.trajectory_cache import (
    TRAJECTORY_CACHE_VERSION,
    build_trajectory_cache_key,
    validate_cached_metric_batch,
)
from utils.load_lens_craft import load_lens_craft_model
from visualization.utils import (
    tSNE_visualize_embeddings,
    tSNE_visualize_embeddings_by_class_type,
)

load_dotenv()
logger = logging.getLogger(__name__)

BASELINE_MODELS = ("ccdm", "et", "gendop")
CAPTIONED_DATASETS = ("simulation", "et")


def _trajectory_cache_key(cfg: DictConfig) -> str:
    config = OmegaConf.to_container(cfg, resolve=True)
    return build_trajectory_cache_key(
        config,
        resolve_path=hydra.utils.to_absolute_path,
    )


def _trajectory_cache_path(cfg: DictConfig, cache_key: str) -> Path:
    cache_dir = hydra.utils.to_absolute_path(
        str(cfg.get("trajectory_cache_dir", ".cache/test_trajectories"))
    )
    return Path(cache_dir) / f"{cache_key}.pt"


def _expected_test_batches(cfg: DictConfig, test_dataloader) -> int:
    available = len(test_dataloader)
    limit = int(cfg.get("limit_test_batches", 0))
    return min(available, limit) if limit else available


def _load_trajectory_cache(
    cfg: DictConfig,
    test_dataloader,
    required_metric_items: list[str],
) -> tuple[Path | None, list | None]:
    if not bool(cfg.get("trajectory_cache", True)):
        return None, None

    cache_key = _trajectory_cache_key(cfg)
    cache_path = _trajectory_cache_path(cfg, cache_key)
    if not cache_path.is_file():
        logger.info("Trajectory cache miss: %s", cache_path)
        return cache_path, None

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        batches = payload.get("batches", [])
        if (
            payload.get("version") != TRAJECTORY_CACHE_VERSION
            or payload.get("config_hash") != cache_key
            or len(batches) != _expected_test_batches(cfg, test_dataloader)
        ):
            raise ValueError("cache metadata or batch count does not match")
        model_type = _model_type_from_cfg(cfg)
        expected_token_count = (
            cinematography_struct_size + simulation_struct_size
        )
        expected_embedding_dim = int(cfg.clip.latent_dim)
        for batch in batches:
            validate_cached_metric_batch(
                batch,
                required_metric_items,
                model_type=model_type,
                require_encoder_features=(
                    model_type == "lens_craft"
                    and bool(cfg.get("caption_top1_metric", False))
                ),
                expected_token_count=expected_token_count,
                expected_embedding_dim=expected_embedding_dim,
                expected_sequence_length=int(cfg.training.model.data_format.seq_length),
            )
        logger.info("Trajectory cache hit: %s", cache_path)
        return cache_path, batches
    except Exception as exc:
        logger.warning("Ignoring invalid trajectory cache %s: %s", cache_path, exc)
        return cache_path, None


def _save_trajectory_cache(cfg: DictConfig, cache_path: Path, batches: list) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp-{os.getpid()}")
    torch.save(
        {
            "version": TRAJECTORY_CACHE_VERSION,
            "config_hash": _trajectory_cache_key(cfg),
            "batches": batches,
        },
        temporary_path,
    )
    os.replace(temporary_path, cache_path)
    logger.info("Saved generated trajectories to %s", cache_path)


def _build_native_clatr_extractor(
    cfg: DictConfig, device: torch.device
) -> NativeCLaTrFeatureExtractor:
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


def _pick_freest_cuda_device() -> torch.device:
    free_by_index = []
    for index in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        free_by_index.append((free_bytes, index))
    best_free, best_index = max(free_by_index)
    logger.info(
        "Auto-selected GPU %d with %.2f GiB free",
        best_index,
        best_free / (1024 ** 3),
    )
    return torch.device(f"cuda:{best_index}")


def _resolve_device(cfg: DictConfig) -> torch.device:
    requested = cfg.get("device")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if requested in (None, "cuda", "auto"):
        return _pick_freest_cuda_device()
    return torch.device(requested)


def _norm_ablation_enabled(cfg: DictConfig, model_type: str, dataset_type: str) -> bool:
    """Whether to report all three baseline post-processing modes."""
    return (
        bool(cfg.get("baseline_norm_ablation", True))
        and model_type in BASELINE_MODELS
        and dataset_type == "simulation"
    )


def _load_eval_models(
    cfg: DictConfig,
    model_type: str,
    device: torch.device,
):
    if model_type == "lens_craft":
        model = load_model(cfg, model_type, device)
        return model, model

    ref_model = load_lens_craft_model(
        model_module=cfg.ref_model.module,
        model_inference=cfg.ref_model.inference,
        device=device,
    )

    model = load_model(cfg, model_type, device)
    return model, ref_model


def _load_clip_embeddings(cfg: DictConfig, model_type: str):
    if model_type == "lens_craft" and cfg.get("caption_top1_metric", False):
        from data.simulation.init_embeddings import initialize_all_clip_embeddings

        return initialize_all_clip_embeddings(
            clip_model_name=str(cfg.clip.model_name),
            cache_file=cfg.training.model.inference.get(
                "clip_embeddings_cache", "clip_embeddings_cache.pkl"
            ),
            chunk_size=int(cfg.clip.chunk_size),
            embedding_dimension=int(cfg.clip.latent_dim),
        )
    return None


def _select_metric_items(
    model_type: str, dataset_type: str, norm_ablation: bool
) -> list[str]:
    if model_type in BASELINE_MODELS:
        if norm_ablation:
            return list(BASELINE_ITEMS)
        return [NORM_ITEM]
    if dataset_type in CAPTIONED_DATASETS:
        return [
            "reconstruction",
            "key_framing",
            "prompt_generation",
            "key_framing+prompt",
            "hybrid_generation",
        ]
    return ["reconstruction", "key_framing"]


def _run_evaluation(
    cfg: DictConfig,
    ref_model,
    model,
    metric_callback: MetricCallback,
    clatr_extractor,
    test_dataloader,
    metric_items: list[str],
    dataset_type: str,
    model_type: str,
    device: torch.device,
) -> None:
    limit = int(cfg.get("limit_test_batches", 0))  # 0 = no limit (smoke-test knob)
    seq_length = cfg.training.model.data_format.seq_length
    cache_path, cached_batches = _load_trajectory_cache(
        cfg, test_dataloader, metric_items
    )
    generated_batches = [] if cached_batches is None else None
    sample_offset = 0

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(test_dataloader)):
            if limit and bi >= limit:
                break
            sample_count = len(batch["text_prompts"])
            generation_seeds = (
                [(int(cfg.get("seed", 42)) + index) % (2**32)
                 for index in range(sample_offset, sample_offset + sample_count)]
                if model_type == "et" else None
            )
            generated = test_batch(
                ref_model, model, batch, metric_callback, device, metric_items,
                dataset_type=dataset_type,
                model_type=model_type,
                seq_length=seq_length,
                clatr_extractor=clatr_extractor,
                cached_outputs=(cached_batches[bi] if cached_batches is not None else None),
                generation_seeds=generation_seeds,
            )
            sample_offset += sample_count
            if generated_batches is not None:
                generated_batches.append(generated)

    if cache_path is not None and generated_batches is not None:
        _save_trajectory_cache(cfg, cache_path, generated_batches)


def _bootstrap_std(
    cfg: DictConfig, metric_callback: MetricCallback, metric_items: list[str]
) -> dict:
    n_boot = int(cfg.get("n_boot", 500))
    boot_max_cfg = cfg.get("boot_max_samples", None)
    boot_max = (
        None if boot_max_cfg in (None, "None", "", 0, "0") else int(boot_max_cfg)
    )
    return {
        item: metric_callback.bootstrap_metrics(
            item, n_boot=n_boot, max_samples=boot_max
        )
        for item in metric_items
        if item in metric_callback.active_metrics
    }


def _snapshot_features(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = torch.cat(list(x), dim=0)
    return x.detach().cpu()


def _collect_metric_features(
    cfg: DictConfig, metric_callback: MetricCallback, metric_items: list[str]
) -> dict:
    features = {item: {"GT": None, "GEN": None} for item in metric_items}
    if not cfg.tsne:
        return features

    for item in metric_items:
        prdc = metric_callback.metrics.get(item, {}).get("clatr_prdc")
        if prdc is None:
            continue
        features[item]["GT"] = _snapshot_features(getattr(prdc, "real_features", None))
        features[item]["GEN"] = _snapshot_features(getattr(prdc, "fake_features", None))
    return features


def _compute_and_log_metrics(
    cfg: DictConfig,
    metric_callback: MetricCallback,
    metric_items: list[str],
    boot_std: dict,
) -> dict:
    metrics = {
        item: metric_callback.compute_clatr_metrics(item)
        for item in metric_items
        if item in metric_callback.active_metrics
    }

    backend = cfg.get("clatr_backend", "native")
    logger.info("Final Metrics (CLaTr backend: %s): %s", backend, metrics)
    for item, center in metrics.items():
        boot = boot_std.get(item, {})
        line = ", ".join(
            (
                f"{key.split('/')[-1]}={mu:.4f}±{boot[key][1]:.4f}"
                if key in boot
                else f"{key.split('/')[-1]}={mu:.4f}"
            )
            for key, mu in center.items()
        )
        logger.info("Final Metrics (%s): %s", item, line)
    return metrics


def _write_metrics_json(
    cfg: DictConfig,
    metrics: dict,
    boot_std: dict,
    model_type: str,
    dataset_type: str,
) -> None:
    try:
        import json

        cfgd = cfg.data.dataset.config
        allowed_movement_types = (
            OmegaConf.to_container(cfgd.allowed_movement_types, resolve=True)
            if "allowed_movement_types" in cfgd
            and cfgd.allowed_movement_types is not None
            else None
        )
        et_type = (
            cfg.training.model.inference.get("et_type", None)
            if model_type == "et"
            else None
        )
        payload = {
            "model_type": model_type,
            "dataset_type": dataset_type,
            "et_type": et_type,
            "clatr_backend": cfg.get("clatr_backend", "native"),
            "prdc_recall_version": 2,
            "set": cfg.get("eval_set", None),
            "variant": cfg.get("variant", None),
            "allowed_movement_types": allowed_movement_types,
            "metrics": metrics,
            "bootstrap_std": boot_std,
        }

        tag = model_type
        if et_type:
            tag += f"_{et_type}"
        if payload["variant"]:
            tag += f"_{payload['variant']}"
        if payload["set"]:
            tag += f"_{payload['set']}"

        os.makedirs(cfg.output_dir, exist_ok=True)
        out_json = os.path.join(cfg.output_dir, f"metrics_{tag}.json")
        with open(out_json, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Wrote metrics JSON to %s", out_json)
    except Exception as exc:
        logger.warning("Failed to write metrics JSON: %s", exc)


def _collect_movement_types(test_dataloader) -> list[str]:
    movement_types: list[str] = []
    for batch in test_dataloader:
        for prompt_params in batch["cinematography_prompt_parameters"]:
            movement_types.append(prompt_params[4][1])
    return movement_types


def _make_tsne_plots(
    cfg: DictConfig,
    metric_features: dict,
    metric_items: list[str],
    model_type: str,
    dataset_type: str,
    test_dataloader,
) -> None:
    save_dir = os.path.dirname(os.path.dirname(cfg.ref_model.inference.config))
    features_save_dir = os.path.join(save_dir, "features")
    os.makedirs(features_save_dir, exist_ok=True)
    features_save_path = os.path.join(
        features_save_dir, f"dataset_{dataset_type}_model_{model_type}.pth"
    )
    torch.save(metric_features, features_save_path)

    os.makedirs(cfg.output_dir, exist_ok=True)
    for item, features in metric_features.items():
        if features["GT"] is not None and features["GEN"] is not None:
            logger.info("Creating t-SNE visualization for %s", item)
            tSNE_visualize_embeddings(
                features,
                title=f"Embedding Visualization using t-SNE ({item})",
                save_path=os.path.join(cfg.output_dir, f"embeddings_tSNE_{item}.png"),
            )

    by_movement_supported = (
        model_type == "lens_craft"
        and dataset_type == "simulation"
        and "prompt_generation" in metric_items
    )
    if not by_movement_supported:
        return

    movement_types = _collect_movement_types(test_dataloader)
    logger.info("Extracted %d movement types from the test set", len(movement_types))

    prompt_gen = metric_features["prompt_generation"]
    if prompt_gen["GT"] is not None and prompt_gen["GEN"] is not None and movement_types:
        tSNE_visualize_embeddings_by_class_type(
            caption_embeddings=prompt_gen["GT"],
            encoder_embeddings=prompt_gen["GEN"],
            class_types=movement_types,
            title="Embedding Visualization using t-SNE (Coloured by Movement Type)",
            save_path=os.path.join(
                cfg.output_dir, "embeddings_tSNE_by_movement_type.png"
            ),
        )


@hydra.main(version_base=None, config_path="../config", config_name="test")
def main(cfg: DictConfig) -> None:
    device = _resolve_device(cfg)

    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    model_type = _model_type_from_cfg(cfg)
    L.seed_everything(cfg.get("seed", 42), workers=True)

    if "SimulationDataset" in cfg.data.dataset.config["_target_"]:
        # Read native model inputs and the common reference directly from the
        # original clip, avoiding a second resampling of the ground truth.
        with open_dict(cfg.data.dataset.config):
            cfg.data.dataset.config.reference_frame_count = cfg.data.dataset.info.seq_length

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()
    dataset_type: DatasetType = resolve_dataset_type(cfg.data.dataset.config["_target_"])
    test_dataloader = data_module.test_dataloader()

    norm_ablation = _norm_ablation_enabled(cfg, model_type, dataset_type)
    if model_type in BASELINE_MODELS:
        logger.info(
            "Reporting baseline raw, normalized, and normalized + LensCraft "
            "initial-position modes in this run: %s",
            norm_ablation,
        )

    model, ref_model = _load_eval_models(cfg, model_type, device)
    clip_embeddings = _load_clip_embeddings(cfg, model_type)

    metric_callback = MetricCallback(
        num_cams=1, device=device, clip_embeddings=clip_embeddings
    )
    clatr_extractor = _build_clatr_extractor(cfg, device)
    metric_items = _select_metric_items(model_type, dataset_type, norm_ablation)

    _run_evaluation(
        cfg, ref_model, model, metric_callback, clatr_extractor,
        test_dataloader, metric_items, dataset_type, model_type,
        device,
    )

    boot_std = _bootstrap_std(cfg, metric_callback, metric_items)
    metric_features = _collect_metric_features(cfg, metric_callback, metric_items)
    metrics = _compute_and_log_metrics(cfg, metric_callback, metric_items, boot_std)

    _write_metrics_json(cfg, metrics, boot_std, model_type, dataset_type)

    if cfg.tsne:
        _make_tsne_plots(
            cfg, metric_features, metric_items, model_type, dataset_type, test_dataloader
        )


if __name__ == "__main__":
    main()
