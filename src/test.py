import logging
import os
from typing import Literal

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from tqdm import tqdm

from models.camera_trajectory_model import MultiTaskAutoencoder
from data.datamodule import CameraTrajectoryDataModule
from testing.metrics.callback import MetricCallback
from utils.checkpoint import load_checkpoint
from visualization.tsne import tSNE_visualize_embeddings
from models.ccdm_adapter import CCDMAdapter
from models.et_adapter import ETAdapter
from testing.ccdm import process_ccdm_batch
from testing.lens_craft import process_lens_craft_batch

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

DatasetType = Literal["ccdm", "et", "simulation"]

@hydra.main(version_base=None, config_path="../config", config_name="test")
def main(cfg: DictConfig) -> None:
    if cfg.get("device"):
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    clip_embeddings = None
    
    model_type = cfg.training.model.data_format.get("type", "simulation")
    if model_type == "simulation" and cfg.get("caption_top1_metric", False):
        from data.simulation.init_embeddings import initialize_all_clip_embeddings
        clip_embeddings = initialize_all_clip_embeddings(cache_file=cfg.training.model.inference.get("clip_embeddings_cache", "clip_embeddings_cache.pkl"))

    model = None

    if model_type == "ccdm":
        if CCDMAdapter is None:
            raise ImportError("CCDM adapter not found. Please ensure models/ccdm_adapter.py exists.")
        logger.info("Using CCDM model for inference")
        model = CCDMAdapter(cfg.training.model.inference, device)
    elif model_type == "et":
        model = ETAdapter(cfg.training.model.inference, device)
    elif model_type in "simulation":
        checkpoint_cfg_path = to_absolute_path(cfg.training.model.inference.config)
        checkpoint_cfg  = OmegaConf.load(checkpoint_cfg_path)
        model: MultiTaskAutoencoder = instantiate(checkpoint_cfg.training.model.module)
        model = load_checkpoint(cfg.training.model.inference.checkpoint_path, model, device)
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()

    metric_callback = MetricCallback(num_cams=1, device=device, clip_embeddings=clip_embeddings)

    target = cfg.data.dataset.config["_target_"]
    dataset_type = (
        "ccdm" if "CCDMDataset" in target else "et" if "ETDataset" in target else "simulation"
    )

    test_dataloader = data_module.test_dataloader()

    if model_type in ["ccdm", "et"]:
        metric_items = ["prompt_generation"]
    else:
        metric_items = (
            ["reconstruction", "prompt_generation", "hybrid_generation"]
            if dataset_type == "simulation"
            else ["reconstruction"]
        )

    metric_features = {metric_type: {"GT": None, "GEN": None} for metric_type in metric_items}

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            text_prompts = batch["text_prompts"]
            subject_trajectory = batch["subject_trajectory"]
            padding_mask = batch.get("padding_mask", None)
            if model_type == "ccdm":
                process_ccdm_batch(model, batch, metric_callback, device)
            elif model_type == "et":
                model.generate_using_text(text_prompts, subject_trajectory, padding_mask)
            else:
                process_lens_craft_batch(model, batch, metric_callback, dataset_type, device)

        for metric_type in metric_items:
            if metric_type in metric_callback.active_metrics:
                if metric_type in metric_callback.metrics and "clatr_prdc" in metric_callback.metrics[metric_type]:
                    prdc = metric_callback.metrics[metric_type]["clatr_prdc"]
                    if hasattr(prdc, "real_features") and prdc.real_features is not None:
                        metric_features[metric_type]["GT"] = prdc.real_features
                    if hasattr(prdc, "fake_features") and prdc.fake_features is not None:
                        metric_features[metric_type]["GEN"] = prdc.fake_features

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for metric_type, features in metric_features.items():
        if features["GT"] is not None and features["GEN"] is not None:
            logger.info(f"Creating t‑SNE visualization for {metric_type}")
            save_path = os.path.join(output_dir, f"embeddings_tSNE_{metric_type}.png")
            tSNE_visualize_embeddings(
                features,
                title=f"Embedding Visualization using t‑SNE ({metric_type})",
                save_path=save_path,
            )

    metrics = {
        item: metric_callback.compute_clatr_metrics(item)
        for item in metric_items
        if item in metric_callback.active_metrics
    }

    logger.info(f"Final Metrics: {metrics}")


if __name__ == "__main__":
    main()
