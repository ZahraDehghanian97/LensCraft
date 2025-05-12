import logging
import os
from typing import Literal

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.datamodule import CameraTrajectoryDataModule
from testing.lens_craft import load_simulation_model
from testing.process import test_batch
from testing.metrics.callback import MetricCallback
from visualization.tsne import tSNE_visualize_embeddings
from models.ccdm_adapter import CCDMAdapter
from models.et_adapter import ETAdapter

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
    sim_model = None
    
    if model_type == "simulation":
        model = load_simulation_model(model_module=cfg.training.model.module, model_inference=cfg.training.model.module, device=device)
        sim_model = model
    else:
        if model_type == "ccdm":
            model = CCDMAdapter(cfg.training.model.inference, device)
        elif model_type == "et":
            model = ETAdapter(cfg.training.model.inference, device)        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        sim_model = load_simulation_model(model_module=cfg.sim_model.module, model_inference=cfg.sim_model.inference, device=device)
    
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


    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            test_batch(sim_model, model, batch, metric_callback, device, metric_items, dataset_type='simulation', model_type='simulation')

    metric_features = {metric_item: {"GT": None, "GEN": None} for metric_item in metric_items}
    for metric_item in metric_items:
        if metric_item in metric_callback.active_metrics:
            if metric_item in metric_callback.metrics and "clatr_prdc" in metric_callback.metrics[metric_item]:
                prdc = metric_callback.metrics[metric_item]["clatr_prdc"]
                if hasattr(prdc, "real_features") and prdc.real_features is not None:
                    metric_features[metric_item]["GT"] = prdc.real_features
                if hasattr(prdc, "fake_features") and prdc.fake_features is not None:
                    metric_features[metric_item]["GEN"] = prdc.fake_features

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for metric_item, features in metric_features.items():
        if features["GT"] is not None and features["GEN"] is not None:
            logger.info(f"Creating t‑SNE visualization for {metric_item}")
            save_path = os.path.join(output_dir, f"embeddings_tSNE_{metric_item}.png")
            tSNE_visualize_embeddings(
                features,
                title=f"Embedding Visualization using t‑SNE ({metric_item})",
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
