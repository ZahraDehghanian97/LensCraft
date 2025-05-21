import logging
import os
from typing import Literal

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.datamodule import CameraTrajectoryDataModule
from utils.load_lens_craft import load_lens_craft_model
from testing.process import test_batch
from testing.metrics.callback import MetricCallback
from visualization.utils import tSNE_visualize_embeddings, tSNE_visualize_embeddings_by_class_type
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
    
    data_format_type = cfg.training.model.data_format.get("type", "simulation")
    model_type = "lens_craft" if data_format_type == "simulation" else data_format_type
    if model_type == "lens_craft" and cfg.get("caption_top1_metric", False):
        from data.simulation.init_embeddings import initialize_all_clip_embeddings
        clip_embeddings = initialize_all_clip_embeddings(cache_file=cfg.training.model.inference.get("clip_embeddings_cache", "clip_embeddings_cache.pkl"))

    model = None
    ref_model = None
    
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()
    
    target = cfg.data.dataset.config["_target_"]
    dataset_type = (
        "ccdm" if "CCDMDataset" in target else "et" if "ETDataset" in target else "simulation"
    )
    
    trajectories_dir = os.path.join(cfg.cache_dir, "generated_trajectory")
    os.makedirs(trajectories_dir, exist_ok=True)
    if model_type == "et":
        trajectory_save_path = os.path.join(trajectories_dir, f"dataset_{dataset_type}_model_{model_type}_{cfg.training.model.inference.et_type}.pth")
    else:
        trajectory_save_path = os.path.join(trajectories_dir, f"dataset_{dataset_type}_model_{model_type}.pth")        

    if model_type == "lens_craft":
        model = load_lens_craft_model(model_module=cfg.training.model.module, model_inference=cfg.training.model.inference, device=device)
        ref_model = model
    else:
        ref_model = load_lens_craft_model(model_module=cfg.ref_model.module, model_inference=cfg.ref_model.inference, device=device)
        if not os.path.exists(trajectory_save_path):
            if model_type == "ccdm":
                model = CCDMAdapter(cfg.training.model.inference, device)
            elif model_type == "et":
                model = ETAdapter(cfg.training.model.inference, device)        
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

    metric_callback = MetricCallback(num_cams=1, device=device, clip_embeddings=clip_embeddings)

    test_dataloader = data_module.test_dataloader()

    if model_type in ["ccdm", "et"]:
        metric_items = ["prompt_generation"]
    else:
        metric_items = (
            ["reconstruction", "prompt_generation", "hybrid_generation"]
            if dataset_type in ["simulation", "et"]
            else ["reconstruction"]
        )
    
    if os.path.exists(trajectory_save_path) and model_type in ["ccdm", "et"]:
        logger.info(f"Loading pre-generated trajectories from {trajectory_save_path}")
        generated_trajectories = torch.load(trajectory_save_path)
        logger.info(f"Loaded {len(generated_trajectories)} pre-generated trajectories")
        
        with torch.no_grad():
            for batch, generated_trajectory in tqdm(zip(test_dataloader, generated_trajectories)):
                test_batch(
                    ref_model, model, batch, metric_callback, device, metric_items, 
                    dataset_type=dataset_type, model_type=model_type, 
                    seq_length=cfg.training.model.data_format.seq_length,
                    pre_generated_trajectory=generated_trajectory.to(device)
                )
    else:
        all_generated_trajectories = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                generated_trajectory_data = test_batch(
                    ref_model, model, batch, metric_callback, device, metric_items, 
                    dataset_type=dataset_type, model_type=model_type, 
                    seq_length=cfg.training.model.data_format.seq_length
                )
                
                if generated_trajectory_data is not None:
                    all_generated_trajectories.append(generated_trajectory_data)

        if all_generated_trajectories and model_type in ["ccdm", "et"]:
            torch.save(all_generated_trajectories, trajectory_save_path)
            logger.info(f"Saved {len(all_generated_trajectories)} generated trajectories to {trajectory_save_path}")

    metric_features = {metric_item: {"GT": None, "GEN": None} for metric_item in metric_items}
    for metric_item in metric_items:
        if metric_item in metric_callback.active_metrics:
            if metric_item in metric_callback.metrics and "clatr_prdc" in metric_callback.metrics[metric_item]:
                prdc = metric_callback.metrics[metric_item]["clatr_prdc"]
                if hasattr(prdc, "real_features") and prdc.real_features is not None:
                    metric_features[metric_item]["GT"] = prdc.real_features
                if hasattr(prdc, "fake_features") and prdc.fake_features is not None:
                    metric_features[metric_item]["GEN"] = prdc.fake_features
    
    
    save_dir = os.path.dirname(os.path.dirname(cfg.ref_model.inference.config))
    features_save_dir = os.path.join(save_dir, "features")
    os.makedirs(features_save_dir, exist_ok=True)
    features_save_path = os.path.join(features_save_dir, f"dataset_{dataset_type}_model_{model_type}.pth")
    
    torch.save(metric_features, features_save_path)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.tsne:
        for metric_item, features in metric_features.items():
            if features["GT"] is not None and features["GEN"] is not None:
                logger.info(f"Creating t-SNE visualization for {metric_item}")
                save_path = os.path.join(cfg.output_dir, f"embeddings_tSNE_{metric_item}.png")
                tSNE_visualize_embeddings(
                    features,
                    title=f"Embedding Visualization using t-SNE ({metric_item})",
                    save_path=save_path,
                )

        if model_type == "lens_craft" and dataset_type == "simulation" and "prompt_generation" in metric_items:
            movement_types = []
            
            if dataset_type == "simulation":
                for batch in test_dataloader:
                    batch_movement_types = []
                    for prompt_params in batch["cinematography_prompt_parameters"]:
                        movement_type = prompt_params[4][1]
                        batch_movement_types.append(movement_type)
                    movement_types.extend(batch_movement_types)
                
                logger.info(f"Extracted {len(movement_types)} movement types from the test dataset")

            if (metric_features["prompt_generation"]["GT"] is not None and 
                metric_features["prompt_generation"]["GEN"] is not None and 
                len(movement_types) > 0):
                
                logger.info("Creating t-SNE visualization colored by movement type")
                
                tSNE_visualize_embeddings_by_class_type(
                    caption_embeddings=metric_features["prompt_generation"]["GT"],
                    encoder_embeddings=metric_features["prompt_generation"]["GEN"],
                    class_types=movement_types,
                    title=f"Embedding Visualization using t-SNE (Colored by Movement Type)",
                    save_path=os.path.join(cfg.output_dir, "embeddings_tSNE_by_movement_type.png"),
                )
                
                logger.info(f"Movement type t-SNE visualization saved to {cfg.output_dir}/embeddings_tSNE_by_movement_type.png")

    metrics = {
        item: metric_callback.compute_clatr_metrics(item)
        for item in metric_items
        if item in metric_callback.active_metrics
    }

    logger.info(f"Final Metrics: {metrics}")


if __name__ == "__main__":
    main()