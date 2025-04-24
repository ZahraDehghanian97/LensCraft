import logging
import os
from typing import Literal

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from models.camera_trajectory_model import MultiTaskAutoencoder
from data.datamodule import CameraTrajectoryDataModule
from testing.metrics.callback import MetricCallback
from utils.checkpoint import load_checkpoint
from visualization import tSNE_visualize_embeddings  
from models.ccdm_adapter import CCDMAdapter
from testing.ccdm import process_ccdm_batch
from testing.lens_craft import process_lens_craft_batch

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

DatasetType = Literal["ccdm", "et", "simulation"]

@hydra.main(version_base=None, config_path="../config", config_name="test")
def main(cfg: DictConfig) -> None:
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    
    device = torch.device(cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")
    
    use_ccdm = cfg.get("use_ccdm", False)
    ccdm_adapter = None
    
    if use_ccdm:
        if CCDMAdapter is None:
            raise ImportError("CCDM adapter not found. Please ensure models/ccdm_adapter.py exists.")
        logger.info("Using CCDM model for inference")
        ccdm_adapter = CCDMAdapter(cfg, device)
    else:
        logger.info("Using MultiTaskAutoencoder model for inference")
        model: MultiTaskAutoencoder = instantiate(cfg.training.model)
        model = load_checkpoint(cfg.checkpoint_path, model, device)
        model.to(device)
        model.eval()
    
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size
    )
    data_module.setup()
    
    metric_callback = MetricCallback(num_cams=1, device=device)
    
    target = cfg.data.dataset.config['_target_']
    dataset_type = "ccdm" if 'CCDMDataset' in target else "et" if 'ETDataset' in target else "simulation"
    
    test_dataloader = data_module.test_dataloader()
    
    if use_ccdm:
        metric_items = (
            ["reconstruction", "prompt_generation", "hybrid_generation"] 
            if dataset_type == 'simulation' or dataset_type == 'ccdm'
            else ["reconstruction"]
        )
    else:
        metric_items = (
            ["reconstruction", "prompt_generation", "hybrid_generation"] 
            if dataset_type == 'simulation' 
            else ["reconstruction"]
        )
    
    metric_features = {metric_type: {"GT": None, "GEN": None} for metric_type in metric_items}
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            if use_ccdm:
                process_ccdm_batch(ccdm_adapter, batch, metric_callback, device)
            else:
                process_lens_craft_batch(model, batch, metric_callback, dataset_type, device)
        
        for metric_type in metric_items:
            if metric_type in metric_callback.active_metrics:
                if metric_type in metric_callback.metrics and "clatr_prdc" in metric_callback.metrics[metric_type]:
                    prdc = metric_callback.metrics[metric_type]["clatr_prdc"]
                    if hasattr(prdc, 'real_features') and prdc.real_features is not None:
                        metric_features[metric_type]["GT"] = prdc.real_features
                    if hasattr(prdc, 'fake_features') and prdc.fake_features is not None:
                        metric_features[metric_type]["GEN"] = prdc.fake_features
    
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_type, features in metric_features.items():
        if features["GT"] is not None and features["GEN"] is not None:
            logger.info(f"Creating t-SNE visualization for {metric_type}")
            save_path = os.path.join(output_dir, f"embeddings_tSNE_{metric_type}.png")
            tSNE_visualize_embeddings(
                features,
                title=f'Embedding Visualization using t-SNE ({metric_type})',
                save_path=save_path
            )
    
    metrics = {
        item: metric_callback.compute_clatr_metrics(item) 
        for item in metric_items if item in metric_callback.active_metrics
    }
    
    logger.info(f"Final Metrics: {metrics}")


if __name__ == "__main__":
    main()
