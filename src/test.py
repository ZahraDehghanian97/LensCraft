import logging
from typing import Dict, Optional, Literal, Any

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from models.camera_trajectory_model import MultiTaskAutoencoder
from data.datamodule import CameraTrajectoryDataModule
from metrics.callback import MetricCallback
from utils.checkpoint import load_checkpoint

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
    log_interval = 10
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            process_test_batch(model, batch, metric_callback, dataset_type, device)
            
            if (batch_idx + 1) % log_interval == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches")
    
    metric_items = (
        ["reconstruction", "prompt_generation", "hybrid_generation"] 
        if dataset_type == 'simulation' 
        else ["reconstruction"]
    )
    
    metrics = {
        item: metric_callback.compute_clatr_metrics(item) 
        for item in metric_items
    }
    
    logger.info(f"Final Metrics: {metrics}")


def prepare_batch_data(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:    
    prepared_data = {}
    
    for key in ["camera_trajectory", "subject_trajectory", "subject_volume", "padding_mask", "cinematography_prompt"]:
        if key in batch and batch[key] is not None:
            prepared_data[key] = batch[key].to(device)
        else:
            prepared_data[key] = None
    
    return prepared_data


def compute_subject_embedding(
    model: MultiTaskAutoencoder, 
    subject_trajectory: torch.Tensor, 
    subject_volume: torch.Tensor
) -> torch.Tensor:
    subject_embedding_loc_rot = model.subject_projection_loc_rot(subject_trajectory)
    subject_embedding_vol = model.subject_projection_vol(subject_volume)
    return torch.cat([subject_embedding_loc_rot, subject_embedding_vol], 1)


def compute_reference_embedding(
    model: MultiTaskAutoencoder, 
    camera_trajectory: torch.Tensor, 
    subject_embedding: torch.Tensor
) -> torch.Tensor:
    embedding = model.encoder(camera_trajectory, subject_embedding)
    return embedding[:model.memory_tokens_count, ...]


def generate_camera_trajectory(
    model: MultiTaskAutoencoder,
    data: Dict[str, torch.Tensor],
    memory_teacher_forcing_ratio: Optional[float] = None,
    caption_embedding: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    generation_params = {
        "subject_trajectory": data["subject_trajectory"],
        "subject_volume": data["subject_volume"],
        "camera_trajectory": data["camera_trajectory"],
        "padding_mask": data["padding_mask"]
    }
    
    if memory_teacher_forcing_ratio is not None:
        generation_params["memory_teacher_forcing_ratio"] = memory_teacher_forcing_ratio
    
    if caption_embedding is not None:
        generation_params["caption_embedding"] = caption_embedding
    
    return model.generate_camera_trajectory(**generation_params)


def update_metrics(
    model: MultiTaskAutoencoder,
    data: Dict[str, torch.Tensor],
    reference_embedding: torch.Tensor,
    metric_callback: MetricCallback,
    metric_name: str,
    memory_teacher_forcing_ratio: Optional[float] = None,
    caption_embedding: Optional[torch.Tensor] = None
) -> None:
    generation = generate_camera_trajectory(
        model,
        data,
        memory_teacher_forcing_ratio,
        caption_embedding
    )
    
    generation_embedding = generation['embeddings'][:model.memory_tokens_count, ...]
    
    batch_size = generation_embedding.shape[1]
    
    generation_embedding_reshaped = generation_embedding.permute(1, 0, 2).reshape(batch_size, -1).clone()
    reference_embedding_reshaped = reference_embedding.permute(1, 0, 2).reshape(batch_size, -1).clone()
    text_prompt_reshaped = caption_embedding.permute(1, 0, 2).reshape(batch_size, -1).clone() if caption_embedding != None else None
    
    metric_callback.update_clatr_metrics(
        metric_name,
        generation_embedding_reshaped,
        reference_embedding_reshaped,
        text_prompt_reshaped
    )


def process_test_batch(
    model: MultiTaskAutoencoder,
    batch: Dict[str, torch.Tensor],
    metric_callback: MetricCallback,
    dataset_type: DatasetType,
    device: torch.device
) -> None:
    prepared_data = prepare_batch_data(batch, device)
    
    subject_embedding = compute_subject_embedding(
        model, 
        prepared_data["subject_trajectory"], 
        prepared_data["subject_volume"]
    )
    
    reference_embedding = compute_reference_embedding(
        model, 
        prepared_data["camera_trajectory"], 
        subject_embedding
    )
    
    cinematography_prompt = prepared_data.get("cinematography_prompt")
    
    update_metrics(
        model, 
        prepared_data,
        reference_embedding,
        metric_callback,
        "reconstruction",
        memory_teacher_forcing_ratio=0,
        caption_embedding=cinematography_prompt
    )
    
    if dataset_type == 'simulation':
        update_metrics(
            model,
            prepared_data,
            reference_embedding,
            metric_callback,
            "prompt_generation",
            memory_teacher_forcing_ratio=1.0,
            caption_embedding=cinematography_prompt
        )
        
        update_metrics(
            model,
            prepared_data,
            reference_embedding,
            metric_callback,
            "hybrid_generation",
            memory_teacher_forcing_ratio=0.4,
            caption_embedding=cinematography_prompt
        )


if __name__ == "__main__":
    main()
