from testing.metrics.callback import MetricCallback
from models.camera_trajectory_model import MultiTaskAutoencoder
from typing import Dict, Optional, Any
import torch

from models.camera_trajectory_model import MultiTaskAutoencoder
from testing.metrics.callback import MetricCallback


def prepare_batch_data(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    prepared_data = {}
    
    for key in ["camera_trajectory", "subject_trajectory", "subject_volume", "padding_mask", "cinematography_prompt"]:
        if key in batch and batch[key] is not None:
            prepared_data[key] = batch[key].to(device)
        else:
            prepared_data[key] = None
    
    return prepared_data

def update_metrics(
    model: MultiTaskAutoencoder,
    data: Dict[str, torch.Tensor],
    metric_callback: MetricCallback,
    metric_name: str,
    memory_teacher_forcing_ratio: Optional[float] = None,
    caption_embedding: Optional[torch.Tensor] = None
) -> None:
    generation_output = model.generate_camera_trajectory(
        subject_trajectory=data["subject_trajectory"],
        subject_volume=data["subject_volume"],
        camera_trajectory=data["camera_trajectory"],
        padding_mask=data["padding_mask"],
        memory_teacher_forcing_ratio=memory_teacher_forcing_ratio,
        caption_embedding=caption_embedding
    )
    
    encoder_embedding = generation_output['embeddings'][:model.memory_tokens_count, ...]

    subject_embedding = torch.cat([
        model.subject_projection_loc_rot(data["subject_trajectory"]),
        model.subject_projection_vol(data["subject_volume"])
    ], 1)
    
    reconstructed_embedding = model.encoder(
        generation_output["reconstructed"], 
        subject_embedding
    )[:model.memory_tokens_count, ...].detach().clone()    
    
    batch_size = reconstructed_embedding.shape[1]
    
    reconstructed_embedding_reshaped = reconstructed_embedding.permute(1, 0, 2).reshape(batch_size, -1).clone()
    encoder_embedding_reshaped = encoder_embedding.permute(1, 0, 2).reshape(batch_size, -1).clone()
    
    caption_embedding_reshaped = None
    if caption_embedding is not None:
        caption_embedding_reshaped = caption_embedding.permute(1, 0, 2).reshape(batch_size, -1).clone()
    
    metric_callback.update_clatr_metrics(
        metric_name,
        reconstructed_embedding_reshaped,
        encoder_embedding_reshaped,
        caption_embedding_reshaped
    )

def process_lens_craft_batch(
    model: MultiTaskAutoencoder,
    batch: Dict[str, torch.Tensor],
    metric_callback: MetricCallback,
    dataset_type: str,
    device: torch.device
) -> None:
    prepared_data = prepare_batch_data(batch, device)
    
    caption_embedding = prepared_data.get("cinematography_prompt")
    
    update_metrics(
        model,
        prepared_data,
        metric_callback,
        "reconstruction",
        memory_teacher_forcing_ratio=0,
        caption_embedding=caption_embedding
    )
    
    if dataset_type == 'simulation':
        update_metrics(
            model,
            prepared_data,
            metric_callback,
            "prompt_generation",
            memory_teacher_forcing_ratio=1.0,
            caption_embedding=caption_embedding
        )
        
        update_metrics(
            model,
            prepared_data,
            metric_callback,
            "hybrid_generation",
            memory_teacher_forcing_ratio=0.4,
            caption_embedding=caption_embedding
        )
