from typing import Any, Dict, List, Optional

import torch

from data.convertor.convertor import convert_to_target


def to_cuda(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    prepared_data: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value is not None and torch.is_tensor(value):
            prepared_data[key] = value.to(device)
        else:
            prepared_data[key] = value
    return prepared_data


def _memory_teacher_forcing_for(metric_item: str) -> float:
    if metric_item == "reconstruction":
        return 0.0
    if metric_item == "key_framing":
        return 0.0
    if metric_item == "prompt_generation":
        return 1.0
    if metric_item == "key_framing+prompt":
        return 0.5
    if metric_item == "hybrid_generation":
        return 0.5
    raise ValueError(f"Unknown metric item: {metric_item}")


def test_batch(ref_model, model, batch, metric_callback, device, metric_items, dataset_type='simulation', model_type='lens_craft', seq_length=30, pre_generated_trajectory=None):

    batch = to_cuda(batch, device)
    batch_size = len(batch["text_prompts"])

    generated_trajectory_data: Optional[torch.Tensor] = None

    if dataset_type != "simulation":
        (
            sim_camera_trajectory,
            sim_subject_trajectory,
            sim_subject_volume,
            sim_padding_mask,
        ) = convert_to_target(
            dataset_type,
            "simulation",
            batch["camera_trajectory"],
            batch["subject_trajectory"],
            batch["subject_volume"],
            batch["padding_mask"],
            30,
        )
    else:
        sim_camera_trajectory = batch["camera_trajectory"]
        sim_subject_trajectory = batch["subject_trajectory"]
        sim_subject_volume = batch["subject_volume"]
        sim_padding_mask = batch["padding_mask"]

    (
        camera_trajectory,
        subject_trajectory,
        subject_volume,
        padding_mask,
    ) = convert_to_target(
        dataset_type,
        model_type,
        batch["camera_trajectory"],
        batch["subject_trajectory"],
        batch["subject_volume"],
        batch["padding_mask"],
        seq_length,
        torch.full((batch_size,), 30, device=device), # fix me for other datasets
    )

    for metric_item in metric_items:
        caption_embedding = (
            batch.get("cinematography_prompt", None)
            if dataset_type in ("simulation", "et")
            else None
        )
        memory_teacher_forcing_ratio = _memory_teacher_forcing_for(metric_item)

        if metric_item in ("key_framing", "key_framing+prompt"):
            batch["padding_mask"] = torch.rand((batch_size, 30), device=device) > 1.0 / 6.0

        ref_output = ref_model.generate_camera_trajectory(
            subject_trajectory=sim_subject_trajectory,
            subject_volume=sim_subject_volume,
            camera_trajectory=sim_camera_trajectory,
            padding_mask=sim_padding_mask,
            memory_teacher_forcing_ratio=memory_teacher_forcing_ratio,
            caption_embedding=caption_embedding,
        )

        decoder_memory = ref_output['embeddings'][:ref_model.memory_tokens_count, ...]
        subject_embedding = ref_output['subject_embedding']
        decoder_memory = decoder_memory.permute(1, 0, 2).reshape(batch_size, -1).clone()
        
        if model_type in ["ccdm", "et"]:
            if pre_generated_trajectory is not None:
                generated_trajecotry = pre_generated_trajectory
            else:
                generated_trajecotry = model.generate_using_text(
                    batch["text_prompts"],
                    subject_trajectory,
                    camera_trajectory,
                    padding_mask,
                )

                generated_trajectory_data = generated_trajecotry.detach().cpu()
            
            sim_generated_trajectory, _, _, _ = convert_to_target(
                model_type,
                "simulation",
                generated_trajecotry,
                subject_trajectory,
                batch["subject_volume"],
                padding_mask,
                30,
            )
        elif model_type == "lens_craft":
            sim_generated_trajectory = ref_output["reconstructed"]

        reconstructed_memory = ref_model.encoder(
            sim_generated_trajectory, 
            subject_embedding
        )[:ref_model.memory_tokens_count, ...].detach().clone()
            
        reconstructed_memory = reconstructed_memory.permute(1, 0, 2).reshape(batch_size, -1).clone()
        
        if caption_embedding is not None:
            caption_embedding = caption_embedding.permute(1, 0, 2).reshape(batch_size, -1).clone()
        
        metric_callback.update_clatr_metrics(
            metric_item,
            reconstructed_memory,
            decoder_memory,
            caption_embedding
        )

    return generated_trajectory_data
