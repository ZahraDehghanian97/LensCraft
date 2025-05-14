from typing import Dict

import torch

from data.convertor.convertor import convert_to_target


def to_cuda(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    prepared_data = {}
    
    for key in batch.keys():
        prepared_data[key] = batch[key]
        if prepared_data[key] is not None and torch.is_tensor(prepared_data[key]):
                prepared_data[key] = prepared_data[key].to(device)
    
    return prepared_data


def test_batch(sim_model, model, batch, metric_callback, device, metric_items, dataset_type='simulation', model_type='simulation', seq_length=30):
    batch = to_cuda(batch, device)
    batch_size = len(batch["text_prompts"])
    
    for metric_item in metric_items:
        if metric_item == 'reconstruction':
            memory_teacher_forcing_ratio = 0
        elif metric_item == 'prompt_generation':
            memory_teacher_forcing_ratio = 1
        elif metric_item == 'hybrid_generation':
            memory_teacher_forcing_ratio = 0.5
        
        caption_embedding = None
        if dataset_type == "simulation":
            caption_embedding = batch.get("cinematography_prompt", None)
        
        
        if dataset_type != "simulation":
            sim_camera_trajectory, sim_subject_trajectory, sim_subject_volume, sim_padding_mask = convert_to_target(
                dataset_type,
                "simulation",
                batch["camera_trajectory"],
                batch["subject_trajectory"],
                batch["subject_volume"],
                batch["padding_mask"],
                30
            )
        else:
            sim_camera_trajectory, sim_subject_trajectory, sim_subject_volume, sim_padding_mask = \
                batch["camera_trajectory"], batch["subject_trajectory"], batch["subject_volume"], batch["padding_mask"]
        
        sim_output = sim_model.generate_camera_trajectory(
            subject_trajectory=sim_subject_trajectory,
            subject_volume=sim_subject_volume,
            camera_trajectory=sim_camera_trajectory,
            padding_mask=sim_padding_mask,
            memory_teacher_forcing_ratio=memory_teacher_forcing_ratio,
            caption_embedding=caption_embedding
        )
        
        decoder_memory = sim_output['embeddings'][:sim_model.memory_tokens_count, ...]
        subject_embedding = sim_output['subject_embedding']
        decoder_memory = decoder_memory.permute(1, 0, 2).reshape(batch_size, -1).clone()
        
        if model_type in ["ccdm", "et"] and dataset_type == "simulation":
            trajectory, subject_trajectory, _, padding_mask = convert_to_target(
                dataset_type,
                model_type,
                batch["camera_trajectory"],
                batch["subject_trajectory"],
                batch["subject_volume"],
                batch["padding_mask"],
                seq_length
            )
            generated_trajecotry = model.generate_using_text(
                batch["text_prompts"],
                subject_trajectory,
                trajectory
                # (~padding_mask).sum(dim=1)
            )
            
            sim_generated_trajectory, _, _, _ = convert_to_target(
                model_type,
                "simulation",
                generated_trajecotry,
                batch["subject_trajectory"],
                batch["subject_volume"],
                batch["padding_mask"],
                seq_length
            )
            
            reconstructed_memory = sim_model.encoder(
                sim_generated_trajectory, 
                subject_embedding
            )[:sim_model.memory_tokens_count, ...].detach().clone()
        
        elif model_type == "simulation":
            reconstructed_memory = model.encoder(
                sim_output["reconstructed"], 
                subject_embedding
            )[:model.memory_tokens_count, ...].detach().clone()
            
        reconstructed_memory = reconstructed_memory.permute(1, 0, 2).reshape(batch_size, -1).clone()
        
        if caption_embedding is not None:
            caption_embedding = caption_embedding.permute(1, 0, 2).reshape(batch_size, -1).clone()
        
        metric_callback.update_clatr_metrics(
            metric_item,
            reconstructed_memory,
            decoder_memory,
            caption_embedding
        )
