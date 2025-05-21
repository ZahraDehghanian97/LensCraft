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


def inference_batch(model, batch, device, dataset_type='simulation', model_type='lens_craft', seq_length=30):
    batch = to_cuda(batch, device)
    batch_size = len(batch["text_prompts"])
        
    caption_embedding = batch.get("cinematography_prompt", None) if dataset_type in ["simulation", "et"] else None
    
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
        
    trajectory, subject_trajectory, subject_volume, padding_mask = convert_to_target(
        dataset_type,
        model_type,
        batch["camera_trajectory"],
        batch["subject_trajectory"],
        batch["subject_volume"],
        batch["padding_mask"],
        seq_length,
        torch.full((batch_size,), 30, device=device) # fix me for other datasets
    )
    
    if model_type in ["ccdm", "et"]:
        generated_trajecotry = model.generate_using_text(
            batch["text_prompts"],
            subject_trajectory,
            trajectory,
            padding_mask
        )
                
        sim_generated_trajectory, _, _, _ = convert_to_target(
            model_type,
            "simulation",
            generated_trajecotry,
            subject_trajectory,
            batch["subject_volume"],
            padding_mask,
            30
        )
        return {'prompt_generation': sim_generated_trajectory}
        
    results = {}
    for mode in ['prompt_generation', 'reconstruction', 'key_framing+prompt', 'key_framing', 'source_trajectory']:
        current_caption_embedding = caption_embedding
        if mode == 'reconstruction':
            memory_teacher_forcing_ratio = 0
        elif mode == 'key_framing':
            template = torch.cat([torch.ones(26, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)])
            padding_mask = torch.zeros((batch_size, 30), dtype=torch.bool, device=device)
            for i in range(batch_size):
                shuffled = template[torch.randperm(30)]
                padding_mask[i] = shuffled
            batch["padding_mask"] = padding_mask
            memory_teacher_forcing_ratio = 0
        elif mode == 'prompt_generation':
            memory_teacher_forcing_ratio = 1
        elif mode == 'key_framing+prompt':
            template = torch.cat([torch.ones(26, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)])
            padding_mask = torch.zeros((batch_size, 30), dtype=torch.bool, device=device)
            for i in range(batch_size):
                shuffled = template[torch.randperm(30)]
                padding_mask[i] = shuffled
            batch["padding_mask"] = padding_mask
            memory_teacher_forcing_ratio = 0.5
        elif mode == 'source_trajectory':
            memory_teacher_forcing_ratio = 0.5
            random_indices = batch["random_prompt_index"]
            current_caption_embedding = []
            for idx in random_indices:
                current_caption_embedding.append(batch["cinematography_prompt"][:, idx, :])
            current_caption_embedding = torch.stack(current_caption_embedding, dim=1).to(device)
        
        output = model.generate_camera_trajectory(
            subject_trajectory=sim_subject_trajectory,
            subject_volume=sim_subject_volume,
            camera_trajectory=sim_camera_trajectory,
            padding_mask=sim_padding_mask,
            memory_teacher_forcing_ratio=memory_teacher_forcing_ratio,
            caption_embedding=current_caption_embedding
        )
        results[mode] = output["reconstructed"]
    
    return results
