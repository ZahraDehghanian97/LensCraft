import os
import json
from typing import Any, Dict
import torch
from torch.utils.data import Dataset

from data.et.config import STANDARDIZATION_CONFIG_TORCH
from data.simulation.utils import fix_prompts_and_instructions, load_clip_means

from .load import load_et_dataset


class ETDataset(Dataset):
    def __init__(self, project_config_dir: str, dataset_dir: str, et_cin_lang_path: str, fill_none_with_mean: bool, 
                 clip_embeddings: Dict, set_name: str, split: str, normalize: bool):
        original_dataset = load_et_dataset(
            project_config_dir, dataset_dir, set_name, split)
        
        
        target_ids = None # {'2011_4lQ_MjU4QHw_00005_00005': 1328, '2011_KQM0klOXck8_00023_00000': 4891, '2012_ZryPGAMBuF4_00004_00002': 18583, '2014_pDEJr2Sqhxc_00005_00000': 24719, '2015_2ad7SgwLNXo_00014_00002': 25731, '2016_ux9JHznPT8E_00009_00001': 36731, '2017_JjfbxBMmXTI_00001_00000': 40366}
        
        self.original_dataset = [original_dataset[i] for i in target_ids.values()] if target_ids else original_dataset

        self.focal_length = self.original_dataset[0]['intrinsics'][0]
        self.fill_none_with_mean = fill_none_with_mean
        self.clip_embeddings = clip_embeddings
        self.normalize = normalize
        
        with open(et_cin_lang_path, 'r') as f:
            prompt_data = json.load(f)
            
        self.prompt_lookup = {item['custom_id']: item for item in prompt_data}

        if self.fill_none_with_mean:
            self.embedding_means = load_clip_means()
        else:
            self.embedding_means = None

    def __len__(self) -> int:
        return len(self.original_dataset)
    
    @staticmethod
    def normalize_item(camera_trajectory, subject_trajectory, subject_volume=None, normalize:bool= True):
        device = camera_trajectory.device
        camera_trajectory = camera_trajectory.clone()
        subject_trajectory = subject_trajectory.clone()
        
        if normalize:
            camera_trajectory[..., 0, 6:] -= STANDARDIZATION_CONFIG_TORCH["shift_mean"].to(device)
            camera_trajectory[..., 0, 6:] /= STANDARDIZATION_CONFIG_TORCH["shift_std"].to(device)

            camera_trajectory[..., 1:, 6:] -= STANDARDIZATION_CONFIG_TORCH["norm_mean"].to(device)
            camera_trajectory[..., 1:, 6:] /= STANDARDIZATION_CONFIG_TORCH["norm_std"].to(device)

            if len(STANDARDIZATION_CONFIG_TORCH["norm_mean_h"]) == 6:
                subject_trajectory[..., 0, :] -= STANDARDIZATION_CONFIG_TORCH["norm_mean_h"][:3].to(device)
                subject_trajectory[..., 0, :] /= STANDARDIZATION_CONFIG_TORCH["norm_std_h"][:3].to(device)

                subject_trajectory[..., 1:, :] -= STANDARDIZATION_CONFIG_TORCH["norm_mean_h"][3:].to(device)
                subject_trajectory[..., 1:, :] /= STANDARDIZATION_CONFIG_TORCH["norm_std_h"][3:].to(device)
            else:
                subject_trajectory -= STANDARDIZATION_CONFIG_TORCH["norm_mean_h"].to(device)
                subject_trajectory /= STANDARDIZATION_CONFIG_TORCH["norm_std_h"].to(device)

        else:
            camera_trajectory[..., 0, 6:] = (
                camera_trajectory[..., 0, 6:] * STANDARDIZATION_CONFIG_TORCH["shift_std"].to(device)
                + STANDARDIZATION_CONFIG_TORCH["shift_mean"].to(device)
            )

            camera_trajectory[..., 1:, 6:] = (
                camera_trajectory[..., 1:, 6:] * STANDARDIZATION_CONFIG_TORCH["norm_std"].to(device)
                + STANDARDIZATION_CONFIG_TORCH["norm_mean"].to(device)
            )

            if len(STANDARDIZATION_CONFIG_TORCH["norm_mean_h"]) == 6:
                subject_trajectory[..., 0, :] = (
                    subject_trajectory[..., 0, :]
                    * STANDARDIZATION_CONFIG_TORCH["norm_std_h"][:3].to(device)
                    + STANDARDIZATION_CONFIG_TORCH["norm_mean_h"][:3].to(device)
                )
                subject_trajectory[..., 1:, :] = (
                    subject_trajectory[..., 1:, :]
                    * STANDARDIZATION_CONFIG_TORCH["norm_std_h"][3:].to(device)
                    + STANDARDIZATION_CONFIG_TORCH["norm_mean_h"][3:].to(device)
                )
            else:
                subject_trajectory = (
                    subject_trajectory
                    * STANDARDIZATION_CONFIG_TORCH["norm_std_h"].to(device)
                    + STANDARDIZATION_CONFIG_TORCH["norm_mean_h"].to(device)
                )

        return camera_trajectory, subject_trajectory, subject_volume

    def _get_average_caption_feat(self, item):
        caption_feat = item['caption_feat']
        clip_seq_mask = item['caption_raw']['clip_seq_mask']
        clip_seq_mask = clip_seq_mask.bool().unsqueeze(0)
        valid_sum = (caption_feat * clip_seq_mask).sum(dim=1)
        num_valid_tokens = clip_seq_mask.sum().clamp(min=1)
        return valid_sum / num_valid_tokens

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.original_dataset[index]
        item_id = os.path.splitext(item['traj_filename'])[0]
        item_data = self.prompt_lookup[f"et-{item_id}"]
        
        instruction = item_data["simulationInstructions"][0]
        prompt = item_data["cinematographyPrompts"][0]
        
        simulation_instruction_tensor, cinematography_prompt_tensor, prompt_none_mask, simulation_instruction_parameters, cinematography_prompt_parameters = \
            fix_prompts_and_instructions(instruction, prompt, self.clip_embeddings, self.fill_none_with_mean, self.embedding_means)

        
        camera_trajectory = item["traj_feat"].permute(1, 0)
        subject_trajectory = item['char_feat'].permute(1, 0)
        
        if not self.normalize:
            camera_trajectory, subject_trajectory, _ = \
                ETDataset.normalize_item(camera_trajectory, subject_trajectory , None, False) # Checkme: is it need to premute inputs?
        
        caption_feat = self._get_average_caption_feat(item)
                
        return {
            'camera_trajectory': camera_trajectory,
            'subject_trajectory': subject_trajectory,
            'subject_volume': None,
            'padding_mask': ~item['padding_mask'].to(torch.bool),
            'caption_feat': caption_feat,
            'intrinsics': torch.tensor(item['intrinsics'], dtype=torch.float32),
            "simulation_instruction": simulation_instruction_tensor,
            "cinematography_prompt": cinematography_prompt_tensor,
            "simulation_instruction_parameters": simulation_instruction_parameters,
            "cinematography_prompt_parameters": cinematography_prompt_parameters,
            "prompt_none_mask": prompt_none_mask,
            "raw_prompt": prompt,
            "raw_instruction": instruction,
            'text_prompts': item['caption_raw']['caption'],
            'item_id': os.path.splitext(item['traj_filename'])[0],
        }


def collate_fn(batch):
    if len(batch) > 0 and batch[0]['subject_volume'] is None:
        subject_volume = None
    else:
        subject_volume = torch.stack([item["subject_volume"] for item in batch])
        
    if len(batch) > 0 and batch[0]['subject_trajectory'] is None:
        subject_trajectory = None
    else:
        subject_trajectory = torch.stack([item["subject_trajectory"] for item in batch])
        
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": subject_trajectory,
        "subject_volume": subject_volume,
        'padding_mask': torch.stack([item['padding_mask'] for item in batch]),
        'caption_feat': torch.stack([item['caption_feat'] for item in batch]),
        'intrinsics': torch.stack([item['intrinsics'] for item in batch]),
        "simulation_instruction": torch.stack([item["simulation_instruction"] for item in batch]).transpose(0, 1),
        "cinematography_prompt": torch.stack([item["cinematography_prompt"] for item in batch]).transpose(0, 1),
        "simulation_instruction_parameters": [
            item["simulation_instruction_parameters"] for item in batch
        ],
        "cinematography_prompt_parameters": [
            item["cinematography_prompt_parameters"] for item in batch
        ],
        "prompt_none_mask": torch.stack([item["prompt_none_mask"] for item in batch]),
        "raw_prompt": [item["raw_prompt"] for item in batch],
        "raw_instruction": [item["raw_instruction"] for item in batch],
        'text_prompts': [item['text_prompts'] for item in batch],
        'item_ids': [item['item_id'] for item in batch],
    }
