import os
import json
from typing import Any, Dict, Optional
import torch
from torch.utils.data import Dataset

from data.et.config import STANDARDIZATION_CONFIG_TORCH
from data.categorical_conditioning import metadata_from_prompt
from data.collate_utils import collate_structured_conditioning, collate_trajectories

from .load import load_et_dataset


class ETDataset(Dataset):
    def __init__(self, project_config_dir: str, dataset_dir: str, et_cin_lang_path: str,
                 set_name: str, split: str, normalize: bool = True):
        self.original_dataset = load_et_dataset(
            project_config_dir, dataset_dir, set_name, split)

        self.focal_length = self.original_dataset[0]['intrinsics'][0]
        self.normalize = normalize

        with open(et_cin_lang_path, 'r') as f:
            prompt_data = json.load(f)

        self.prompt_lookup = {item['custom_id']: item for item in prompt_data}

    def __len__(self) -> int:
        return len(self.original_dataset)

    @staticmethod
    def normalize_item(
        camera_trajectory: torch.Tensor,
        subject_trajectory: Optional[torch.Tensor],
        subject_volume: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ):
        device = camera_trajectory.device
        camera_trajectory = camera_trajectory.clone()
        if subject_trajectory is not None:
            subject_trajectory = subject_trajectory.clone()

        if normalize:
            camera_trajectory[..., 0, 6:] -= STANDARDIZATION_CONFIG_TORCH["shift_mean"].to(device)
            camera_trajectory[..., 0, 6:] /= STANDARDIZATION_CONFIG_TORCH["shift_std"].to(device)

            camera_trajectory[..., 1:, 6:] -= STANDARDIZATION_CONFIG_TORCH["norm_mean"].to(device)
            camera_trajectory[..., 1:, 6:] /= STANDARDIZATION_CONFIG_TORCH["norm_std"].to(device)

            if subject_trajectory is not None:
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

            if subject_trajectory is not None:
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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.original_dataset[index]
        item_id = os.path.splitext(item['traj_filename'])[0]
        item_data = self.prompt_lookup[f"et-{item_id}"]

        instruction = item_data["simulationInstructions"][0]
        prompt = item_data["cinematographyPrompts"][0]

        conditioning = metadata_from_prompt(prompt, instruction)

        camera_trajectory = item["traj_feat"].permute(1, 0)
        subject_trajectory = item['char_feat'].permute(1, 0)

        if not self.normalize:
            camera_trajectory, subject_trajectory, _ = ETDataset.normalize_item(
                camera_trajectory, subject_trajectory, None, False
            )

        result = {
            'camera_trajectory': camera_trajectory,
            'subject_trajectory': subject_trajectory,
            'subject_volume': None,
            'padding_mask': ~item['padding_mask'].to(torch.bool),
            'intrinsics': torch.tensor(item['intrinsics'], dtype=torch.float32),
            **conditioning,
            "raw_prompt": prompt,
            "raw_instruction": instruction,
            'text_prompts': item['caption_raw']['caption'],
            'item_id': os.path.splitext(item['traj_filename'])[0],
        }
        return result


def collate_fn(batch):
    conditioning = collate_structured_conditioning(batch)
    result = {
        **collate_trajectories(batch),
        'intrinsics': torch.stack([item['intrinsics'] for item in batch]),
        **conditioning,
        "raw_prompt": [item["raw_prompt"] for item in batch],
        "raw_instruction": [item["raw_instruction"] for item in batch],
        'text_prompts': [item['text_prompts'] for item in batch],
        'item_ids': [item['item_id'] for item in batch],
    }
    return result
