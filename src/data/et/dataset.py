import torch
import os
from typing import Any, Dict
from torch.utils.data import Dataset

from .load import load_et_dataset


class ETDataset(Dataset):
    def __init__(self, project_config_dir: str, dataset_dir: str, set_name: str, split: str):
        self.original_dataset = load_et_dataset(
            project_config_dir, dataset_dir, set_name, split)
        self.focal_length = self.original_dataset[0]['intrinsics'][0]

    def __len__(self) -> int:
        return len(self.original_dataset)

    def _get_average_caption_feat(self, item):
        caption_feat = item['caption_feat']
        clip_seq_mask = item['caption_raw']['clip_seq_mask']
        clip_seq_mask = clip_seq_mask.bool().unsqueeze(0)
        valid_sum = (caption_feat * clip_seq_mask).sum(dim=1)
        num_valid_tokens = clip_seq_mask.sum().clamp(min=1)
        return valid_sum / num_valid_tokens

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.original_dataset[index]
        caption_feat = self._get_average_caption_feat(item)
                
        return {
            'camera_trajectory': item["traj_feat"].permute(1, 0),
            'subject_trajectory': item['char_feat'].permute(1, 0),
            'subject_volume': None,
            'padding_mask': ~item['padding_mask'].to(torch.bool),
            'caption_feat': caption_feat,
            'intrinsics': torch.tensor(item['intrinsics'], dtype=torch.float32),
            'text_prompts': item['caption_raw']['caption'],
            'item_id': os.path.splitext(item['traj_filename'])[0]
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
        'text_prompts': [item['text_prompts'] for item in batch],
        'item_ids': [item['item_id'] for item in batch]
    }
