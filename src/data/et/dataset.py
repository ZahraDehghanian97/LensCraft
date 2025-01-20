import torch
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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        original_item = self.original_dataset[index]
        '''raw_trans = original_item['traj_feat'][6:].permute(1, 0)
        velocity = raw_trans[1:] - raw_trans[:-1]
        raw_trans[1:] = velocity  
        original_item['traj_feat'][6:] = raw_trans.permute(1, 0)'''
        return self.process_item(original_item)

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        subject_trajectory = self.char_feat_to_subject_trajectory(
            item['char_feat'])

        caption_feat = item['caption_feat']
        clip_seq_mask = item['caption_raw']['clip_seq_mask']

        clip_seq_mask = clip_seq_mask.bool().unsqueeze(0)

        valid_sum = (caption_feat * clip_seq_mask).sum(dim=1)
        num_valid_tokens = clip_seq_mask.sum().clamp(min=1)
        averaged_caption_feat = valid_sum / num_valid_tokens

        processed_item = {
            'camera_trajectory': item['traj_feat'].transpose(0, 1),
            'subject_trajectory': subject_trajectory,
            'padding_mask': ~item['padding_mask'].to(torch.bool),
            'caption_feat': averaged_caption_feat,
            'intrinsics': torch.tensor(item['intrinsics'], dtype=torch.float32)
        }
        return processed_item

    def char_feat_to_subject_trajectory(self, char_feat: torch.Tensor) -> torch.Tensor:
        subject_trajectory = []
        char_positions = char_feat[:3].transpose(0, 1)

        for pos in char_positions:
            subject_frame = [
                pos[0].item(), pos[1].item(), pos[2].item(),
                0.5, 1.7, 0.3,  # Default size values
                0, 0, 0  # Default rotation values
            ]
            subject_trajectory.append(subject_frame)

        return torch.tensor(subject_trajectory, dtype=torch.float32)


def collate_fn(batch):
    return {
        'camera_trajectory': torch.stack([item['camera_trajectory'] for item in batch]),
        'subject_trajectory': torch.stack([item['subject_trajectory'] for item in batch]),
        'padding_mask': torch.stack([item['padding_mask'] for item in batch]),
        'caption_feat': torch.stack([item['caption_feat'] for item in batch]),
        'intrinsics': torch.stack([item['intrinsics'] for item in batch])
    }
