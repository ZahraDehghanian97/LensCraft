import torch
from typing import Any, Dict
from torch.utils.data import Dataset
from .load import load_et_dataset


class ETDataset(Dataset):
    def __init__(self, project_config_dir: str, dataset_dir: str, set_name: str, split: str):
        self.original_dataset = load_et_dataset(
            project_config_dir, dataset_dir, set_name, split)

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        original_item = self.original_dataset[index]
        return self.process_item(original_item)

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        processed_item = {
            'traj_feat': item['traj_feat'],
            'padding_mask': item['padding_mask'],
            'char_feat': item['char_feat'],
            'caption_feat': item['caption_feat'],
            'intrinsics': torch.tensor(item['intrinsics'], dtype=torch.float32)
        }
        return processed_item


def et_batch_collate(batch):
    return {
        'traj_feat': torch.stack([item['traj_feat'] for item in batch]),
        'padding_mask': torch.stack([item['padding_mask'] for item in batch]),
        'char_feat': torch.stack([item['char_feat'] for item in batch]),
        'caption_feat': torch.stack([item['caption_feat'] for item in batch]),
        'intrinsics': torch.stack([item['intrinsics'] for item in batch])
    }
