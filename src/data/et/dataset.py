import torch
from typing import Any, Dict
from torch.utils.data import Dataset

from data.convertor.convertor import convert_to_target

from .load import load_et_dataset


class ETDataset(Dataset):
    def __init__(self, project_config_dir: str, dataset_dir: str, set_name: str, split: str, target = None):
        self.original_dataset = load_et_dataset(
            project_config_dir, dataset_dir, set_name, split)
        self.focal_length = self.original_dataset[0]['intrinsics'][0]
        self.target = target # TODO

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        original_item = self.original_dataset[index]
        return self.process_item(original_item)

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        caption_feat = item['caption_feat']
        clip_seq_mask = item['caption_raw']['clip_seq_mask']

        clip_seq_mask = clip_seq_mask.bool().unsqueeze(0)

        valid_sum = (caption_feat * clip_seq_mask).sum(dim=1)
        num_valid_tokens = clip_seq_mask.sum().clamp(min=1)
        averaged_caption_feat = valid_sum / num_valid_tokens
        
        camera_trajectory = item["traj_feat"]
        subject_trajectory = item['char_feat']
        subject_volume = None
        padding_mask = item['padding_mask'].to(torch.bool)
        
        if "type" in self.target and self.target["type"] != "et":
            camera_trajectory, subject_trajectory, subject_volume, padding_mask = convert_to_target(
                source="et",
                target=self.target["type"],
                trajectory=camera_trajectory,
                subject_trajectory=subject_trajectory,
                subject_volume=subject_volume,
                padding_mask=padding_mask,
                target_len=self.target.get("seq_length", 30)
            )
        
        return {
            'camera_trajectory': camera_trajectory,
            'subject_trajectory': subject_trajectory,
            'subject_volume': subject_volume,
            'padding_mask': padding_mask,
            'caption_feat': averaged_caption_feat,
            'intrinsics': torch.tensor(item['intrinsics'], dtype=torch.float32),
            "original_camera_trajectory": item['traj_feat'],
            "text_prompts": item["caption_raw"]["caption"]
        }


def collate_fn(batch):
    return {
        'camera_trajectory': torch.stack([item['camera_trajectory'] for item in batch]),
        'subject_trajectory': torch.stack([item['subject_trajectory'] for item in batch]),
        'subject_volume': torch.stack([item["subject_volume"] for item in batch]),
        'padding_mask': torch.stack([item['padding_mask'] for item in batch]),
        'caption_feat': torch.stack([item['caption_feat'] for item in batch]),
        'intrinsics': torch.stack([item['intrinsics'] for item in batch]),
        'text_prompts': [item['text_prompts'] for item in batch],
        'original_camera_trajectory': torch.stack([item["original_camera_trajectory"] for item in batch]),

    }
