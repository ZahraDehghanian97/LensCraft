import torch
from typing import Any, Dict, List
from torch.utils.data import Dataset

from utils.calaculation3d import euler_from_matrix, rotation_6d_to_matrix
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
        return self.process_item(original_item)

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        camera_trajectory = self.traj_feat_to_camera_trajectory(
            item['traj_feat'])
        subject_trajectory = self.char_feat_to_subject_trajectory(
            item['char_feat'])

        processed_item = {
            'camera_trajectory': camera_trajectory,
            'subject_trajectory': subject_trajectory,
            'padding_mask': item['padding_mask'],
            'caption_feat': item['caption_feat'],
            'intrinsics': torch.tensor(item['intrinsics'], dtype=torch.float32)
        }
        return processed_item

    def traj_feat_to_camera_trajectory(self, traj_feat: torch.Tensor) -> torch.Tensor:
        camera_trajectory = []
        for frame in range(traj_feat.shape[1]):
            rotation_6d = traj_feat[:6, frame]
            translation = traj_feat[6:, frame]

            rotation_matrix = rotation_6d_to_matrix(rotation_6d)
            euler_angles = euler_from_matrix(rotation_matrix)

            camera_frame = [
                translation[0].item(),
                translation[1].item(),
                translation[2].item(),
                self.focal_length,
                euler_angles[0].item(),
                euler_angles[1].item(),
                euler_angles[2].item()
            ]
            camera_trajectory.append(camera_frame)

        return torch.tensor(camera_trajectory, dtype=torch.float32)

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


def et_batch_collate(batch):
    return {
        'camera_trajectory': torch.stack([item['camera_trajectory'] for item in batch]),
        'subject_trajectory': torch.stack([item['subject_trajectory'] for item in batch]),
        'padding_mask': torch.stack([item['padding_mask'] for item in batch]),
        'caption_feat': torch.stack([item['caption_feat'] for item in batch]),
        'intrinsics': torch.stack([item['intrinsics'] for item in batch])
    }
