from typing import Optional
import torch


class GenDoPDataset:
    @staticmethod
    def normalize_item(
        camera_trajectory: torch.Tensor,
        subject_trajectory: Optional[torch.Tensor] = None,
        subject_volume: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ):
        return camera_trajectory, subject_trajectory, subject_volume
