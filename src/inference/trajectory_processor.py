from pathlib import Path
import torch
import shutil
import json
import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class TrajectoryData:
    subject_trajectory: torch.Tensor
    camera_trajectory: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    src_key_mask: Optional[torch.Tensor] = None
    caption_feat: Optional[torch.Tensor] = None
    embedding_masks: Optional[torch.Tensor] = None
    teacher_forcing_ratio: Optional[int] = 0.0

class TrajectoryProcessor:
    def __init__(self, output_dir: str, dataset_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir

    def prepare_output_directory(self, sample_id: Optional[str] = None) -> str:
        dir_path = os.path.join(self.output_dir, sample_id if sample_id else 'simulation_output')
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def copy_dataset_files(self, sample_id: str, output_dir: str):
        if self.dataset_dir:
            shutil.copy2(self.dataset_dir / 'char' / f"{sample_id}.npy", output_dir / "char.npy")
            shutil.copy2(self.dataset_dir / 'traj' / f"{sample_id}.txt", output_dir / "traj.txt")
            
            
    