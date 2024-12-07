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
    caption_feat: Optional[torch.Tensor] = None

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

    def save_simulation_format(
        self,
        camera_trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor,
        output_path: str
    ):
        simulation_data = {
            "simulations": [{
                "subjects": [{
                    "position": {
                        "x": subject_trajectory[0, 0, 0].item(),
                        "y": subject_trajectory[0, 0, 1].item(),
                        "z": subject_trajectory[0, 0, 2].item()
                    },
                    "size": {
                        "x": subject_trajectory[0, 0, 3].item(),
                        "y": subject_trajectory[0, 0, 4].item(),
                        "z": subject_trajectory[0, 0, 5].item()
                    },
                    "rotation": {
                        "x": subject_trajectory[0, 0, 6].item(),
                        "y": subject_trajectory[0, 0, 7].item(),
                        "z": subject_trajectory[0, 0, 8].item()
                    }
                }],
                "cameraFrames": [
                    {
                        "position": {
                            "x": camera_trajectory[i, 0].item(),
                            "y": camera_trajectory[i, 1].item(),
                            "z": camera_trajectory[i, 2].item()
                        },
                        "focalLength": camera_trajectory[i, 3].item(),
                        "angle": {
                            "x": camera_trajectory[i, 4].item(),
                            "y": camera_trajectory[i, 5].item(),
                            "z": camera_trajectory[i, 6].item()
                        }
                    }
                    for i in range(camera_trajectory.size(0))
                ],
                "instructions": [{
                    "frameCount": camera_trajectory.size(0),
                    "cameraMovement": "dolly",
                    "movementEasing": "linear",
                    "initialCameraAngle": "mediumAngle",
                    "initialShotType": "mediumShot"
                }]
            }]
        }
        
        with open(output_path, 'w') as f:
            json.dump(simulation_data, f, indent=2)