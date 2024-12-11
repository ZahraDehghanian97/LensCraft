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
    teacher_forcing_ratio: Optional[int] = 0.5

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
            
            
    def generate_simulation_format(self, camera, subject):
        return {
            "subjects": [{
                "position": {
                    "x": subject[0, 0].item(),
                    "y": subject[0, 1].item(),
                    "z": subject[0, 2].item()
                },
                "size": {
                    "x": subject[0, 3].item(),
                    "y": subject[0, 4].item(),
                    "z": subject[0, 5].item()
                },
                "rotation": {
                    "x": subject[0, 6].item(),
                    "y": subject[0, 7].item(),
                    "z": subject[0, 8].item()
                }
            }],
            "cameraFrames": [
                {
                    "position": {
                        "x": camera[i, 0].item(),
                        "y": camera[i, 1].item(),
                        "z": camera[i, 2].item()
                    },
                    "focalLength": camera[i, 3].item(),
                    "angle": {
                        "x": camera[i, 4].item(),
                        "y": camera[i, 5].item(),
                        "z": camera[i, 6].item()
                    }
                }
                for i in range(camera.size(0))
            ],
            "instructions": [{
                "frameCount": camera.size(0),
                "cameraMovement": "",
                "movementEasing": "",
                "initialCameraAngle": "",
                "initialShotType": ""
            }]
        }

    def save_simulation_format(self, data, output_dir):
        for i, item in enumerate(data):
            output_path = os.path.join(output_dir, f'simulation-out-{i}.json')
            simulation_data = {
                "simulations": [self.generate_simulation_format(item['camera'], item['subject']),
                                self.generate_simulation_format(item['rec'], item['subject']),
                                self.generate_simulation_format(item['gen'], item['subject'])]
            }
            
            with open(output_path, 'w') as f:
                json.dump(simulation_data, f, indent=2)