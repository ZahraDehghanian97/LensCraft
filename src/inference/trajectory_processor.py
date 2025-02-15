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
            
            
    def generate_simulation_format(self, camera, subject, helper_keyframes=None):
        res = {
            "subjects": [{
                "frames": [{
                    "position": {
                        "x": frame[0].item(),
                        "y": frame[1].item(),
                        "z": frame[2].item()
                    },
                    "rotation": {
                        "x": frame[6].item(),
                        "y": frame[7].item(),
                        "z": frame[8].item()
                    }
                } for frame in subject],
                "dimensions": {
                    "width": subject[0, 3].item(),
                    "height": subject[0, 4].item(),
                    "depth": subject[0, 5].item()
                },
            }],
            "cameraFrames": [
                {
                    "position": {
                        "x": camera[i, 0].item(),
                        "y": camera[i, 1].item(),
                        "z": camera[i, 2].item()
                    },
                    "angle": {
                        "x": camera[i, 3].item(),
                        "y": camera[i, 4].item(),
                        "z": camera[i, 5].item()
                    },
                    "focalLength": camera[i, 6].item(),
                }
                for i in range(camera.size(0))
            ],
            "instructions": [],
        }
        
        if helper_keyframes is not None:
            res['helper_keyframes'] = [
                {
                    "position": {
                        "x": helper_keyframes[i, 0].item(),
                        "y": helper_keyframes[i, 1].item(),
                        "z": helper_keyframes[i, 2].item()
                    },
                    "focalLength": helper_keyframes[i, 3].item(),
                    "angle": {
                        "x": helper_keyframes[i, 4].item(),
                        "y": helper_keyframes[i, 5].item(),
                        "z": helper_keyframes[i, 6].item()
                    }
                }
                for i in range(helper_keyframes.size(0))
            ]
        
        return res
    def save_simulation_format(self, data, output_dir):
        output_path = os.path.join(output_dir, f'simulation-out.json')
        simulations = []
        for item in data:
            print(item["simulation_instructions"], item["cinematography_prompts"])
            simulations += [
                self.generate_simulation_format(item['camera'], item['subject']),
                self.generate_simulation_format(item['rec'], item['subject']),
                self.generate_simulation_format(item['hybrid_gen'], item['subject']),
                self.generate_simulation_format(item['prompt_gen'], item['subject'])
            ]
            
        with open(output_path, 'w') as f:
            json.dump({"simulations": simulations}, f, indent=2)