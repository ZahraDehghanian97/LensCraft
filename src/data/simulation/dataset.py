from typing import Dict, List
from torch.utils.data import Dataset
import torch
import msgpack
import json
from pathlib import Path

from .constants import (
    cinematography_struct,
    cinematography_struct_size,
    simulation_struct,
    simulation_struct_size,
)
from .utils import get_parameters

def unround_floats(obj, factor=1000.0):
    if isinstance(obj, int):
        return obj / factor
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, list):
        return [unround_floats(x, factor) for x in obj]
    elif isinstance(obj, dict):
        return {k: unround_floats(v, factor) for k, v in obj.items()}
    else:
        return obj

class SimulationDataset(Dataset):
    def __init__(self, data_path: str, clip_embeddings: Dict):
        self.clip_embeddings = clip_embeddings
        self.embedding_dim = 512
        self.raw_data_list = []
        
        path = Path(data_path)
        
        if path.is_file():
            self.raw_data_list = self._load_file(path)
        elif path.is_dir():
            for file_path in path.glob('*'):
                if file_path.suffix.lower() in ['.json', '.mpack']:
                    file_data = self._load_file(file_path)
                    if isinstance(file_data, list):
                        self.raw_data_list.extend(file_data)
                    else:
                        self.raw_data_list.append(file_data)

        if not self.raw_data_list:
            raise ValueError(f"No valid data files found in {data_path}")

    def _load_file(self, file_path: Path) -> List[Dict]:
        with open(file_path, 'rb') as file:
            binary_data = file.read()
        
        if file_path.suffix.lower() == '.json':
            return json.loads(binary_data)
        else:
            return msgpack.unpackb(binary_data, raw=False)

    def __len__(self) -> int:
        return len(self.raw_data_list)

    def __getitem__(self, index: int) -> Dict:
        unrounded = unround_floats(self.raw_data_list[index])
        return self._process_single_simulation(unrounded, index)

    def _process_single_simulation(self, simulation_data: Dict, index: int) -> Dict:
        camera_trajectory = self._extract_camera_trajectory(simulation_data["cameraFrames"])
        subject_trajectory = self._extract_subject_trajectory(simulation_data["subjectsInfo"])
        instruction = simulation_data["simulationInstructions"][0]
        prompt = simulation_data["cinematographyPrompts"][0]

        simulation_instruction = get_parameters(
            data=instruction,
            struct=simulation_struct,
            clip_embeddings=self.clip_embeddings
        )
        cinematography_prompt = get_parameters(
            data=prompt,
            struct=cinematography_struct,
            clip_embeddings=self.clip_embeddings
        )

        return {
            "camera_trajectory": torch.tensor(camera_trajectory, dtype=torch.float32),
            "subject_trajectory": torch.tensor(subject_trajectory, dtype=torch.float32),
            "simulation_instruction_parameters": simulation_instruction,
            "cinematography_prompt_parameters": cinematography_prompt
        }

    def _extract_camera_trajectory(self, camera_frames: List[Dict]) -> List[List[float]]:
        return [
            [
                frame["position"]["x"],
                frame["position"]["y"],
                frame["position"]["z"],
                frame["rotation"]["x"],
                frame["rotation"]["y"],
                frame["rotation"]["z"],
                frame["focalLength"]
            ]
            for frame in camera_frames
        ]

    def _extract_subject_trajectory(self, subjects_info: List[Dict]) -> List[List[float]]:
        subject_info = subjects_info[0]
        subject = subject_info["subject"]

        return [
            [
                frame["position"]["x"], 
                frame["position"]["y"], 
                frame["position"]["z"],
                subject["dimensions"]["width"],
                subject["dimensions"]["height"],
                subject["dimensions"]["depth"],
                frame["rotation"]["x"],
                frame["rotation"]["y"],
                frame["rotation"]["z"]
            ]
            for frame in subject_info["frames"]
        ]

def collate_fn(batch):
    batch_size = len(batch)
    simulation_instruction_tensor = torch.full(
        (simulation_struct_size, batch_size, 512),
        -1,
        dtype=torch.float
    )
    cinematography_prompt_tensor = torch.full(
        (cinematography_struct_size, batch_size, 512),
        -1,
        dtype=torch.float
    )
    
    for batch_idx, item in enumerate(batch):
        for param_idx, (_, _, _, embedding) in enumerate(item["simulation_instruction_parameters"]):
            if embedding is not None:
                simulation_instruction_tensor[param_idx, batch_idx] = embedding
        
        for param_idx, (_, _, _, embedding) in enumerate(item["cinematography_prompt_parameters"]):
            if embedding is not None:
                cinematography_prompt_tensor[param_idx, batch_idx] = embedding
    
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": torch.stack([item["subject_trajectory"] for item in batch]),
        "simulation_instruction": simulation_instruction_tensor,
        "cinematography_prompt": cinematography_prompt_tensor,
        "simulation_instruction_parameters": [
            item["simulation_instruction_parameters"] for item in batch
        ],
        "cinematography_prompt_parameters": [
            item["cinematography_prompt_parameters"] for item in batch
        ],
    }
