from typing import Dict, List
from torch.utils.data import Dataset
import torch
import msgpack
from pathlib import Path

from .constants import (
    cinematography_struct,
    cinematography_struct_size,
    simulation_struct,
    simulation_struct_size,
)
from .utils import get_parameters
from .loader import load_simulation_file

class SimulationDataset(Dataset):
    def __init__(self, data_path: str, clip_embeddings: Dict, embedding_dim: int):
        self.clip_embeddings = clip_embeddings
        self.embedding_dim = embedding_dim
        self.data_path = Path(data_path)
        
        if not self.data_path.is_dir():
            raise ValueError(f"Expected directory at {data_path}")
            
        dict_path = self.data_path / "parameter_dictionary.msgpack"
        if not dict_path.exists():
            raise ValueError(f"parameter_dictionary.msgpack not found in {data_path}")
            
        with open(dict_path, 'rb') as f:
            self.parameter_dictionary = msgpack.unpackb(f.read(), raw=False)
            
        self.simulation_files = sorted(
            self.data_path.glob('simulation_*.msgpack')
        )
        
        if not self.simulation_files:
            raise ValueError(f"No simulation files found in {data_path}")

    def __len__(self) -> int:
        return len(self.simulation_files)

    def __getitem__(self, index: int) -> Dict:
        file_path = self.simulation_files[index]
        data = load_simulation_file(file_path, self.parameter_dictionary)
        
        if data is None:
            raise ValueError(f"Failed to load simulation file at index {index}")
            
        return self._process_single_simulation(data)

    def _process_single_simulation(self, simulation_data: Dict) -> Dict:
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

        simulation_instruction_tensor = self._create_instruction_tensor(
            simulation_instruction,
            simulation_struct_size
        )
        cinematography_prompt_tensor = self._create_instruction_tensor(
            cinematography_prompt,
            cinematography_struct_size
        )

        return {
            "camera_trajectory": torch.tensor(camera_trajectory, dtype=torch.float32),
            "subject_trajectory": torch.tensor(subject_trajectory, dtype=torch.float32),
            "simulation_instruction": simulation_instruction_tensor,
            "cinematography_prompt": cinematography_prompt_tensor,
            "simulation_instruction_parameters": simulation_instruction,
            "cinematography_prompt_parameters": cinematography_prompt
        }

    def _create_instruction_tensor(self, parameters: List, struct_size: int) -> torch.Tensor:
        instruction_tensor = torch.full((struct_size, self.embedding_dim), -1, dtype=torch.float)
        
        for param_idx, (_, _, _, embedding) in enumerate(parameters):
            if embedding is not None:
                instruction_tensor[param_idx] = embedding
                
        return instruction_tensor

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
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": torch.stack([item["subject_trajectory"] for item in batch]),
        "simulation_instruction": torch.stack([item["simulation_instruction"] for item in batch]).transpose(0, 1),
        "cinematography_prompt": torch.stack([item["cinematography_prompt"] for item in batch]).transpose(0, 1),
        "simulation_instruction_parameters": [
            item["simulation_instruction_parameters"] for item in batch
        ],
        "cinematography_prompt_parameters": [
            item["cinematography_prompt_parameters"] for item in batch
        ],
    }
