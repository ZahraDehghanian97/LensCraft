from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import torch
import msgpack
from pathlib import Path

from data.convertor.convertor import convert_to_target
from .loader import parse_simulation_file_to_dict

from .constants import (
    cinematography_struct,
    cinematography_struct_size,
    simulation_struct,
    simulation_struct_size,
)

from .utils import (
    extract_cinematography_parameters,
    convert_parameters_to_embedding_tensor,
    load_clip_means,
    extract_text_prompt,
    create_prompt_none_mask,
)


class SimulationDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 embedding_dim: int, 
                 fill_none_with_mean: bool, 
                 clip_embeddings: Dict, 
                 target = None):
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        self.fill_none_with_mean = fill_none_with_mean
        self.clip_embeddings = clip_embeddings
        self.target = target

        if self.fill_none_with_mean:
            self.embedding_means = load_clip_means()
        else:
            self.embedding_means = None
        
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
        # return len(self.simulation_files)
        return 100000

    def __getitem__(self, index: int) -> Dict:
        file_path = self.simulation_files[index]
        data = parse_simulation_file_to_dict(file_path, self.parameter_dictionary)
        
        camera_trajectory = self._extract_camera_trajectory(data["cameraFrames"])
        subject_trajectory, subject_volume = self._extract_subject_components(data["subjectsInfo"])
        instruction = data["simulationInstructions"][0]
        prompt = data["cinematographyPrompts"][0]

        simulation_instruction = extract_cinematography_parameters(
            data=instruction,
            struct=simulation_struct,
            clip_embeddings=self.clip_embeddings,
            fill_none_with_mean=self.fill_none_with_mean,
            embedding_means=self.embedding_means,
        )

        cinematography_prompt = extract_cinematography_parameters(
            data=prompt,
            struct=cinematography_struct,
            clip_embeddings=self.clip_embeddings,
            fill_none_with_mean=self.fill_none_with_mean,
            embedding_means=self.embedding_means,
        )

        simulation_instruction_tensor = convert_parameters_to_embedding_tensor(
            simulation_instruction,
            simulation_struct_size
        )
        
        cinematography_prompt_tensor = convert_parameters_to_embedding_tensor(
            cinematography_prompt,
            cinematography_struct_size
        )
        
        text_prompt = extract_text_prompt(prompt)

        n_clip_embs = len(cinematography_prompt) + len(simulation_instruction)
        
        prompt_none_mask = create_prompt_none_mask(
            cinematography_prompt_parameters=cinematography_prompt,
            simulation_instruction_parameters=simulation_instruction,
            n_clip_embs=n_clip_embs
        )
        
        padding_mask = None
        
        if "type" in self.target and self.target["type"] != "simulation":
            camera_trajectory, subject_trajectory, subject_volume, padding_mask = convert_to_target(
                "simulation",
                self.target["type"],
                camera_trajectory,
                subject_trajectory,
                subject_volume,
                padding_mask,
                self.target.get("seq_length", 30)
            )
        else:
            padding_mask = torch.zeros(30, dtype=torch.bool)

        return {
            "camera_trajectory": camera_trajectory,
            "subject_trajectory": subject_trajectory,
            "subject_volume": subject_volume,
            "padding_mask": padding_mask,
            "simulation_instruction": simulation_instruction_tensor,
            "cinematography_prompt": cinematography_prompt_tensor,
            "simulation_instruction_parameters": simulation_instruction,
            "cinematography_prompt_parameters": cinematography_prompt,
            "text_prompt": text_prompt,
            "prompt_none_mask": prompt_none_mask
        }


    def _extract_camera_trajectory(self, camera_frames: List[Dict]) -> List[List[float]]:
        return torch.tensor([
            [
                frame["position"]["x"],
                frame["position"]["y"],
                frame["position"]["z"],
                frame["rotation"]["x"],
                frame["rotation"]["y"],
                frame["rotation"]["z"],
                # frame["focalLength"]
            ]
            for frame in camera_frames
        ], dtype=torch.float32)

    def _extract_subject_components(self, subjects_info: List[Dict]) -> Tuple[List[List[float]], List[List[float]]]:
        subject_info = subjects_info[0]
        subject = subject_info["subject"]
        
        loc_rot = torch.tensor([
            [
                frame["position"]["x"],
                frame["position"]["y"],
                frame["position"]["z"],
                frame["rotation"]["x"],
                frame["rotation"]["y"],
                frame["rotation"]["z"]
            ]
            for frame in subject_info["frames"]
        ], dtype=torch.float32)
        
        vol = torch.tensor([
            [
                subject["dimensions"]["width"],
                subject["dimensions"]["height"],
                subject["dimensions"]["depth"]
            ]
        ], dtype=torch.float32)
        
        return loc_rot, vol

def collate_fn(batch):
    if len(batch) > 0 and batch[0]['subject_volume'] is None:
        subject_volume = None
    else:
        subject_volume = torch.stack([item["subject_volume"] for item in batch])
        
    if len(batch) > 0 and batch[0]['subject_trajectory'] is None:
        subject_trajectory = None
    else:
        subject_trajectory = torch.stack([item["subject_trajectory"] for item in batch])
    
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": subject_trajectory,
        "subject_volume": subject_volume,
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
        "simulation_instruction": torch.stack([item["simulation_instruction"] for item in batch]).transpose(0, 1),
        "cinematography_prompt": torch.stack([item["cinematography_prompt"] for item in batch]).transpose(0, 1),
        "simulation_instruction_parameters": [
            item["simulation_instruction_parameters"] for item in batch
        ],
        "cinematography_prompt_parameters": [
            item["cinematography_prompt_parameters"] for item in batch
        ],
        "text_prompts": [item["text_prompt"] for item in batch],
        "prompt_none_mask": torch.stack([item["prompt_none_mask"] for item in batch])
    }
