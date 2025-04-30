from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import torch
import msgpack
from pathlib import Path
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
    create_prompt_none_mask_matrix,
)


class SimulationDataset(Dataset):
    def __init__(self, data_path: str, embedding_dim: int, fill_none_with_mean: bool, clip_embeddings: Dict):
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        self.fill_none_with_mean = fill_none_with_mean
        self.clip_embeddings = clip_embeddings

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
        
        if data is None:
            raise ValueError(f"Failed to load simulation file at index {index}")
            
        return self._convert_simulation_to_model_inputs(data)

    def _convert_simulation_to_model_inputs(self, simulation_data: Dict) -> Dict:
        camera_trajectory = self._extract_camera_trajectory(simulation_data["cameraFrames"])
        subject_loc_rot, subject_vol = self._extract_subject_components(simulation_data["subjectsInfo"])
        instruction = simulation_data["simulationInstructions"][0]
        prompt = simulation_data["cinematographyPrompts"][0]

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
        
        prompt_none_mask = create_prompt_none_mask_matrix(
            cinematography_prompt_parameters=[cinematography_prompt],
            simulation_instruction_parameters=[simulation_instruction],
            n_clip_embs=n_clip_embs
        )

        return {
            "camera_trajectory": torch.tensor(camera_trajectory, dtype=torch.float32),
            "subject_trajectory": torch.tensor(subject_loc_rot, dtype=torch.float32),
            "subject_volume": torch.tensor(subject_vol, dtype=torch.float32),
            "simulation_instruction": simulation_instruction_tensor,
            "cinematography_prompt": cinematography_prompt_tensor,
            "simulation_instruction_parameters": simulation_instruction,
            "cinematography_prompt_parameters": cinematography_prompt,
            "text_prompt": text_prompt,
            "prompt_none_mask": prompt_none_mask
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
                # frame["focalLength"]
            ]
            for frame in camera_frames
        ]

    def _extract_subject_components(self, subjects_info: List[Dict]) -> Tuple[List[List[float]], List[List[float]]]:
        subject_info = subjects_info[0]
        subject = subject_info["subject"]
        
        loc_rot = [
            [
                frame["position"]["x"],
                frame["position"]["y"],
                frame["position"]["z"],
                frame["rotation"]["x"],
                frame["rotation"]["y"],
                frame["rotation"]["z"]
            ]
            for frame in subject_info["frames"]
        ]
        
        vol = [
            [
                subject["dimensions"]["width"],
                subject["dimensions"]["height"],
                subject["dimensions"]["depth"]
            ]
        ]
        
        return loc_rot, vol

def collate_fn(batch):
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": torch.stack([item["subject_trajectory"] for item in batch]),
        "subject_volume": torch.stack([item["subject_volume"] for item in batch]),
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
