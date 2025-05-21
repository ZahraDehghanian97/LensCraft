import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset
import torch
from pathlib import Path

from .loader import (
    parse_simulation_file_to_dict,
    validate_dataset_directory,
    load_parameter_dictionary,
    find_simulation_files,
    generate_movement_types_file,
    filter_files_by_movement_types,
    load_or_calculate_normalization_parameters,
    extract_camera_trajectory,
    extract_subject_components
)

from .utils import fix_prompts_and_instructions, load_clip_means, extract_text_prompt


class SimulationDataset(Dataset):
    _normalization_parameters = None
    
    def __init__(self, 
                 data_path: str, 
                 embedding_dim: int, 
                 fill_none_with_mean: bool, 
                 clip_embeddings: Dict,
                 allowed_movement_types: List[str] = None,
                 normalize: bool = True):
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        self.fill_none_with_mean = fill_none_with_mean
        self.clip_embeddings = clip_embeddings
        self.allowed_movement_types = allowed_movement_types or []
        self.normalize = normalize

        if self.fill_none_with_mean:
            self.embedding_means = load_clip_means()
        else:
            self.embedding_means = None
        
        validate_dataset_directory(self.data_path)
        self.parameter_dictionary = load_parameter_dictionary(self.data_path)
        self.simulation_files = find_simulation_files(self.data_path)
        
        generate_movement_types_file(self.data_path, self.simulation_files, self.parameter_dictionary)
        
        if self.allowed_movement_types:
            self.simulation_files = filter_files_by_movement_types(
                self.simulation_files,
                self.allowed_movement_types,
                self.data_path
            )
            print(f"Filtered to {len(self.simulation_files)} files with movement types: {self.allowed_movement_types}")
        
        self.get_normalization_parameters(self.data_path)
        print("Normalization enabled. Using normalization parameters.")
    
    @staticmethod
    def get_normalization_parameters(data_path: str=os.environ.get('SIMULATION_DATA_PATH', '/media/disk1/arash/abolghasemi/simulation-data-3-mini')) -> Dict:
        if SimulationDataset._normalization_parameters is not None:
            return SimulationDataset._normalization_parameters
            
        SimulationDataset._normalization_parameters = load_or_calculate_normalization_parameters(data_path)
        return SimulationDataset._normalization_parameters

    def __len__(self) -> int:
        # return len(self.simulation_files)
        return min(100000, len(self.simulation_files))
    
    @staticmethod
    def _normalize_tensor(tensor: torch.Tensor, param_key: str, position_indices: Optional[List[int]] = None) -> torch.Tensor:
        mean =  SimulationDataset._normalization_parameters[param_key]['mean'].to(tensor.device)
        std =  SimulationDataset._normalization_parameters[param_key]['std'].to(tensor.device)
        
        if position_indices is None:
            return (tensor - mean) / std
        else:
            result = tensor.clone()
            result[..., position_indices] = (tensor[..., position_indices] - mean) / std
            return result

    @staticmethod
    def _denormalize_tensor(tensor: torch.Tensor, param_key: str, position_indices: Optional[List[int]] = None) -> torch.Tensor:        
        mean = SimulationDataset._normalization_parameters[param_key]['mean'].to(tensor.device)
        std = SimulationDataset._normalization_parameters[param_key]['std'].to(tensor.device)
        
        if position_indices is None:
            return tensor * std + mean
        else:
            result = tensor.clone()
            result[..., position_indices] = tensor[..., position_indices] * std + mean
            return result
    
    @staticmethod
    def normalize_item(camera_trajectory, subject_trajectory, subject_volume, normalize:bool= True):
        SimulationDataset.get_normalization_parameters()
        func = SimulationDataset._normalize_tensor if normalize else SimulationDataset._denormalize_tensor
        camera_trajectory = func(
            camera_trajectory,
            'camera_position',
            position_indices=[0, 1, 2]
        )
        
        if subject_trajectory is not None:
            subject_trajectory = func(
                subject_trajectory,
                'subject_position',
                position_indices=[0, 1, 2]
            )
        
        if subject_volume is not None:
            subject_volume = func(
                subject_volume,
                'subject_dimensions'
            )
        
        return camera_trajectory, subject_trajectory, subject_volume

    def __getitem__(self, index: int) -> Dict:
        file_path = self.simulation_files[index]
        data = parse_simulation_file_to_dict(file_path, self.parameter_dictionary)
        
        camera_trajectory = extract_camera_trajectory(data["cameraFrames"])
        subject_trajectory, subject_volume = extract_subject_components(data["subjectsInfo"])
        instruction = data["simulationInstructions"][0]
        prompt = data["cinematographyPrompts"][0]
        
        simulation_instruction_tensor, cinematography_prompt_tensor, prompt_none_mask, simulation_instruction_parameters, cinematography_prompt_parameters = \
            fix_prompts_and_instructions(instruction, prompt, self.clip_embeddings, self.fill_none_with_mean, self.embedding_means)

        if self.normalize:
            camera_trajectory, subject_trajectory, subject_volume = \
                SimulationDataset.normalize_item(camera_trajectory, subject_trajectory, subject_volume)

        
        padding_mask = torch.zeros(30, dtype=torch.bool)
        
        return {
            "camera_trajectory": camera_trajectory,
            "subject_trajectory": subject_trajectory,
            "subject_volume": subject_volume,
            "padding_mask": padding_mask,
            "simulation_instruction": simulation_instruction_tensor,
            "cinematography_prompt": cinematography_prompt_tensor,
            "simulation_instruction_parameters": simulation_instruction_parameters,
            "cinematography_prompt_parameters": cinematography_prompt_parameters,
            "text_prompt": extract_text_prompt(prompt),
            "prompt_none_mask": prompt_none_mask,
        }


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
        "prompt_none_mask": torch.stack([item["prompt_none_mask"] for item in batch]),
    }
