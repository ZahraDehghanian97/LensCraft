import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .caption import extract_text_prompt
from .loader import (
    extract_camera_trajectory,
    extract_subject_components,
    filter_files_by_movement_types,
    find_simulation_files,
    generate_movement_types_file,
    load_or_calculate_normalization_parameters,
    load_parameter_dictionary,
    parse_simulation_file_to_dict,
)
from .metadata import (
    SIMULATION_FRAME_COUNT,
    compute_dataset_fingerprint,
    get_fixed_point_scale,
    load_dataset_manifest,
)
from .utils import fix_prompts_and_instructions, load_clip_means
from data.collate_utils import stack_optional


class SimulationDataset(Dataset):
    _normalization_parameters = None
    _normalization_parameters_cache = {}

    def __init__(
        self,
        data_path: str,
        embedding_dim: int,
        fill_none_with_mean: bool,
        clip_embeddings: Dict,
        allowed_movement_types: List[str] = None,
        normalize: bool = True,
    ):
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        self.fill_none_with_mean = fill_none_with_mean
        self.clip_embeddings = clip_embeddings
        self.allowed_movement_types = allowed_movement_types or []
        self.normalize = normalize

        self.embedding_means = load_clip_means() if self.fill_none_with_mean else None

        self.manifest = load_dataset_manifest(self.data_path)
        self.fixed_point_scale = get_fixed_point_scale(self.manifest)
        self.parameter_dictionary = load_parameter_dictionary(self.data_path)
        self.simulation_files = find_simulation_files(self.data_path)
        self.dataset_fingerprint = compute_dataset_fingerprint(
            self.data_path, self.manifest, self.simulation_files
        )

        normalization_files = list(self.simulation_files)
        if self.allowed_movement_types:
            generate_movement_types_file(
                self.data_path,
                self.simulation_files,
                self.parameter_dictionary,
                self.fixed_point_scale,
                self.dataset_fingerprint,
            )
            self.simulation_files = filter_files_by_movement_types(
                self.simulation_files,
                self.allowed_movement_types,
                self.data_path,
            )
            print(
                f"Filtered to {len(self.simulation_files)} files with "
                f"movement types: {self.allowed_movement_types}"
            )

        if self.normalize:
            self.normalization_parameters = self.get_normalization_parameters(
                self.data_path,
                parameter_dictionary=self.parameter_dictionary,
                simulation_files=normalization_files,
                fixed_point_scale=self.fixed_point_scale,
                dataset_fingerprint=self.dataset_fingerprint,
            )
            print("Normalization enabled. Using normalization parameters.")
        else:
            self.normalization_parameters = None

    @staticmethod
    def get_normalization_parameters(
        data_path: str = None,
        *,
        parameter_dictionary: Optional[Dict] = None,
        simulation_files: Optional[Sequence[Path]] = None,
        fixed_point_scale: Optional[float] = None,
        dataset_fingerprint: Optional[str] = None,
    ) -> Dict:
        if (
            data_path is None
            and SimulationDataset._normalization_parameters is not None
        ):
            return SimulationDataset._normalization_parameters

        if data_path is None:
            data_path = os.environ.get("SIMULATION_DATA_PATH")
            if data_path is None:
                raise ValueError(
                    "SIMULATION_DATA_PATH environment variable must be set"
                )

        data_path = Path(data_path)
        if simulation_files is None:
            simulation_files = find_simulation_files(data_path)
        if fixed_point_scale is None or dataset_fingerprint is None:
            manifest = load_dataset_manifest(data_path)
            if fixed_point_scale is None:
                fixed_point_scale = get_fixed_point_scale(manifest)
            if dataset_fingerprint is None:
                dataset_fingerprint = compute_dataset_fingerprint(
                    data_path, manifest, simulation_files
                )

        cache_key = f"{data_path.resolve()}:{dataset_fingerprint}"
        parameters = SimulationDataset._normalization_parameters_cache.get(cache_key)
        if parameters is None:
            parameters = load_or_calculate_normalization_parameters(
                data_path,
                parameter_dictionary=parameter_dictionary,
                simulation_files=simulation_files,
                fixed_point_scale=fixed_point_scale,
                dataset_fingerprint=dataset_fingerprint,
            )
            SimulationDataset._normalization_parameters_cache[cache_key] = parameters

        # Existing inference helpers use this process-wide "current" dataset.
        SimulationDataset._normalization_parameters = parameters
        return parameters

    def __len__(self) -> int:
        return len(self.simulation_files)

    @staticmethod
    def _normalize_tensor(
        tensor: torch.Tensor,
        param_key: str,
        position_indices: Optional[List[int]] = None,
        normalization_parameters: Optional[Dict] = None,
    ) -> torch.Tensor:
        parameters = (
            normalization_parameters
            if normalization_parameters is not None
            else SimulationDataset._normalization_parameters
        )
        mean = parameters[param_key]["mean"].to(tensor.device)
        std = parameters[param_key]["std"].to(tensor.device)

        if position_indices is None:
            return (tensor - mean) / std

        result = tensor.clone()
        result[..., position_indices] = (tensor[..., position_indices] - mean) / std
        return result

    @staticmethod
    def _denormalize_tensor(
        tensor: torch.Tensor,
        param_key: str,
        position_indices: Optional[List[int]] = None,
        normalization_parameters: Optional[Dict] = None,
    ) -> torch.Tensor:
        parameters = (
            normalization_parameters
            if normalization_parameters is not None
            else SimulationDataset._normalization_parameters
        )
        mean = parameters[param_key]["mean"].to(tensor.device)
        std = parameters[param_key]["std"].to(tensor.device)

        if position_indices is None:
            return tensor * std + mean

        result = tensor.clone()
        result[..., position_indices] = tensor[..., position_indices] * std + mean
        return result

    @staticmethod
    def normalize_item(
        camera_trajectory,
        subject_trajectory,
        subject_volume,
        normalize: bool = True,
    ):
        parameters = SimulationDataset.get_normalization_parameters()
        return SimulationDataset._normalize_item_with_parameters(
            camera_trajectory,
            subject_trajectory,
            subject_volume,
            parameters,
            normalize,
        )

    @staticmethod
    def _normalize_item_with_parameters(
        camera_trajectory,
        subject_trajectory,
        subject_volume,
        normalization_parameters,
        normalize: bool = True,
    ):
        transform = (
            SimulationDataset._normalize_tensor
            if normalize
            else SimulationDataset._denormalize_tensor
        )
        camera_trajectory = transform(
            camera_trajectory,
            "camera_position",
            position_indices=[0, 1, 2],
            normalization_parameters=normalization_parameters,
        )

        if subject_trajectory is not None:
            subject_trajectory = transform(
                subject_trajectory,
                "subject_position",
                position_indices=[0, 1, 2],
                normalization_parameters=normalization_parameters,
            )
        if subject_volume is not None:
            subject_volume = transform(
                subject_volume,
                "subject_dimensions",
                normalization_parameters=normalization_parameters,
            )

        return camera_trajectory, subject_trajectory, subject_volume

    def __getitem__(self, index: int) -> Dict:
        data = parse_simulation_file_to_dict(
            self.simulation_files[index],
            self.parameter_dictionary,
            self.fixed_point_scale,
        )

        camera_trajectory = extract_camera_trajectory(data["cameraFrames"])
        subject_trajectory, subject_volume = extract_subject_components(
            data["subjectsInfo"]
        )
        movement_type = data["subjectsInfo"][0]["movementType"]
        instruction = data["simulationInstructions"][0]
        prompt = data["cinematographyPrompts"][0]

        (
            simulation_instruction_tensor,
            cinematography_prompt_tensor,
            prompt_none_mask,
            simulation_instruction_parameters,
            cinematography_prompt_parameters,
        ) = fix_prompts_and_instructions(
            instruction,
            prompt,
            self.clip_embeddings,
            self.fill_none_with_mean,
            self.embedding_means,
        )

        if self.normalize:
            (
                camera_trajectory,
                subject_trajectory,
                subject_volume,
            ) = SimulationDataset._normalize_item_with_parameters(
                camera_trajectory,
                subject_trajectory,
                subject_volume,
                self.normalization_parameters,
            )

        return {
            "camera_trajectory": camera_trajectory,
            "subject_trajectory": subject_trajectory,
            "subject_volume": subject_volume,
            "padding_mask": torch.zeros(SIMULATION_FRAME_COUNT, dtype=torch.bool),
            "simulation_instruction": simulation_instruction_tensor,
            "cinematography_prompt": cinematography_prompt_tensor,
            "simulation_instruction_parameters": simulation_instruction_parameters,
            "cinematography_prompt_parameters": cinematography_prompt_parameters,
            "raw_prompt": prompt,
            "raw_instruction": instruction,
            "text_prompt": extract_text_prompt(prompt, movement_type),
            "prompt_none_mask": prompt_none_mask,
        }


def collate_fn(batch):
    subject_volume = stack_optional(batch, "subject_volume")
    subject_trajectory = stack_optional(batch, "subject_trajectory")

    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": subject_trajectory,
        "subject_volume": subject_volume,
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
        "simulation_instruction": torch.stack(
            [item["simulation_instruction"] for item in batch]
        ).transpose(0, 1),
        "cinematography_prompt": torch.stack(
            [item["cinematography_prompt"] for item in batch]
        ).transpose(0, 1),
        "simulation_instruction_parameters": [
            item["simulation_instruction_parameters"] for item in batch
        ],
        "cinematography_prompt_parameters": [
            item["cinematography_prompt_parameters"] for item in batch
        ],
        "raw_prompt": [item["raw_prompt"] for item in batch],
        "raw_instruction": [item["raw_instruction"] for item in batch],
        "text_prompts": [item["text_prompt"] for item in batch],
        "prompt_none_mask": torch.stack([item["prompt_none_mask"] for item in batch]),
    }
