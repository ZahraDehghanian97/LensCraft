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
    resample_paired_euler_trajectories,
)
from .metadata import (
    SIMULATION_FRAME_COUNT,
    compute_dataset_fingerprint,
    get_fixed_point_scale,
    load_dataset_manifest,
    resolve_dataset_root,
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
        target_frame_count: int = SIMULATION_FRAME_COUNT,
    ):
        self.data_path = resolve_dataset_root(Path(data_path))
        self.embedding_dim = embedding_dim
        self.fill_none_with_mean = fill_none_with_mean
        self.clip_embeddings = clip_embeddings
        self.allowed_movement_types = allowed_movement_types or []
        self.normalize = normalize
        if target_frame_count < 2:
            raise ValueError("target_frame_count must be at least 2")
        self.target_frame_count = int(target_frame_count)

        self.embedding_means = load_clip_means() if self.fill_none_with_mean else None

        self.manifest = load_dataset_manifest(self.data_path)
        if self.manifest is not None:
            export_config = self.manifest.get("config", {})
            for count_name in ("subjectCount", "instructionCount"):
                count = export_config.get(count_name)
                if count is not None and count != 1:
                    raise ValueError(
                        f"Simulation datasets require {count_name}=1; "
                        f"manifest declares {count!r}"
                    )
        self.fixed_point_scale = get_fixed_point_scale(self.manifest)
        self.parameter_dictionary = load_parameter_dictionary(self.data_path)
        manifest_dataset_id = (
            self.manifest.get("datasetId") if self.manifest is not None else None
        )
        dictionary_dataset_id = self.parameter_dictionary.get("datasetId")
        schema_v2 = (
            (self.manifest or {}).get("schemaVersion", 0) >= 2
            or self.parameter_dictionary.get("schemaVersion", 0) >= 2
        )
        if schema_v2 and (
            not isinstance(manifest_dataset_id, str)
            or not manifest_dataset_id
            or not isinstance(dictionary_dataset_id, str)
            or not dictionary_dataset_id
        ):
            raise ValueError(
                "Schema-v2 simulation datasets require the same non-empty "
                "datasetId in both manifest and parameter dictionary"
            )
        if (
            manifest_dataset_id is not None
            and dictionary_dataset_id is not None
            and manifest_dataset_id != dictionary_dataset_id
        ):
            raise ValueError(
                "Dataset manifest and parameter dictionary have different "
                f"datasetId values: {manifest_dataset_id!r} != "
                f"{dictionary_dataset_id!r}"
            )
        self.simulation_files = find_simulation_files(self.data_path)
        dataset_id = dictionary_dataset_id or manifest_dataset_id
        if dataset_id is not None:
            expected_prefix = f"simulation_{dataset_id}_"
            mismatched_files = [
                path.name
                for path in self.simulation_files
                if not path.name.startswith(expected_prefix)
            ]
            if mismatched_files:
                raise ValueError(
                    "Simulation files do not belong to datasetId "
                    f"{dataset_id!r}: {mismatched_files[:3]}"
                )
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
                target_frame_count=self.target_frame_count,
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
        target_frame_count: int = SIMULATION_FRAME_COUNT,
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

        data_path = resolve_dataset_root(Path(data_path))
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

        normalization_fingerprint = (
            f"{dataset_fingerprint}:target_frames={target_frame_count}"
        )
        cache_key = f"{data_path.resolve()}:{normalization_fingerprint}"
        parameters = SimulationDataset._normalization_parameters_cache.get(cache_key)
        if parameters is None:
            parameters = load_or_calculate_normalization_parameters(
                data_path,
                parameter_dictionary=parameter_dictionary,
                simulation_files=simulation_files,
                fixed_point_scale=fixed_point_scale,
                dataset_fingerprint=normalization_fingerprint,
                target_frame_count=target_frame_count,
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
        original_frame_count = camera_trajectory.shape[0]
        if subject_trajectory.shape[0] != original_frame_count:
            raise ValueError(
                f"Camera/subject frame mismatch in {self.simulation_files[index]}: "
                f"{original_frame_count} camera frames versus "
                f"{subject_trajectory.shape[0]} subject frames"
            )
        camera_trajectory, subject_trajectory = (
            resample_paired_euler_trajectories(
                camera_trajectory,
                subject_trajectory,
                self.target_frame_count,
            )
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
            embedding_dim=self.embedding_dim,
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
            "padding_mask": torch.zeros(
                self.target_frame_count, dtype=torch.bool
            ),
            "original_frame_count": original_frame_count,
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
        "original_frame_count": torch.tensor(
            [item["original_frame_count"] for item in batch], dtype=torch.long
        ),
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
