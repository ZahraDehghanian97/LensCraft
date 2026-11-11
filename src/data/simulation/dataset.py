import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .caption import extract_text_prompt
from .loader import (
    extract_camera_trajectory,
    extract_camera_intrinsics,
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
from data.categorical_conditioning import metadata_from_prompt
from .observation import ModelObservationPolicy
from data.collate_utils import (
    collate_structured_conditioning,
    collate_trajectories,
    stack_optional,
)


class SimulationDataset(Dataset):
    _normalization_parameters = None
    _normalization_parameters_cache = {}
    _normalization_parameters_override = None

    def __init__(
        self,
        data_path: str,
        allowed_movement_types: List[str] = None,
        normalize: bool = True,
        target_frame_count: int = SIMULATION_FRAME_COUNT,
        normalization_frame_count: Optional[int] = None,
        reference_frame_count: Optional[int] = None,
    ):
        self.data_path = resolve_dataset_root(Path(data_path))
        self.allowed_movement_types = allowed_movement_types or []
        self.normalize = normalize
        if target_frame_count < 2:
            raise ValueError("target_frame_count must be at least 2")
        self.target_frame_count = int(target_frame_count)
        self.normalization_frame_count = (
            self.target_frame_count
            if normalization_frame_count is None
            else int(normalization_frame_count)
        )
        self.reference_frame_count = (
            None if reference_frame_count is None else int(reference_frame_count)
        )
        if self.normalization_frame_count < 2 or (
            self.reference_frame_count is not None and self.reference_frame_count < 2
        ):
            raise ValueError("Normalization and reference frame counts must be at least 2")

        self.manifest = load_dataset_manifest(self.data_path)
        self.model_observation_policy = ModelObservationPolicy.from_manifest(self.manifest)
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
                target_frame_count=self.normalization_frame_count,
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
        """Use this dataset's statistics, or an explicit shared reference export.

        Set SIMULATION_NORMALIZATION_DATA_PATH for training and evaluation to
        share an independent dataset's coordinate scales. The override applies
        to both dataset instances and the static ET/CCDM conversion helpers.
        """
        normalization_data_path = os.environ.get("SIMULATION_NORMALIZATION_DATA_PATH")
        override_key = (
            (str(Path(normalization_data_path).expanduser().absolute()), target_frame_count)
            if normalization_data_path else None
        )
        if (
            data_path is None
            and SimulationDataset._normalization_parameters is not None
            and SimulationDataset._normalization_parameters_override == override_key
        ):
            return SimulationDataset._normalization_parameters

        if normalization_data_path:
            data_path = Path(normalization_data_path).expanduser()
            # These arguments describe the samples being loaded, not the
            # reference export. Never decode the reference with their dictionary
            # or use their inventory/fingerprint to populate its cache.
            parameter_dictionary = None
            simulation_files = None
            fixed_point_scale = None
            dataset_fingerprint = None

        if data_path is None:
            data_path = os.environ.get("SIMULATION_DATA_PATH")
            if data_path is None:
                raise ValueError(
                    "SIMULATION_DATA_PATH environment variable must be set"
                )

        data_path = resolve_dataset_root(Path(data_path))
        if simulation_files is None:
            simulation_files = find_simulation_files(data_path)
        if normalization_data_path and not simulation_files:
            raise ValueError(f"Normalization reference has no simulation files: {data_path}")
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
        SimulationDataset._normalization_parameters_override = override_key
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
        reference = None
        if self.reference_frame_count is not None:
            reference_camera, reference_subject = resample_paired_euler_trajectories(
                camera_trajectory, subject_trajectory, self.reference_frame_count
            )
            reference_volume = subject_volume
            reference_intrinsics = extract_camera_intrinsics(
                data["cameraFrames"], self.reference_frame_count
            )
            reference_geometry = None
            if self.model_observation_policy is not None:
                reference_geometry = {
                    "simulation_normalized": False,
                    "camera_trajectory": reference_camera.clone(),
                    "subject_trajectory": reference_subject.clone(),
                    "subject_volume": reference_volume.clone(),
                    "camera_intrinsics": reference_intrinsics,
                    "padding_mask": torch.zeros(self.reference_frame_count, dtype=torch.bool),
                }
                reference_subject, reference_volume = self.model_observation_policy.subject_condition(
                    reference_subject, reference_volume,
                )
            if self.normalize:
                reference_camera, reference_subject, reference_volume = (
                    SimulationDataset._normalize_item_with_parameters(
                        reference_camera, reference_subject, reference_volume,
                        self.normalization_parameters,
                    )
                )
            reference = {
                "camera_intrinsics": reference_intrinsics,
                "camera_trajectory": reference_camera,
                "subject_trajectory": reference_subject,
                "subject_volume": reference_volume,
                "padding_mask": torch.zeros(self.reference_frame_count, dtype=torch.bool),
            }
            if reference_geometry is not None:
                reference["geometry_reference"] = reference_geometry
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
        visible_prompt, visible_instruction = prompt, instruction
        observed_motion_text = None
        geometry_reference = None
        if self.model_observation_policy is not None:
            # This truth is kept separately for physical geometry/inspection;
            # it is never passed through the model's subject conditioning.
            geometry_reference = {
                "simulation_normalized": False,
                "camera_trajectory": camera_trajectory.clone(),
                "subject_trajectory": subject_trajectory.clone(),
                "subject_volume": subject_volume.clone(),
                "camera_intrinsics": extract_camera_intrinsics(
                    data["cameraFrames"], self.target_frame_count
                ),
                "padding_mask": torch.zeros(self.target_frame_count, dtype=torch.bool),
            }
            subject_trajectory, subject_volume = self.model_observation_policy.subject_condition(
                subject_trajectory, subject_volume,
            )
            visible_prompt, visible_instruction = self.model_observation_policy.caption_condition(
                prompt, instruction,
            )
            observed_motion_text = self.model_observation_policy.observed_motion_text(prompt)

        conditioning = metadata_from_prompt(visible_prompt, visible_instruction)

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

        item = {
            "camera_intrinsics": extract_camera_intrinsics(
                data["cameraFrames"], self.target_frame_count
            ),
            "simulation_normalized": self.normalize,
            "camera_trajectory": camera_trajectory,
            "subject_trajectory": subject_trajectory,
            "subject_volume": subject_volume,
            "padding_mask": torch.zeros(
                self.target_frame_count, dtype=torch.bool
            ),
            "original_frame_count": original_frame_count,
            **conditioning,
            "raw_prompt": prompt,
            "raw_instruction": instruction,
            "text_prompt": (
                observed_motion_text if observed_motion_text is not None
                else extract_text_prompt(visible_prompt, movement_type)
            ),
        }
        if reference is not None:
            item["simulation_reference"] = reference
        if geometry_reference is not None:
            item["geometry_reference"] = geometry_reference
            item["subject_orientation_available"] = torch.tensor(False, dtype=torch.bool)
            item["subject_dimensions_available"] = torch.tensor(False, dtype=torch.bool)
            item["model_observation"] = self.model_observation_policy.provenance()
        return item


def collate_fn(batch):
    conditioning = collate_structured_conditioning(batch)

    def collate_intrinsics(items):
        values = [item.get("camera_intrinsics") for item in items]
        if all(value is None for value in values):
            return None
        # Missing lens metadata is excluded from projection metrics per sample.
        template = next(value for value in values if value is not None)
        return torch.stack([
            value if value is not None else torch.full_like(template, float("nan"))
            for value in values
        ])

    def collate_geometry_references(references):
        result = {
            key: stack_optional(references, key)
            for key in ("camera_trajectory", "subject_trajectory", "subject_volume", "padding_mask")
        }
        result["camera_intrinsics"] = collate_intrinsics(references)
        result["simulation_normalized"] = False
        return result

    result = {
        "camera_intrinsics": collate_intrinsics(batch),
        "simulation_normalized": batch[0].get("simulation_normalized", True),
        **collate_trajectories(batch),
        "original_frame_count": torch.tensor(
            [item["original_frame_count"] for item in batch], dtype=torch.long
        ),
        **conditioning,
        "raw_prompt": [item["raw_prompt"] for item in batch],
        "raw_instruction": [item["raw_instruction"] for item in batch],
        "text_prompts": [item["text_prompt"] for item in batch],
    }
    if "simulation_reference" in batch[0]:
        references = [item["simulation_reference"] for item in batch]
        result["simulation_reference"] = {
            key: stack_optional(references, key)
            for key in ("camera_trajectory", "subject_trajectory", "subject_volume", "padding_mask")
        }
        result["simulation_reference"]["camera_intrinsics"] = collate_intrinsics(references)
        if "geometry_reference" in references[0]:
            result["simulation_reference"]["geometry_reference"] = collate_geometry_references(
                [reference["geometry_reference"] for reference in references]
            )
    if any("geometry_reference" in item for item in batch):
        if not all("geometry_reference" in item for item in batch):
            raise ValueError("A simulation batch cannot mix model observation policies")
        references = [item["geometry_reference"] for item in batch]
        result["geometry_reference"] = collate_geometry_references(references)
        for key in ("subject_orientation_available", "subject_dimensions_available"):
            result[key] = torch.stack([item[key] for item in batch])
        result["model_observation"] = [item["model_observation"] for item in batch]
    return result
