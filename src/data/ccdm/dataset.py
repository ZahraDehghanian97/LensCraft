import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from torch.utils.data import Dataset
from data.collate_utils import collate_structured_conditioning, collate_trajectories

class CCDMDataset(Dataset):
    _normalization_parameters = None

    def __init__(
        self,
        data_path: str,
        normalize: bool = True,
        original_seq_len: int = 300,
        hfov_deg: float | None = None,
        aspect: float = 16 / 9,
        device: str | torch.device | None = None,
        annotations_path: str | None = None,
        sample_indices_path: str | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        CCDMDataset.get_normalization_parameters(self.data_path.parent)
        self.normalize = normalize
        self.original_seq_len = original_seq_len
        self.sample_indices_path = sample_indices_path
        self.annotations = None

        from data.ccdm.convertor import CCDMConvertor
        projection = CCDMConvertor(hfov_deg=hfov_deg, aspect=aspect)
        self.hfov_deg = projection.hfov_deg
        self.vfov_deg = projection.vfov_deg
        self.aspect = projection.aspect

        self.device = torch.device(device or "cpu")
        self.padding_masks = []

        self._load_camera_trajectory_data()
        if annotations_path is not None:
            self._load_annotations(annotations_path)
        self.texts = [" ".join(description) for description in self.text_descriptions]

    def _load_annotations(self, annotations_path):
        from annotator.gateway import validate_translation_schema
        rows = json.loads(Path(annotations_path).read_text())
        self.annotations = {}
        for row in rows:
            item_id = row["custom_id"]
            if item_id in self.annotations:
                raise ValueError(f"Duplicate CCDM annotation: {item_id}")
            self.annotations[item_id] = row
        for source_index, description in zip(self.source_indices, self.text_descriptions):
            item_id = f"ccdm-{source_index:06d}"
            row = self.annotations.get(item_id)
            if row is None:
                raise ValueError(f"Missing CCDM annotation: {item_id}")
            if row.get("source_prompt") != " ".join(description):
                raise ValueError(f"CCDM annotation caption mismatch: {item_id}")
            # Persisted benchmark annotations retain their original motion aliases.
            # Fresh Gateway translations are checked during preparation; loading
            # older cohorts validates their schema without revising their content.
            translation = {key: row[key] for key in ("cinematographyPrompts", "simulationInstructions")}
            shots = translation["cinematographyPrompts"]
            shot = shots[0] if isinstance(shots, list) and len(shots) == 1 else None
            movement = shot.get("movement") if isinstance(shot, dict) else None
            movement = movement.get("type") if isinstance(movement, dict) else None
            if isinstance(movement, str) and movement in {"nativePan", "nativeOrbit"}:
                # The categorical schema adds two explicit native labels.
                # Validate the remaining stored schema with the historical
                # validator without mutating the annotation or legacy enum.
                from copy import deepcopy

                translation = deepcopy(translation)
                translation["cinematographyPrompts"][0]["movement"]["type"] = None
            validate_translation_schema(translation)

    @staticmethod
    def get_normalization_parameters(data_path: str | Path = None) -> dict[str, torch.Tensor]:
        if CCDMDataset._normalization_parameters is not None:
            return CCDMDataset._normalization_parameters

        if data_path is None:
            data_path = os.environ.get('CCDM_DATA_DIR')
            if data_path is None:
                raise ValueError("CCDM_DATA_DIR environment variable must be set")

        data_path = Path(data_path)
        stats_file = data_path / "Mean_Std.npy"
        stats = np.load(stats_file, allow_pickle=True).item()

        CCDMDataset._normalization_parameters = {
            "mean": torch.tensor(stats["Mean"], dtype=torch.float32),
            "std":  torch.tensor(stats["Std"],  dtype=torch.float32),
        }
        return CCDMDataset._normalization_parameters

    @staticmethod
    def normalize_item(
        camera_trajectory: torch.Tensor,
        subject_trajectory: Optional[torch.Tensor] = None,
        subject_volume:    Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        CCDMDataset.get_normalization_parameters()
        mean = CCDMDataset._normalization_parameters["mean"].to(camera_trajectory.device)
        std  = CCDMDataset._normalization_parameters["std"].to(camera_trajectory.device)
        eps  = 1e-8

        if normalize:
            camera_trajectory = (camera_trajectory - mean) / (std + eps)
        else:
            camera_trajectory = camera_trajectory * std + mean

        return camera_trajectory, subject_trajectory, subject_volume

    def _load_camera_trajectory_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        raw_data = np.load(self.data_path, allow_pickle=True)[()]
        if len(raw_data["cam"]) != len(raw_data["info"]):
            raise ValueError("CCDM camera and caption counts differ")
        self.source_indices = list(range(len(raw_data["cam"])))
        if self.sample_indices_path is not None:
            indices = json.loads(Path(self.sample_indices_path).read_text())
            if (not isinstance(indices, list) or not indices
                    or any(type(i) is not int or not 0 <= i < len(raw_data["cam"])
                           for i in indices) or len(indices) != len(set(indices))):
                raise ValueError("CCDM sample indices must be unique valid integers")
            self.source_indices = indices
        self.camera_trajectories = [torch.tensor(raw_data["cam"][i], dtype=torch.float32)
                                    for i in self.source_indices]
        self.text_descriptions = [raw_data["info"][i] for i in self.source_indices]


        for index, camera_trajectory in enumerate(self.camera_trajectories):
            if (camera_trajectory.ndim != 2 or camera_trajectory.shape[1] != 5
                    or len(camera_trajectory) == 0 or not torch.isfinite(camera_trajectory).all()):
                raise ValueError(f"Invalid CCDM trajectory at source index {self.source_indices[index]}")
            current_len = camera_trajectory.shape[0]
            padding_mask = torch.zeros(current_len, dtype=torch.bool)

            if current_len < self.original_seq_len:
                num_padding = self.original_seq_len - current_len
                pad_values = camera_trajectory[-1:].repeat(num_padding, 1)
                camera_trajectory = torch.cat([camera_trajectory, pad_values], dim=0)
                padding_mask = torch.cat([padding_mask, torch.ones(num_padding, dtype=torch.bool)], dim=0)
            elif current_len > self.original_seq_len:
                camera_trajectory = camera_trajectory[:self.original_seq_len]
                padding_mask = padding_mask[:self.original_seq_len]

            self.padding_masks.append(padding_mask)
            self.camera_trajectories[index] = camera_trajectory

    def __len__(self) -> int:
        return len(self.camera_trajectories)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        camera_trajectory = self.camera_trajectories[index].clone()
        padding_mask = self.padding_masks[index]

        subject_trajectory = None
        subject_volume = None

        if self.normalize:
            camera_trajectory, _, _ = CCDMDataset.normalize_item(camera_trajectory, None, None, True)

        text = self.texts[index]
        result = {
            "camera_trajectory": camera_trajectory,
            "subject_trajectory": subject_trajectory,
            "subject_volume": subject_volume,
            "padding_mask": padding_mask,
            "text_prompts": text,
            "item_id": f"ccdm-{self.source_indices[index]:06d}",
        }
        from data.categorical_conditioning import metadata_from_prompt

        row = self.annotations[result["item_id"]] if self.annotations is not None else None
        prompt = row["cinematographyPrompts"][0] if row is not None else {}
        instruction = row["simulationInstructions"][0] if row is not None else {}
        result.update(metadata_from_prompt(prompt, instruction), raw_prompt=prompt, raw_instruction=instruction)
        return result

def collate_fn(batch):
    result = {
        **collate_trajectories(batch),
        "text_prompts": [item["text_prompts"] for item in batch],
        "item_ids": [item["item_id"] for item in batch],
    }
    result.update(collate_structured_conditioning(batch))
    result["raw_prompt"] = [item["raw_prompt"] for item in batch]
    result["raw_instruction"] = [item["raw_instruction"] for item in batch]
    return result
