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
        clip_model_name: str = "openai/clip-vit-base-patch32",
        embedding_dim: int = 512,
        normalize: bool = True,
        original_seq_len: int = 300,
        hfov_deg: float | None = None,
        aspect: float = 16 / 9,
        device: str | torch.device | None = None,
        annotations_path: str | None = None,
        sample_indices_path: str | None = None,
        fill_none_with_mean: bool = True,
    ) -> None:
        from models.clip_embeddings import CLIPEmbedder

        self.data_path = Path(data_path)
        CCDMDataset.get_normalization_parameters(self.data_path.parent)
        self.embedding_dim = embedding_dim
        self.clip_model_name = clip_model_name
        self.normalize = normalize
        self.original_seq_len = original_seq_len
        self.sample_indices_path = sample_indices_path
        self.fill_none_with_mean = fill_none_with_mean
        self.annotations = None

        from data.ccdm.convertor import CCDMConvertor
        projection = CCDMConvertor(hfov_deg=hfov_deg, aspect=aspect)
        self.hfov_deg = projection.hfov_deg
        self.vfov_deg = projection.vfov_deg
        self.aspect = projection.aspect

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.clip_embedder = CLIPEmbedder(model_name=self.clip_model_name, device=self.device)
        self.padding_masks = []

        self._load_camera_trajectory_data()
        if annotations_path is not None:
            self._load_annotations(annotations_path)
        self._precompute_text_embeddings()

    def _load_annotations(self, annotations_path):
        from annotator.gateway import validate_translation_schema
        from data.simulation.init_embeddings import initialize_all_clip_embeddings
        from data.simulation.utils import load_clip_means

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
            validate_translation_schema({key: row[key] for key in
                                         ("cinematographyPrompts", "simulationInstructions")})
        cache_dir = Path(os.environ.get("CLIP_EMBEDDINGS_CACHE_DIR", "."))
        self.clip_embeddings = initialize_all_clip_embeddings(
            clip_model_name=self.clip_model_name,
            cache_file=str(cache_dir / f"clip_embeddings_{self.clip_model_name}_cache.pkl"),
            embedding_dimension=self.embedding_dim,
            embedding_mode="default",
        )
        self.embedding_means = load_clip_means() if self.fill_none_with_mean else None


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

    def _precompute_text_embeddings(self) -> None:
        self.texts = [" ".join(text_description) for text_description in self.text_descriptions]
        unique_texts = list(dict.fromkeys(self.texts))
        lookup = {text: i for i, text in enumerate(unique_texts)}
        with torch.no_grad():
            embeddings = self.clip_embedder.extract_clip_embeddings(unique_texts).cpu()
        self.text_embeddings = embeddings[[lookup[text] for text in self.texts]]
        self.clip_embedder = None

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
        text_embedding = self.text_embeddings[index]

        result = {
            "camera_trajectory": camera_trajectory,
            "subject_trajectory": subject_trajectory,
            "subject_volume": subject_volume,
            "padding_mask": padding_mask,
            "caption_feat": text_embedding,
            "text_prompts": text,
            "item_id": f"ccdm-{self.source_indices[index]:06d}",
        }
        if self.annotations is not None:
            from data.simulation.utils import fix_prompts_and_instructions

            row = self.annotations[result["item_id"]]
            prompt = row["cinematographyPrompts"][0]
            instruction = row["simulationInstructions"][0]
            values = fix_prompts_and_instructions(
                instruction, prompt, self.clip_embeddings,
                self.fill_none_with_mean, self.embedding_means,
            )
            result.update(zip(("simulation_instruction", "cinematography_prompt",
                               "prompt_none_mask", "simulation_instruction_parameters",
                               "cinematography_prompt_parameters"), values))
            result.update(raw_prompt=prompt, raw_instruction=instruction)
        return result

def collate_fn(batch):
    result = {
        **collate_trajectories(batch),
        "caption_feat": torch.stack([item["caption_feat"] for item in batch]),
        "text_prompts": [item["text_prompts"] for item in batch],
        "item_ids": [item["item_id"] for item in batch],
    }
    if "cinematography_prompt" in batch[0]:
        result.update(collate_structured_conditioning(batch))
        result["prompt_none_mask"] = torch.stack([item["prompt_none_mask"] for item in batch])
        result["raw_prompt"] = [item["raw_prompt"] for item in batch]
        result["raw_instruction"] = [item["raw_instruction"] for item in batch]
    return result
