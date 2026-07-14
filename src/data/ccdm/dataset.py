import os
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from torch.utils.data import Dataset
from models.clip_embeddings import CLIPEmbedder
from data.convertor.convertor import convert_to_target
from data.collate_utils import stack_optional

class CCDMDataset(Dataset):
    _normalization_parameters = None

    def __init__(
        self,
        data_path: str,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        embedding_dim: int = 512,
        normalize: bool = True,
        original_seq_len: int = 300,
        hfov_deg: float = 45.0,
        aspect: float = 16 / 9,
        device: str | torch.device | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        CCDMDataset.get_normalization_parameters(self.data_path.parent)
        self.embedding_dim = embedding_dim
        self.clip_model_name = clip_model_name
        self.normalize = normalize
        self.original_seq_len = original_seq_len

        self.hfov_deg = hfov_deg
        self.aspect = aspect

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.clip_embedder = CLIPEmbedder(model_name=self.clip_model_name, device=self.device)
        self.padding_masks = []

        self._load_camera_trajectory_data()
        self._precompute_text_embeddings()


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
        self.camera_trajectories = [torch.tensor(camera_trajectory, dtype=torch.float32) for camera_trajectory in raw_data["cam"]]
        self.text_descriptions = raw_data["info"]


        for index, camera_trajectory in enumerate(self.camera_trajectories):
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
        with torch.no_grad():
            embeddings = self.clip_embedder.extract_clip_embeddings(self.texts)
        self.text_embeddings = embeddings.cpu()
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

        return {
            "camera_trajectory": camera_trajectory,
            "subject_trajectory": subject_trajectory,
            "subject_volume": subject_volume,
            "padding_mask": padding_mask,
            "caption_feat": text_embedding,
            "text_prompts": text,
        }

def collate_fn(batch):
    subject_volume = stack_optional(batch, "subject_volume")
    subject_trajectory = stack_optional(batch, "subject_trajectory")

    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": subject_trajectory,
        "subject_volume": subject_volume,
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
        "caption_feat": torch.stack([item["caption_feat"] for item in batch]),
        "text_prompts": [item["text_prompts"] for item in batch],
    }
