import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from torch.utils.data import Dataset
from models.clip_embeddings import CLIPEmbedder
from data.convertor.convertor import convert_to_target

class CCDMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        embedding_dim: int = 512,
        standardize: bool = False,
        original_seq_len: int = 300,
        hfov_deg: float = 45.0,
        aspect: float = 16 / 9,
        device: str | torch.device | None = None,
        target: Optional[Dict[str, Any]] = None
    ) -> None:
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        self.clip_model_name = clip_model_name
        self.standardize = standardize
        self.original_seq_len = original_seq_len
        
        self.hfov_deg = hfov_deg
        self.aspect = aspect

        self.target = target

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.clip_embedder = CLIPEmbedder(model_name=self.clip_model_name, device=self.device)

        self._load_camera_trajectory_data()

    def _load_camera_trajectory_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        raw_data = np.load(self.data_path, allow_pickle=True)[()]
        self.camera_trajectories = [torch.tensor(camera_trajectory, dtype=torch.float32) for camera_trajectory in raw_data["cam"]]
        self.text_descriptions = raw_data["info"]

        if self.standardize:
            stats_file = self.data_path.parent / "Mean_Std.npy"
            stats = np.load(stats_file, allow_pickle=True)[()]
            self.mean = torch.tensor(stats["Mean"], dtype=torch.float32)
            self.std = torch.tensor(stats["Std"], dtype=torch.float32)
            self.camera_trajectories = [(c - self.mean) / (self.std + 1e-8) for c in self.camera_trajectories]
        else:
            feat_dim = self.camera_trajectories[0].shape[-1] if self.camera_trajectories else 5
            self.mean = torch.zeros(feat_dim, dtype=torch.float32)
            self.std = torch.ones(feat_dim, dtype=torch.float32)


    def __len__(self) -> int:
        return len(self.camera_trajectories)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        camera_trajectory = self.camera_trajectories[index].clone()
        text_description = self.text_descriptions[index]
        
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

        subject_trajectory = None
        subject_volume = None

        if self.target and "type" in self.target:
            camera_trajectory, subject_trajectory, subject_volume, padding_mask = convert_to_target(
                "ccdm",
                self.target["type"],
                camera_trajectory,
                subject_trajectory,
                subject_volume,
                padding_mask,
                self.target.get("seq_length", 30)
            )
        

        text = " ".join(text_description)
        with torch.no_grad():
            text_embedding = self.clip_embedder.extract_clip_embeddings([text])[0].cpu()


        return {
            "camera_trajectory": camera_trajectory,
            "subject_trajectory": subject_trajectory,
            "subject_volume": subject_volume,
            "padding_mask": padding_mask,
            "caption_feat": text_embedding,
            "text_prompts": text,
        }

def collate_fn(batch):
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": torch.stack([item["subject_trajectory"] for item in batch]),
        "subject_volume": torch.stack([item["subject_volume"] for item in batch]),
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
        "caption_feat": torch.stack([item["caption_feat"] for item in batch]),
        "text_prompts": [item["text_prompts"] for item in batch],
    }
