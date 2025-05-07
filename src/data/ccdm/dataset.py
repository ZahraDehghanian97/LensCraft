import torch
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from torch.utils.data import Dataset
from models.clip_embeddings import CLIPEmbedder
from data.convertor import camera_ccdm_to_sim, subject_ccdm_to_sim

class CCDMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        embedding_dim: int = 512,
        standardize: bool = False,
        seq_len: int = 30,
        original_seq_len: int = 300,
        default_focal_length: Optional[float] = None,
        fov_degrees: float = 45.0,
        sensor_width: float = 36.0,
        sensor_height: float = 24.0,
        device: str | torch.device | None = None,
        target = None
    ) -> None:
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        self.clip_model_name = clip_model_name
        self.standardize = standardize
        self.seq_len = seq_len
        self.original_seq_len = original_seq_len

        self.fov_degrees = fov_degrees
        self.fov_rad = math.radians(fov_degrees)
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width
        aspect = self.sensor_width / self.sensor_height
        self.fov_y_rad = self.fov_rad
        self.fov_x_rad = 2.0 * math.atan(math.tan(self.fov_y_rad * 0.5) * aspect)

        self.tan_half_fov_y = math.tan(self.fov_y_rad * 0.5)
        self.tan_half_fov_x = math.tan(self.fov_x_rad * 0.5)

        self.target = target

        if default_focal_length is not None:
            self.focal_length_mm = float(default_focal_length)
        else:
            diag = math.hypot(sensor_width, sensor_height)
            self.focal_length_mm = diag / (2 * math.tan(math.radians(fov_degrees * 0.5)))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.clip_embedder = CLIPEmbedder(model_name=self.clip_model_name, device=self.device)

        self._load_camera_trajectory_data()

    def _load_camera_trajectory_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        raw = np.load(self.data_path, allow_pickle=True)[()]
        self.camera_trajectories = [torch.tensor(camera_trajectory, dtype=torch.float32) for camera_trajectory in raw["cam"]]
        self.text_descriptions = raw["info"]

        if self.standardize:
            stats_file = self.data_path.parent / "Mean_Std.npy"
            stats = np.load(stats_file, allow_pickle=True)[()]
            self.mean = torch.tensor(stats["Mean"], dtype=torch.float32)
            self.std = torch.tensor(stats["Std"], dtype=torch.float32)
            self.camera_trajectories = [(c - self.mean) / (self.std + 1e-8) for c in self.camera_trajectories]
        else:
            self.mean = torch.zeros(5)
            self.std = torch.ones(5)

    def __len__(self) -> int:
        return len(self.camera_trajectories)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        camera_trajectory = self.camera_trajectories[index]
        text_description = self.text_descriptions[index]

        if len(camera_trajectory) < self.original_seq_len:
            pad = camera_trajectory[-1:].repeat(self.original_seq_len - len(camera_trajectory), 1)
            raw_padded = torch.cat([camera_trajectory, pad], dim=0)
        else:
            idxs = torch.linspace(0, len(camera_trajectory)-1, self.original_seq_len).long()
            raw_padded = camera_trajectory[idxs]

        if self.target is not None and self.target["type"] == "simulation":
            camera_trajectory_sim, padding_mask = camera_ccdm_to_sim(
                camera_trajectory,
                self.seq_len,
                self.tan_half_fov_x,
                self.tan_half_fov_y,
            )
        

        text = " ".join(text_description)
        with torch.no_grad():
            text_embedding = self.clip_embedder.extract_clip_embeddings([text])[0].cpu()

        subject_loc_rot, subject_volume = subject_ccdm_to_sim(seq_len=self.seq_len)

        return {
            "camera_trajectory": camera_trajectory_sim,
            "subject_trajectory": subject_loc_rot,
            "subject_volume": subject_volume,
            "padding_mask": None if padding_mask is None else ~padding_mask,
            "caption_feat": text_embedding,
            "text_prompts": text,
            "original_camera_trajectory": raw_padded,
        }

def collate_fn(batch):
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": torch.stack([item["subject_trajectory"] for item in batch]),
        "subject_volume": torch.stack([item["subject_volume"] for item in batch]),
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]) if batch[0]["padding_mask"] is not None else None,
        "caption_feat": torch.stack([item["caption_feat"] for item in batch]),
        "text_prompts": [item["text_prompts"] for item in batch],
        "original_camera_trajectory": torch.stack([item["original_camera_trajectory"] for item in batch]),
    }