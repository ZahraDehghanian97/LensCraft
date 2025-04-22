import torch
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import torch.nn.functional as F
from torch.utils.data import Dataset
from models.clip_embeddings import CLIPEmbedder

class CCDMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        embedding_dim: int = 512,
        standardize: bool = True,
        seq_len: int = 30,
        default_focal_length: Optional[float] = None,
        fov_degrees: float = 45.0,
        sensor_width: float = 36.0,
        sensor_height: float = 24.0,
        device: str | torch.device | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        self.clip_model_name = clip_model_name
        self.standardize = standardize
        self.seq_len = seq_len

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
        self.camera_trajectories = [torch.tensor(camera_trajectory, dtype=torch.float32) for camera_trajectory in raw["camera_trajectory"]]
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
    
    def _transform_camera_format_to_simulation(self, camera_trajectory: torch.Tensor) -> torch.Tensor:
        frames_count = camera_trajectory.shape[0]
        simulation_format = torch.zeros((frames_count, 7), dtype=torch.float32)
        
        simulation_format[:, 0:3] = camera_trajectory[:, 0:3]
        
        for i in range(frames_count):
            p_x, p_y = camera_trajectory[i, 3:5]
            
            rot_x = torch.atan2(p_y, torch.tensor(1.0, dtype=torch.float32))
            rot_y = torch.atan2(p_x, torch.tensor(1.0, dtype=torch.float32))
            rot_z = torch.tensor(0.0, dtype=torch.float32)
            
            simulation_format[i, 3:7] = torch.tensor([rot_x, rot_y, rot_z, self.default_focal_length], dtype=torch.float32)
        
        return simulation_format
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        camera_trajectory = self.camera_trajectories[index]
        text_description = self.text_descriptions[index]
        
        traj_length = len(camera_trajectory)
        padding_mask = None
        
        if traj_length < self.seq_len:
            padding = camera_trajectory[-1:].repeat(self.seq_len - traj_length, 1)
            camera_trajectory = torch.cat([camera_trajectory, padding], dim=0)
            padding_mask = torch.cat([
                torch.ones(traj_length, dtype=torch.bool),
                torch.zeros(self.seq_len - traj_length, dtype=torch.bool)
            ])
        else:
            indices = torch.linspace(0, traj_length - 1, self.seq_len).long()
            camera_trajectory = camera_trajectory[indices]
        
        camera_trajectory_sim = self._transform_camera_format_to_simulation(camera_trajectory)
        
        text = " ".join(text_description)
        with torch.no_grad():
            text_embedding = self.clip_embedder.extract_clip_embeddings([text])[0].cpu()
        
        subject_loc_rot = torch.zeros((self.seq_len, 6), dtype=torch.float32)
        subject_volume = torch.tensor([[0.5, 1.7, 0.3]], dtype=torch.float32)
        
        return {
            "camera_trajectory": camera_trajectory_sim,
            "subject_trajectory": subject_loc_rot,
            "subject_volume": subject_volume,
            "padding_mask": None if padding_mask is None else ~padding_mask,
            "caption_feat": text_embedding,
            "raw_text": text,
            "original_camera_trajectory": camera_trajectory,
        }

def collate_fn(batch):
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": torch.stack([item["subject_trajectory"] for item in batch]),
        "subject_volume": torch.stack([item["subject_volume"] for item in batch]),
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]) if batch[0]["padding_mask"] is not None else None,
        "caption_feat": torch.stack([item["caption_feat"] for item in batch]),
        "original_camera_trajectory": torch.stack([item["original_camera_trajectory"] for item in batch]),
    }
