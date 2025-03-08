import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import Dataset
from models.clip_embeddings import CLIPEmbedder

class CCDMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        embedding_dim: int = 512,
        standardize: bool = True,
        max_seq_len: int = 300,
    ):
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        self.clip_model_name = clip_model_name
        self.standardize = standardize
        self.max_seq_len = max_seq_len
        
        self._load_data()
        self.clip_embedder = CLIPEmbedder(
            model_name=self.clip_model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def _load_data(self):
        data_file = self.data_path
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found at {data_file}")
        
        data = np.load(data_file, allow_pickle=True)[()]
        
        self.camera_trajectories = data['cam']
        self.text_descriptions = data['info']
        
        if self.standardize:
            mean_std_file = self.data_path.parent / "Mean_Std.npy"
            if mean_std_file.exists():
                mean_std_data = np.load(mean_std_file, allow_pickle=True)[()]
                self.mean = mean_std_data['Mean']
                self.std = mean_std_data['Std']
            else:
                d = np.concatenate(self.camera_trajectories, 0)
                self.mean = np.mean(d, 0)
                self.std = np.std(d, 0)
                np.save(mean_std_file, {"Mean": self.mean, "Std": self.std})
            
            for i in range(len(self.camera_trajectories)):
                self.camera_trajectories[i] = (self.camera_trajectories[i] - self.mean[None, :]) / (self.std[None, :] + 1e-8)
    
    def __len__(self) -> int:
        return len(self.camera_trajectories)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        camera_trajectory = self.camera_trajectories[index]
        text_description = self.text_descriptions[index]
        
        traj_length = len(camera_trajectory)
        
        if traj_length < self.max_seq_len:
            padding = np.repeat(camera_trajectory[-1:], self.max_seq_len - traj_length, axis=0)
            camera_trajectory = np.concatenate([camera_trajectory, padding], axis=0)
            padding_mask = np.concatenate([
                np.ones(traj_length), 
                np.zeros(self.max_seq_len - traj_length)
            ])
        else:
            camera_trajectory = camera_trajectory[:self.max_seq_len]
            padding_mask = np.ones(self.max_seq_len)
        
        text = " ".join(text_description)
        with torch.no_grad():
            text_embedding = self.clip_embedder.get_embeddings([text])[0].cpu()
        
        subject_trajectory = np.zeros((self.max_seq_len, 9), dtype=np.float32)
        subject_trajectory[:, 3:6] = np.array([0.5, 1.7, 0.3])
        
        return {
            "camera_trajectory": torch.tensor(camera_trajectory, dtype=torch.float32),
            "subject_trajectory": torch.tensor(subject_trajectory, dtype=torch.float32),
            "padding_mask": torch.tensor(~padding_mask.astype(bool), dtype=torch.bool),
            "caption_feat": text_embedding,
            "raw_text": text,
        }

def collate_fn(batch):
    subject_trajectory = torch.stack([item["subject_trajectory"] for item in batch])
    subject_loc_rot = torch.cat([
        subject_trajectory[:, :, :3],
        subject_trajectory[:, :, 6:],
    ], dim=2)
    subject_vol = subject_trajectory[:, 0:1, 3:6].permute(0, 2, 1)
    
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": torch.stack([item["subject_trajectory"] for item in batch]),
        "subject_trajectory_loc_rot": subject_loc_rot,
        "subject_volume": subject_vol,
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
        "caption_feat": torch.stack([item["caption_feat"] for item in batch]),
    }
