import os
import sys
import numpy as np
import torch
import logging
import gdown

from models.clip_embeddings import CLIPEmbedder
from data.ccdm.utils import ccdm_to_simulation, generate_subject

logger = logging.getLogger(__name__)

class CCDMAdapter:    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.n_T = 1000
        self.n_feature = 5
        self.n_textemb = 512
        
        self.seq_len = getattr(config, "seq_len", 300)
        self.sim_seq_len = 30
        self.tan_half_fov_x = getattr(config, "tan_half_fov_x", 0.3639)
        self.tan_half_fov_y = getattr(config, "tan_half_fov_y", 0.2055)
        
        self.ddpm, self.clip_embedder, self.mean, self.std = self._load_models()
        
    def _load_models(self):
        current_sys_path = sys.path.copy()
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "third_parties/Camera-control/[2024][EG]Text+keyframe"))
        from main import Transformer, DDPM
        sys.path = current_sys_path
        
        transformer = Transformer(n_feature=self.n_feature, n_textemb=self.n_textemb)
        ddpm = DDPM(nn_model=transformer, betas=(1e-4, 0.02), n_T=self.n_T, device=self.device)
        ddpm.to(self.device)
        
        if not os.path.exists(self.config.ccdm_checkpoint_path):
            logger.info(f"Checkpoint file not found at {self.config.ccdm_checkpoint_path}, downloading...")
            os.makedirs(os.path.dirname(self.config.ccdm_checkpoint_path), exist_ok=True)
            gdown.download(id="136IZeL4PSf9L6FJ4n_jFM6QFLTDjbvr1", output=self.config.ccdm_checkpoint_path, quiet=False)
            logger.info(f"Downloaded checkpoint to {self.config.ccdm_checkpoint_path}")
        
        ddpm.load_state_dict(torch.load(self.config.ccdm_checkpoint_path, map_location=self.device))
        ddpm.eval()
        
        clip_embedder = CLIPEmbedder(
            model_name=getattr(self.config, "clip_model_name", "openai/clip-vit-base-patch32"),
            device=self.device,
            chunk_size=getattr(self.config, "chunk_size", 100)
        )
        
        mean_std_path = os.path.join(os.path.dirname(self.config.ccdm_data_dir), "Mean_Std.npy")
        data_path = os.path.join(os.path.dirname(self.config.ccdm_data_dir), "data.npy")
        
        if not os.path.exists(data_path):
            logger.info(f"Data file not found at {data_path}, downloading...")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            gdown.download(id="1VxmGy9szWShOKzWvIxrmgaNEkeqGPLJU", output=data_path, quiet=False)
            logger.info(f"Downloaded data file to {data_path}")
            
        if not os.path.exists(mean_std_path):
            data = np.load(data_path, allow_pickle=True)[()]
            d = np.concatenate(data["cam"], 0)
            mean, std = np.mean(d, 0), np.std(d, 0)
            np.save(mean_std_path, {"Mean": mean, "Std": std})
        
        mean_std_data = np.load(mean_std_path, allow_pickle=True)[()]
        
        mean = torch.tensor(mean_std_data["Mean"], dtype=torch.float32, device=self.device)
        std = torch.tensor(mean_std_data["Std"], dtype=torch.float32, device=self.device)
        
        return ddpm, clip_embedder, mean, std
    
    def _smooth_trajectory_batch(self, batch_trajectories, window_size=10, iterations=4):
        B, T, F = batch_trajectories.shape
        device = batch_trajectories.device
        smoothed = batch_trajectories

        idx = torch.arange(T, device=device)
        kernel = (idx[None, :] - idx[:, None]).abs() <= window_size
        kernel = kernel.float()
        kernel /= kernel.sum(dim=1, keepdim=True)
        kernel = kernel.unsqueeze(0).expand(B, -1, -1)

        traj = smoothed.permute(0, 2, 1)

        for _ in range(iterations):
            traj = torch.bmm(traj, kernel)

        return traj.permute(0, 2, 1)
        
    def process_batch(self, batch):
        text_prompts = batch["text_prompts"]
        
        with torch.no_grad():
            text_embeddings = self.clip_embedder.extract_clip_embeddings(text_prompts, return_seq=False).to(self.device)
            
            guide_w = float(self.config.get("guidance_scale", 2.0))
            generated = self.ddpm.sample(
                n_sample=len(text_prompts),
                c=text_embeddings,
                size=(300, self.n_feature),
                device=self.device,
                guide_w=guide_w
            )
            
            denormalized = generated * self.std[None, None, :] + self.mean[None, None, :]
            smoothed_batch = self._smooth_trajectory_batch(denormalized)
            
            camera_trajectory_sim, padding_mask = ccdm_to_simulation(
                smoothed_batch,
                self.sim_seq_len,
                self.tan_half_fov_x,
                self.tan_half_fov_y,
            )
            
            subject_loc_rot, subject_volume = generate_subject(self.seq_len)
            
            return {
                "camera_trajectory": camera_trajectory_sim,
                "subject_trajectory": subject_loc_rot,
                "subject_volume": subject_volume,
                "padding_mask": None if padding_mask is None else ~padding_mask,
                "caption_feat": text_embeddings,
                "raw_text": text_prompts,
                "original_camera_trajectory": smoothed_batch,
            }
