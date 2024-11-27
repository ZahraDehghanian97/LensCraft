import os

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from models.clip_embeddings import CLIPEmbedder
from .checkpoint_utils import load_checkpoint
from .trajectory_converter import TrajectoryConverter


class ModelInference:
    def __init__(self, cfg: DictConfig):
        self.device = torch.device(
            cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")

        self.clip_embedder = CLIPEmbedder(
            model_name=cfg.clip.model_name,
            device=self.device
        )

        self.model = instantiate(cfg.training.model)
        self.model = load_checkpoint(
            cfg.checkpoint_path, self.model, self.device)
        self.model.to(self.device)
        self.model.eval()

        self.converter = TrajectoryConverter()

    def generate_from_text(
        self,
        text: str,
        subject_trajectory: torch.Tensor,
        output_path: str
    ):
        with torch.no_grad():
            caption_feat = self.clip_embedder.get_embeddings([text])
            self.generate_from_caption_feat(
                caption_feat, 
                subject_trajectory, 
                output_path
            )
    
    def generate_from_caption_feat(
        self,
        caption_feat: torch.Tensor,
        subject_trajectory: torch.Tensor,
        padding_mask: torch.Tensor,
        output_path: str
    ):
        with torch.no_grad():
            caption_feat = caption_feat.to(self.device)
            subject_trajectory = subject_trajectory.to(self.device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            
            subject_embedded = self.model.subject_projection(subject_trajectory)
            output = self.model.single_step_decode(
                caption_feat.unsqueeze(0), 
                subject_embedded,
                tgt_key_padding_mask=padding_mask
            ).squeeze(0)
            
            self.converter.convert_and_save_outputs(
                output, 
                os.path.join(output_path, "gen_traj.txt"), 
                is_camera=True
            )

    def reconstruct_trajectory(
        self,
        camera_trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor,
        padding_mask: torch.Tensor,
        output_path: str
    ):
        with torch.no_grad():
            camera_trajectory = camera_trajectory.transpose(1, 2)
            
            camera_trajectory = camera_trajectory.to(self.device)
            subject_trajectory = subject_trajectory.to(self.device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)

            output = self.model(
                camera_trajectory, 
                subject_trajectory,
                tgt_key_padding_mask=padding_mask
            )['reconstructed'].squeeze(0)
            
            self.converter.convert_and_save_outputs(
                output, 
                os.path.join(output_path, "rec_traj.txt"), 
                is_camera=True
            )
