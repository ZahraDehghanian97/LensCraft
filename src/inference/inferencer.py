import os
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Optional

from models.clip_embeddings import CLIPEmbedder
from .checkpoint_utils import load_checkpoint
from .trajectory_converter import TrajectoryConverter
from .trajectory_processor import TrajectoryData, TrajectoryProcessor

class ModelInference:
    def __init__(self, cfg: DictConfig):
        self.device = torch.device(
            cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")
        
        self.clip_embedder = CLIPEmbedder(
            model_name=cfg.clip.model_name,
            device=self.device
        )
        
        self.model = self._initialize_model(cfg)
        self.converter = TrajectoryConverter()
        self.processor = TrajectoryProcessor(cfg.output_dir)

    def _initialize_model(self, cfg: DictConfig) -> torch.nn.Module:
        model = instantiate(cfg.training.model)
        model = load_checkpoint(cfg.checkpoint_path, model, self.device)
        model.to(self.device)
        model.eval()
        return model


    def generate_from_text(
        self,
        text: str,
        data: TrajectoryData,
        output_path: str
    ):
        with torch.no_grad():
            caption_feat = self.clip_embedder.get_embeddings([text])
            data.caption_feat = caption_feat
            self.generate_from_caption_feat(data, output_path)
    
    def generate_from_caption_feat(
        self,
        data: TrajectoryData,
        output_path: str
    ):
        with torch.no_grad():
            caption_feat = data.caption_feat.to(self.device)
            
            subject_embedded = self.model.subject_projection(data['subject_trajectory'])
            output = self.model.single_step_decode(
                caption_feat.unsqueeze(0), 
                subject_embedded,
                tgt_key_padding_mask=data['padding_mask']
            ).squeeze(0)
            
            self.converter.convert_and_save_outputs(
                output, 
                os.path.join(output_path, "gen_traj.txt"), 
                is_camera=True
            )

    def reconstruct_trajectory(
        self,
        data: TrajectoryData,
        output_path: str,
        is_simulation: bool = False
    ) -> Optional[torch.Tensor]:
        with torch.no_grad():
            if not is_simulation:
                data['camera_trajectory'] = data['camera_trajectory'].transpose(1, 2)
            
            padding_mask = data.padding_mask.to(self.device) if data.padding_mask is not None else None
            output = self.model(
                data.camera_trajectory.to(self.device),
                data.subject_trajectory.to(self.device),
                tgt_key_padding_mask=padding_mask
            )['reconstructed'].squeeze(0)
            
            if is_simulation:
                return output
            
            self.converter.convert_and_save_outputs(
                output,
                os.path.join(output_path, "rec_traj.txt"),
                is_camera=True
            )
