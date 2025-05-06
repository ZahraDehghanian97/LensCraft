import os, sys
import numpy as np
import torch
import logging

from hydra.utils import instantiate

from models.clip_embeddings import CLIPEmbedder
from data.et.utils import et_to_sim_cam_traj, sim_to_et_subject_traj
from data.et.load import load_et_config
from utils.seed import set_random_seed
from utils.importing import ModuleImporter

logger = logging.getLogger(__name__)

class ETAdapter:    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.guidance_scale = config.get("guidance_scale", 1.4)
        self._load_models(config["project_config_dir"])
        set_random_seed(42)
        self.clip_embedder = CLIPEmbedder(model_name="openai/clip-vit-base-patch32", device=device)
    
    
    def _load_models(self, project_config_dir):
        director_config = load_et_config(project_config_dir)

        with ModuleImporter.temporary_module(os.path.dirname(os.path.dirname(project_config_dir))):
            print(sys.path)
            dataset = instantiate(director_config.dataset)
            diffuser = instantiate(director_config.diffuser)
        state_dict = torch.load(director_config.checkpoint_path, map_location=self.device)["state_dict"]
        state_dict["ema.initted"] = diffuser.ema.initted
        state_dict["ema.step"] = diffuser.ema.step
        diffuser.load_state_dict(state_dict, strict=False)
        diffuser.to(self.device).eval()

        
        dataset.set_split("test")
        self.diffuser.modalities = list(dataset.modality_datasets.keys())
        self.diffuser.get_matrix = dataset.get_matrix
        self.diffuser.v_get_matrix = dataset.get_matrix  
        self.diffuser.to(self.device)
        self.diffuser.guidance_weight = self.config.get("guidance_scale", 1.4)

    def add_subject_trajectory_to_et_batch(self, et_batch, batch):
        device = et_batch["char_feat"].device
        subject_positions = batch["subject_trajectory"][:, :, :3].permute(0, 2, 1)
        et_batch["char_feat"] = torch.zeros_like(et_batch["char_feat"], device=device)
        et_batch["char_feat"][:, :, :subject_positions.shape[2]] = subject_positions.to(device)
    
    def _prepare_model_input(self, caption_feat, char_feat, num_frames=30):
        batch_size = caption_feat.shape[0]
        batch = {
            "char_feat": char_feat,
            "caption_feat": caption_feat,
        }
        
        padding_mask = torch.zeros([batch_size, 300])
        padding_mask[:, :num_frames] = 1
        for key in ["padding_mask", "char_padding_mask", "caption_padding_mask"]:
            batch[key] = padding_mask
        
        return batch
    
    def generate_using_text(self, text_prompts, subject_trajectory=None, padding_mask=None):
        self.diffuser.gen_seeds = np.arange(len(text_prompts))

        caption_feat = self.clip_embedder.get_caption_feat(text_prompts, seq_feat=False)
        char_feat = sim_to_et_subject_traj(subject_trajectory, self.device)
        
        batch = self._prepare_model_input(caption_feat, char_feat, padding_mask)
        
        with torch.no_grad():
            out = self.diffuser.predict_step(batch, 0)
            camera_trajectory = et_to_sim_cam_traj(
                out["gen_samples"][out["padding_mask"].to(bool)], 
                self.num_frames
            )
            
            return {
                "generated": camera_trajectory,
                "caption_feat": caption_feat,
            }
