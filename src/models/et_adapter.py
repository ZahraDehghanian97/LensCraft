import os
import numpy as np
import torch
import logging
import gdown
import zipfile
import torch.nn.functional as F

from hydra.utils import instantiate

from data.et.caption_encoder import CaptionEncoder
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
        self.num_frames = config.get("num_frames", 30)
        self._load_models(config["project_config_dir"], config["dataset_dir"])
        set_random_seed(42)
        self.caption_encoder = CaptionEncoder(device=device)
    
    
    def _prepare_checkpoints(self, checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        clatr_output = os.path.join(checkpoints_dir, "clatr-e100.ckpt")
        if not os.path.exists(clatr_output):
            clatr_output = os.path.join(checkpoints_dir, "clatr-e100.ckpt")
            gdown.download(id="1FqN-pa955Wvu3utGViUKiVfza6cL_W0D", output=clatr_output, quiet=False)
        
        director_zip = os.path.join(checkpoints_dir, "director.zip")
        if not os.path.exists(director_zip):
            gdown.download(id="1uYeK1WcS3XI4uewHqi79RmLPggdpWgnG", output=director_zip, quiet=False)
        
        director_dir = os.path.join(checkpoints_dir, "director")
        if not os.path.exists(director_dir):
            with zipfile.ZipFile(director_zip, 'r') as zip_ref:
                zip_ref.extractall(checkpoints_dir)
            
            logger.info(f"Extracted DIRECTOR checkpoints to {director_dir}")
    
    def _load_models(self, project_config_dir, dataset_dir):
        checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(project_config_dir)), "checkpoints")
        self._prepare_checkpoints(checkpoints_dir)
        
        project_dir = os.path.dirname(os.path.dirname(project_config_dir))
        director_config = load_et_config(project_config_dir, "config_viz.yaml", dataset_dir=dataset_dir)
        with ModuleImporter.temporary_module(project_dir, ['utils.file_utils', 'utils.rotation_utils', 'utils.random_utils', 'utils.visualization'], project_dir):
            dataset = instantiate(director_config.dataset).set_split("test")
            self.diffuser = instantiate(director_config.diffuser)
            state_dict = torch.load(director_config.checkpoint_path, map_location=self.device)["state_dict"]
        
        state_dict["ema.initted"] = self.diffuser.ema.initted
        state_dict["ema.step"] = self.diffuser.ema.step
        self.diffuser.load_state_dict(state_dict, strict=False)
        self.diffuser.to(self.device).eval()        
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
    
    def _generate_caption_feat(self, text_prompts):
        caption_seq_list, caption_tokens = self.caption_encoder.encode_text(text_prompts)
        
        if self.diffuser.net.model.clip_sequential:
            padded_seqs = []
            for seq in caption_seq_list:
                padded_seq = F.pad(seq, (0, 0, 0, 77 - seq.shape[0]))
                padded_seqs.append(padded_seq)
            caption_feat = torch.stack(padded_seqs, dim=0)
            caption_feat = caption_feat.permute(0, 2, 1)
        else:
            caption_feat = caption_tokens
        
        return caption_feat
    
    def generate_using_text(self, text_prompts, subject_trajectory=None, padding_mask=None):
        self.diffuser.gen_seeds = np.arange(len(text_prompts))

        caption_feat = self._generate_caption_feat(text_prompts)
        char_feat = sim_to_et_subject_traj(subject_trajectory, self.device)
        
        batch = self._prepare_model_input(caption_feat, char_feat, self.num_frames)
        
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
