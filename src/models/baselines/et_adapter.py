import os
import numpy as np
import torch
import logging
import gdown
import zipfile
import torch.nn.functional as F
import torch

from hydra.utils import instantiate

from data.convertor.constant import default_convertors
from data.et.caption_encoder import CaptionEncoder
from data.et.config import STANDARDIZATION_CONFIG_TORCH
from data.et.load import load_et_config
from utils.seed import set_random_seed
from utils.importing import ModuleImporter

logger = logging.getLogger(__name__)

class ETAdapter:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.guidance_scale = config.get("guidance_scale", 1.4)
        self.undo_edm2_normalization = config.get("undo_edm2_normalization", True)
        self._load_models(config["project_config_dir"], config["dataset_dir"], config["et_type"])
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

    def _load_models(self, project_config_dir, dataset_dir, et_type):
        checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(project_config_dir)), "checkpoints")
        self._prepare_checkpoints(checkpoints_dir)

        project_dir = os.path.dirname(os.path.dirname(project_config_dir))
        director_config = load_et_config(project_config_dir, "config_viz.yaml", dataset_dir=dataset_dir, et_type=et_type)

        with ModuleImporter.temporary_module(project_dir, ['utils.file_utils', 'utils.rotation_utils', 'utils.random_utils', 'utils.visualization'], project_dir):
            dataset = instantiate(director_config.dataset).set_split("test")
            self.diffuser = instantiate(director_config.diffuser)
            checkpoint = torch.load(
                director_config.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            state_dict = checkpoint["state_dict"]

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

    def generate_using_text(self, text_prompts, subject_trajectory=None, trajectory=None, padding_mask=None):
        self.diffuser.gen_seeds = np.arange(len(text_prompts))
        caption_feat = self._generate_caption_feat(text_prompts)

        if self.config["et_type"] == "ca":
            char_feat = subject_trajectory.permute(0, 2, 1)
        else:
            char_feat = subject_trajectory.reshape(subject_trajectory.shape[0], -1)


        batch = {
            "traj_feat": trajectory.permute(0, 2, 1),
            "char_feat": char_feat,
            "caption_feat": caption_feat,
            "padding_mask": ~padding_mask,
            "char_padding_mask": ~padding_mask,
            "caption_padding_mask": ~padding_mask,
            "caption_raw": text_prompts,
            "char_raw": None,
        }

        with torch.no_grad():
            cond_keys = [k for k in batch if "traj" not in k and "feat" in k]
            cond_data = [batch[k] for k in cond_keys]
            _, gen_samples = self.diffuser.sample(
                self.diffuser.ema.ema_model,
                batch["traj_feat"],
                cond_data,
                batch["padding_mask"],
            )
            if getattr(self.diffuser, "edm2_normalization", False) and self.undo_edm2_normalization:
                gen_samples = gen_samples / self.diffuser.loss_fn.sigma_data
            gen_matrices = torch.stack([self.diffuser.get_matrix(x) for x in gen_samples])
            traj_6d = default_convertors['et'].get_feature(gen_matrices)
            self._log_generation_scale(traj_6d)
            return traj_6d

    def _log_generation_scale(self, traj_6d):
        if getattr(self, "_scale_logged", False):
            return
        self._scale_logged = True
        norm_std = STANDARDIZATION_CONFIG_TORCH["norm_std"].to(traj_6d.device)
        vel_std = traj_6d[:, 1:, 6:].reshape(-1, 3).std(dim=0)
        ratio = (vel_std / norm_std).tolist()
        logger.info(
            "E.T. generated velocity std / training norm_std per axis: %s "
            "(~1 = training amplitude; ~0.5 = edm2 sigma_data half-scale not undone)",
            [round(r, 3) for r in ratio],
        )
