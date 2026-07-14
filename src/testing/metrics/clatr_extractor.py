from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

from data.convertor.convertor import convert_to_target
from data.et.caption_encoder import CaptionEncoder
from data.et.load import load_et_config
from utils.importing import ModuleImporter


class CLaTrFeatureExtractor(nn.Module):
    def __init__(
        self,
        project_config_dir: str,
        dataset_dir: str,
        checkpoint_path: str,
        device: torch.device,
        num_cams: int = 120,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.device = device
        self.num_cams = num_cams
        self.latent_dim = latent_dim

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"CLaTr checkpoint not found at {checkpoint_path}. "
                "Download it (gdown id=1FqN-pa955Wvu3utGViUKiVfza6cL_W0D) "
                "before running CLaTr-based evaluation."
            )

        project_dir = os.path.dirname(os.path.dirname(project_config_dir))
        with ModuleImporter.temporary_module(
            project_dir,
            [
                "utils.file_utils",
                "utils.rotation_utils",
                "utils.random_utils",
                "utils.visualization",
            ],
            project_dir,
        ):
            director_cfg = load_et_config(
                project_config_dir,
                "config_viz.yaml",
                dataset_dir=dataset_dir,
                et_type="ca",
            )
            self.clatr = instantiate(director_cfg.diffuser.clatr)

            raw_state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state_dict = raw_state.get("state_dict", raw_state)
            missing, unexpected = self.clatr.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[CLaTr] missing keys ({len(missing)}): {missing[:5]} ...")
            if unexpected:
                print(f"[CLaTr] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

        self.clatr.eval().to(device)
        for p in self.clatr.parameters():
            p.requires_grad_(False)

        self.caption_encoder = CaptionEncoder(device=str(device))

        self._text_ctx = 77

    @torch.no_grad()
    def encode_trajectory(
        self,
        sim_camera_trajectory: torch.Tensor,
        sim_subject_trajectory: torch.Tensor,
        sim_subject_volume: Optional[torch.Tensor],
        sim_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        traj_et, _, _, pad_et = convert_to_target(
            "simulation",
            "et",
            sim_camera_trajectory,
            sim_subject_trajectory,
            sim_subject_volume,
            sim_padding_mask,
            target_len=self.num_cams,
        )

        x = traj_et.to(self.device).float()
        valid_mask = (~pad_et.to(self.device)).bool()

        latent = self.clatr.encode(
            {"x": x, "mask": valid_mask},
            modality="traj",
            sample_mean=True,
        )
        return latent

    @torch.no_grad()
    def encode_text(self, captions: List[str]) -> torch.Tensor:
        seq_list, _ = self.caption_encoder.encode_text(captions)

        padded = torch.stack(
            [
                F.pad(s, (0, 0, 0, max(0, self._text_ctx - s.shape[0])))[: self._text_ctx]
                for s in seq_list
            ],
            dim=0,
        ).to(self.device)

        mask = torch.zeros(
            padded.shape[:2], dtype=torch.bool, device=self.device
        )
        for i, s in enumerate(seq_list):
            mask[i, : min(s.shape[0], self._text_ctx)] = True

        latent = self.clatr.encode(
            {"x": padded.float(), "mask": mask},
            modality="text",
            sample_mean=True,
        )
        return latent

    @property
    def feature_dim(self) -> int:
        return self.latent_dim
