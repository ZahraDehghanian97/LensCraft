from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn

from models.clip_embeddings import CLIPEmbedder
from training.clatr.lightning_module import LightningCLaTr
from utils.clatr_text import prepare_text_features

logger = logging.getLogger(__name__)


class NativeCLaTrFeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        max_text_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Native CLaTr checkpoint not found at '{checkpoint_path}'. "
                "Train one first with `python src/train_clatr.py`, or set "
                "`clatr_backend=et` in your test config to use the legacy "
                "E.T.-trained CLaTr instead."
            )

        self.lit = LightningCLaTr.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
        self.lit.eval()
        for p in self.lit.parameters():
            p.requires_grad_(False)
        self.lit.model.to(device)

        self.device = device
        self.latent_dim = int(self.lit.hparams.latent_dim)
        self.traj_num_feats = int(self.lit.hparams.traj_num_feats)
        self.max_text_tokens = int(
            max_text_tokens
            if max_text_tokens is not None
            else self.lit.hparams.max_text_tokens
        )

        train_clip = getattr(self.lit.hparams, "clip_model_name", clip_model_name)
        self.clip_embedder = CLIPEmbedder(model_name=train_clip, device=str(device))

        logger.info(
            "Loaded native CLaTr (latent_dim=%d, traj_num_feats=%d) from %s",
            self.latent_dim,
            self.traj_num_feats,
            checkpoint_path,
        )

    @torch.no_grad()
    def encode_trajectory(
        self,
        sim_camera_trajectory: torch.Tensor,
        sim_subject_trajectory: Optional[torch.Tensor] = None,  # unused
        sim_subject_volume: Optional[torch.Tensor] = None,      # unused
        sim_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = sim_camera_trajectory.to(self.device).float()
        if x.shape[-1] != self.traj_num_feats:
            raise ValueError(
                f"Trajectory feature dim {x.shape[-1]} does not match the "
                f"native CLaTr's expected {self.traj_num_feats}. Was the "
                "checkpoint trained on a different dataset format?"
            )

        if sim_padding_mask is None:
            valid_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=self.device)
        else:
            valid_mask = (~sim_padding_mask.to(self.device)).bool()

        latent = self.lit.model.encode(
            {"x": x, "mask": valid_mask},
            modality="traj",
            sample_mean=True,
        )
        return latent

    @torch.no_grad()
    def encode_text(self, captions: List[str]) -> torch.Tensor:
        """Encode raw captions into the shared CLaTr latent space."""
        clip_out = self.clip_embedder.extract_clip_embeddings(
            captions, return_seq=True, pad_seq=False
        )
        seqs = clip_out.sequence_features.to(self.device)
        valid_lens = clip_out.valid_lengths.to(self.device)

        padded, mask = prepare_text_features(
            seqs, valid_lens, self.max_text_tokens
        )
        latent = self.lit.model.encode(
            {"x": padded.float(), "mask": mask},
            modality="text",
            sample_mean=True,
        )
        return latent

    @property
    def feature_dim(self) -> int:
        return self.latent_dim
