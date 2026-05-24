from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.clip_embeddings import CLIPEmbedder

from .clatr_model import NativeCLaTr
from .losses import InfoNCEWithFiltering, KLLoss


class LightningCLaTr(L.LightningModule):
    def __init__(
        self,
        traj_num_feats: int = 6,
        text_num_feats: int = 512,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        vae: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        lmd_recons: float = 1.0,
        lmd_latent: float = 1e-5,
        lmd_kl: float = 1e-5,
        lmd_nce: float = 0.1,
        temperature: float = 0.1,
        threshold_selfsim: float = 0.995,
        max_text_tokens: int = 77,
        clip_model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = NativeCLaTr(
            traj_num_feats=traj_num_feats,
            text_num_feats=text_num_feats,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            vae=vae,
        )

        self.recons_loss_fn = nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()
        self.contrastive_loss_fn = InfoNCEWithFiltering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )

        self.lmd = {
            "recons": lmd_recons,
            "latent": lmd_latent,
            "kl": lmd_kl,
            "contrastive": lmd_nce,
        }
        self.max_text_tokens = max_text_tokens

        self._clip_embedder: Optional[CLIPEmbedder] = None

    @property
    def clip_embedder(self) -> CLIPEmbedder:
        if self._clip_embedder is None:
            self._clip_embedder = CLIPEmbedder(
                model_name=self.hparams.clip_model_name,
                device=self.device,
            )
        return self._clip_embedder

    def _encode_text_prompts(
        self, prompts: List[str]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        clip_out = self.clip_embedder.extract_clip_embeddings(
            prompts, return_seq=True, pad_seq=False
        )
        seqs = clip_out.sequence_features.to(self.device)  # (B, L_i, D)
        valid_lens = clip_out.valid_lengths.to(self.device)
        pool = clip_out.pooled_features.to(self.device)

        bs = seqs.shape[0]
        D = seqs.shape[-1]
        L = self.max_text_tokens

        seq_feat = seqs.new_zeros((bs, L, D))
        seq_mask = torch.zeros((bs, L), dtype=torch.bool, device=self.device)
        for i in range(bs):
            li = int(valid_lens[i].item())
            li = max(1, min(li, L))
            seq_feat[i, :li] = seqs[i, :li]
            seq_mask[i, :li] = True

        return seq_feat, seq_mask, pool

    def compute_loss(
        self,
        traj: Tensor,
        traj_mask: Tensor,
        text_seq_feat: Tensor,
        text_seq_mask: Tensor,
        text_pool: Optional[Tensor],
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        """Compute the joint TEMOS + contrastive loss for one batch."""
        t_traj, t_z, t_dists = self.model(
            {"x": text_seq_feat, "mask": text_seq_mask},
            traj_mask,
            return_all=True,
        )
        m_traj, m_z, m_dists = self.model(
            {"x": traj, "mask": traj_mask},
            traj_mask,
            return_all=True,
        )

        losses: Dict[str, Tensor] = {}
        losses["recons"] = (
            self.recons_loss_fn(t_traj, traj) + self.recons_loss_fn(m_traj, traj)
        )
        losses["latent"] = self.latent_loss_fn(t_z, m_z)

        if self.hparams.vae and m_dists is not None and t_dists is not None:
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)
            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)
                + self.kl_loss_fn(m_dists, t_dists)
                + self.kl_loss_fn(m_dists, ref_dists)
                + self.kl_loss_fn(t_dists, ref_dists)
            )

        losses["contrastive"] = self.contrastive_loss_fn(t_z, m_z, text_pool)

        loss = sum(self.lmd[k] * v for k, v in losses.items() if k in self.lmd)
        losses["loss"] = loss
        return losses, t_z, m_z


    def _shared_step(self, batch: Dict[str, Any], stage: str) -> Tensor:
        traj = batch["camera_trajectory"].to(self.device)
        traj_mask = (~batch["padding_mask"].to(self.device)).bool()
        prompts: List[str] = batch["text_prompts"]

        seq_feat, seq_mask, pool = self._encode_text_prompts(prompts)
        losses, _, _ = self.compute_loss(traj, traj_mask, seq_feat, seq_mask, pool)

        bs = traj.shape[0]
        for name, value in losses.items():
            if not torch.is_tensor(value):
                continue
            self.log(
                f"{stage}/{name}",
                value.detach(),
                prog_bar=(name == "loss"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=bs,
                sync_dist=True,
            )
        return losses["loss"]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state = checkpoint.get("state_dict", {})
        for key in list(state.keys()):
            if key.startswith("_clip_embedder"):
                del state[key]
