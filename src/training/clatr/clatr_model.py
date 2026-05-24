from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: Tensor) -> Tensor:  # (B, L, D)
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class ACTORStyleEncoder(nn.Module):
    def __init__(
        self,
        num_feats: int,
        vae: bool = True,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.latent_dim = latent_dim
        self.vae = vae
        self.nbtokens = 2 if vae else 1

        self.projection = nn.Linear(num_feats, latent_dim)
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        bs = x.shape[0]
        device = x.device

        x = self.projection(x)
        dist_tokens = repeat(self.tokens, "n d -> b n d", b=bs)
        xseq = torch.cat([dist_tokens, x], dim=1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=torch.bool, device=device)
        aug_mask = torch.cat([token_mask, mask.to(torch.bool)], dim=1)
        xseq = self.sequence_pos_encoding(xseq)

        # nn.Transformer expects True = "ignore this position"
        out = self.seq_encoder(xseq, src_key_padding_mask=~aug_mask)
        return out[:, : self.nbtokens]


class ACTORStyleDecoder(nn.Module):
    def __init__(
        self,
        num_feats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.latent_dim = latent_dim

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seq_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_layer = nn.Linear(latent_dim, num_feats)

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        bs, nframes = mask.shape
        latent_dim = z.shape[-1]

        memory = z[:, None]  # (B, 1, D)
        time_queries = torch.zeros(bs, nframes, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        out = self.seq_decoder(
            tgt=time_queries,
            memory=memory,
            tgt_key_padding_mask=~mask.to(torch.bool),
        )
        out = self.final_layer(out)
        out[~mask.to(torch.bool)] = 0
        return out


class NativeCLaTr(nn.Module):
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
    ) -> None:
        super().__init__()
        self.vae = vae
        self.latent_dim = latent_dim
        self.traj_num_feats = traj_num_feats
        self.text_num_feats = text_num_feats

        common = dict(
            vae=vae,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.traj_encoder = ACTORStyleEncoder(num_feats=traj_num_feats, **common)
        self.text_encoder = ACTORStyleEncoder(num_feats=text_num_feats, **common)
        self.traj_decoder = ACTORStyleDecoder(
            num_feats=traj_num_feats,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )


    def _find_encoder(self, inputs: Dict[str, Tensor], modality: str) -> ACTORStyleEncoder:
        if modality == "text":
            return self.text_encoder
        if modality == "traj":
            return self.traj_encoder
        if modality != "auto":
            raise ValueError(f"Unknown modality: {modality}")

        if self.traj_num_feats == self.text_num_feats:
            raise ValueError(
                "Cannot auto-detect encoder; traj and text encoders share input dim."
            )
        feat_dim = inputs["x"].shape[-1]
        if feat_dim == self.traj_num_feats:
            return self.traj_encoder
        if feat_dim == self.text_num_feats:
            return self.text_encoder
        raise ValueError(f"Input feature dim {feat_dim} matches no encoder.")

    def encode(
        self,
        inputs: Dict[str, Tensor],
        modality: str = "auto",
        sample_mean: bool = False,
        fact: float = 1.0,
        return_distribution: bool = False,
    ):
        encoder = self._find_encoder(inputs, modality)
        encoded = encoder(inputs["x"], inputs["mask"])  # (B, nbtokens, D)

        if self.vae:
            mu, logvar = encoded.unbind(1)
            if sample_mean:
                z = mu
            else:
                std = logvar.mul(0.5).exp()
                eps = torch.randn_like(std)
                z = mu + fact * eps * std
            dists: Optional[Tuple[Tensor, Tensor]] = (mu, logvar)
        else:
            (z,) = encoded.unbind(1)
            dists = None

        if return_distribution:
            return z, dists
        return z

    def decode(self, z: Tensor, mask: Tensor) -> Tensor:
        return self.traj_decoder(z, mask)

    def forward(
        self,
        inputs: Dict[str, Tensor],
        mask: Tensor,
        sample_mean: bool = False,
        return_all: bool = False,
    ):
        z, dists = self.encode(inputs, sample_mean=sample_mean, return_distribution=True)
        traj = self.decode(z, mask)
        if return_all:
            return traj, z, dists
        return traj
