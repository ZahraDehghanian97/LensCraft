"""Trajectory encoder whose only semantic output is ten categorical tokens."""
from __future__ import annotations

import torch
from torch import nn

from .categorical_latent import (
    FIELD_COUNT,
    FIELD_WIDTHS,
    SEMANTIC_DIM,
    TOKEN_WIDTH,
    fixed_query_codes,
    masked_category_logits,
    masked_softmax,
)
from .positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    """Predict one categorical distribution per cinematography field.

    The Transformer uses ``latent_dim`` internally. Its ten query states pass
    through one shared projection and the per-field category masks. The public
    output contains exactly 93 valid probabilities in ``[10,batch,21]``; the
    remaining 117 coordinates are padding and are exactly zero. Transformer
    states never leave this encoder or live in a mutable side-channel cache.

    ``encode_with_logits`` returns probabilities and masked logits from one
    Transformer pass so training can use stable cross-entropy directly.
    """

    def __init__(
        self, input_dim: int = 6, latent_dim: int = 256, nhead: int = 8,
        num_encoder_layers: int = 4, dim_feedforward: int = 1024,
        dropout_rate: float = 0.1, num_query_tokens: int = FIELD_COUNT,
    ):
        super().__init__()
        if num_query_tokens != FIELD_COUNT:
            raise ValueError("Encoder requires exactly ten cinematography queries")
        for name, value in (("input_dim", input_dim), ("latent_dim", latent_dim),
                            ("nhead", nhead), ("num_encoder_layers", num_encoder_layers),
                            ("dim_feedforward", dim_feedforward)):
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if latent_dim % nhead:
            raise ValueError("latent_dim must be divisible by nhead")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be between zero and one")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_query_tokens = FIELD_COUNT
        self.field_widths = tuple(FIELD_WIDTHS)
        self.token_width = TOKEN_WIDTH
        self.semantic_dim = SEMANTIC_DIM

        self.input_projection = nn.Linear(input_dim, latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim, dropout=dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
        )
        # Field/query identity is fixed analytic position information. There is
        # no learned categorical identity embedding or 46-query legacy path.
        self.register_buffer(
            "query_tokens", fixed_query_codes(width=latent_dim).unsqueeze(1),
            persistent=False,
        )
        self.semantic_projection = nn.Linear(latent_dim, TOKEN_WIDTH)

    @staticmethod
    def _mask(value, shape, device, name):
        if value is None:
            return torch.zeros(shape, dtype=torch.bool, device=device)
        if (not torch.is_tensor(value) or value.dtype != torch.bool
                or tuple(value.shape) != shape):
            raise ValueError(name + " must be boolean [batch, frames]")
        return value.to(device=device)

    def _validate_inputs(self, src, subject_embedded):
        if (not torch.is_tensor(src) or src.ndim != 3
                or src.shape[-1] != self.input_dim
                or src.shape[0] < 1 or src.shape[1] < 1
                or not src.is_floating_point()):
            raise ValueError("src must be floating [batch, frames, input_dim]")
        batch_size, frame_count = src.shape[:2]
        if (not torch.is_tensor(subject_embedded)
                or subject_embedded.shape != (batch_size, frame_count + 1, self.latent_dim)
                or not subject_embedded.is_floating_point()):
            raise ValueError(
                "subject_embedded must be floating [batch, frames + 1, latent_dim], "
                "with one final subject-volume token"
            )
        if subject_embedded.device != src.device:
            raise ValueError("Camera and subject inputs must share a device")
        return batch_size, frame_count

    def encode_with_logits(
        self, src, subject_embedded, src_key_padding_mask=None,
        subject_key_padding_mask=None,
    ):
        """Return ``(probabilities, masked_logits)`` from one encoder pass."""
        batch_size, frame_count = self._validate_inputs(src, subject_embedded)
        shape = (batch_size, frame_count)
        camera_mask = self._mask(src_key_padding_mask, shape, src.device,
                                 "src_key_padding_mask")
        subject_mask = self._mask(subject_key_padding_mask, shape, src.device,
                                  "subject_key_padding_mask")
        volume_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=src.device)
        subject_token_mask = torch.cat([subject_mask, volume_mask], dim=1)

        # Hidden NaNs must be removed before Linear and attention; a key mask
        # alone cannot prevent NaNs from contaminating projections or gradients.
        camera_input = src.masked_fill(camera_mask.unsqueeze(-1), 0)
        subject_input = subject_embedded.masked_fill(subject_token_mask.unsqueeze(-1), 0)
        camera_embedded = self.input_projection(camera_input)
        sequence = torch.cat([subject_input, camera_embedded], dim=1)
        sequence = self.pos_encoder(sequence).transpose(0, 1)
        queries = self.query_tokens.to(dtype=sequence.dtype).expand(-1, batch_size, -1)
        sequence = torch.cat([queries, sequence], dim=0)

        query_mask = torch.zeros(batch_size, FIELD_COUNT, dtype=torch.bool, device=src.device)
        # Order is ten queries, subject frames, one volume, camera frames.
        # Temporal padding never hides the volume token or a semantic query.
        attention_mask = torch.cat([query_mask, subject_token_mask, camera_mask], dim=1)
        encoded = self.transformer_encoder(sequence, src_key_padding_mask=attention_mask)
        logits = masked_category_logits(self.semantic_projection(encoded[:FIELD_COUNT]))
        probabilities = masked_softmax(logits)
        return probabilities, logits

    def forward(
        self, src, subject_embedded, src_key_padding_mask=None,
        subject_key_padding_mask=None,
    ):
        return self.encode_with_logits(
            src, subject_embedded, src_key_padding_mask, subject_key_padding_mask,
        )[0]
