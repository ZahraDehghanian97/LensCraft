"""Decode camera poses while attending directly to ten 21D categorical fields.

The public memory remains a semantic probability tensor. Fixed field codes are
added to a separate attention input; only the attention module's own K/V
projections map that input to the hidden transformer width.
"""
from __future__ import annotations

import torch
from torch import nn

from .categorical_latent import (
    FIELD_COUNT, TOKEN_WIDTH, categorical_presence, fixed_field_codes,
    valid_category_mask, validate_semantic,
)
from .positional_encoding import PositionalEncoding


class SemanticTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Ordinary decoder layer with native 21D cross-attention keys and values."""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__(d_model=d_model, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            kdim=TOKEN_WIDTH, vdim=TOKEN_WIDTH,
        )


class Decoder(nn.Module):
    """Source-compatible camera decoder with unexpanded categorical memory.

    ``memory`` has shape [10, batch, 21]. ``memory_presence_mask`` is boolean
    [batch, 10]. Zero fields are always absent; an explicit mask can additionally
    hide fields. For a sample with no present fields, one zero slot is unmasked
    without a field code. Neither the supplied semantic memory nor its mask is
    modified.
    """

    def __init__(self, output_dim=12, latent_dim=256, nhead=8, num_decoder_layers=4,
                 dim_feedforward=1024, dropout_rate=0.1, seq_length=30):
        nn.Module.__init__(self)
        if output_dim != 12:
            raise ValueError('Categorical decoding uses pose12: position3 plus SVD9D rotation')
        if not isinstance(seq_length, int) or isinstance(seq_length, bool) or seq_length < 1:
            raise ValueError('seq_length must be a positive integer')
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.pos_encoder = PositionalEncoding(latent_dim, dropout=dropout_rate)
        self.embedding = nn.Linear(output_dim, latent_dim)
        presence = torch.ones(latent_dim)
        presence[1::2] = -1
        self.register_buffer('known_pose_presence_encoding',
                             presence / latent_dim ** 0.5, persistent=False)
        valid_categories = valid_category_mask()
        self.register_buffer('valid_categories', valid_categories, persistent=False)
        self.register_buffer('field_codes', fixed_field_codes(width=TOKEN_WIDTH).masked_fill(
            ~valid_categories, 0), persistent=False)
        layer = SemanticTransformerDecoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)
        self.output_projection = nn.Linear(latent_dim, output_dim)

    def prepare_decoder_inputs_with_positioning(self, decoder_input, subject_embedding, tgt_key_padding_mask=None, known_pose_mask=None):
        embedded = self.embedding(decoder_input)
        if known_pose_mask is not None:
            embedded = embedded + known_pose_mask.unsqueeze(-1).to(embedded.dtype) * self.known_pose_presence_encoding.to(embedded)
        embedded = torch.cat([subject_embedding, embedded], dim=1)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)

        if tgt_key_padding_mask is not None:
            batch_size = tgt_key_padding_mask.shape[0]
            subject_volume_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=tgt_key_padding_mask.device)
            tgt_key_padding_mask = torch.cat(
                [tgt_key_padding_mask, subject_volume_mask, tgt_key_padding_mask], dim=1)

        return embedded, tgt_key_padding_mask


    def prepare_attention_memory(self, memory, memory_presence_mask=None):
        """Prepare 21D attention K/V and a safe [batch, 10] padding mask."""
        if (not torch.is_tensor(memory) or memory.ndim != 3
                or memory.shape[0] != FIELD_COUNT or memory.shape[-1] != TOKEN_WIDTH
                or not memory.is_floating_point()):
            raise ValueError('memory must be floating point [10, batch, 21]')
        batch_size = memory.shape[1]
        if memory_presence_mask is None:
            present = torch.ones(batch_size, FIELD_COUNT, device=memory.device, dtype=torch.bool)
        else:
            if (not torch.is_tensor(memory_presence_mask)
                    or memory_presence_mask.dtype != torch.bool
                    or memory_presence_mask.shape != (batch_size, FIELD_COUNT)):
                raise ValueError('memory_presence_mask must be boolean [batch, 10]')
            present = memory_presence_mask.to(device=memory.device)
        # Explicitly absent fields may contain arbitrary hidden fillers. A zero
        # field is also allowed for random field dropout. Every nonzero field
        # must still be a normalized categorical distribution, never logits.
        semantic_memory = memory.masked_fill(~present.T[..., None], 0)
        validate_semantic(semantic_memory, atol=max(1e-5, 2 * torch.finfo(memory.dtype).eps))
        present = categorical_presence(semantic_memory)
        # This separate tensor is used only as cross-attention keys and values.
        attention_memory = semantic_memory + self.field_codes.to(memory)[:, None, :]
        attention_memory = attention_memory.masked_fill(~self.valid_categories[:, None, :], 0)
        attention_memory = attention_memory.masked_fill(~present.T[..., None], 0)
        safe_present = present.clone()
        safe_present[:, 0] |= ~present.any(dim=1)
        return attention_memory, ~safe_present

    def _validate_context(self, memory, subject_embedding, tgt_key_padding_mask):
        expected = (memory.shape[1], self.seq_length + 1, self.latent_dim)
        if (not torch.is_tensor(subject_embedding) or subject_embedding.shape != expected
                or subject_embedding.device != memory.device):
            raise ValueError(f'subject_embedding must have shape {expected} on the memory device')
        if tgt_key_padding_mask is not None and (
                not torch.is_tensor(tgt_key_padding_mask)
                or tgt_key_padding_mask.dtype != torch.bool
                or tgt_key_padding_mask.shape != (memory.shape[1], self.seq_length)
                or tgt_key_padding_mask.device != memory.device):
            raise ValueError('tgt_key_padding_mask must be boolean [batch, frames] on the memory device')
        if tgt_key_padding_mask is not None:
            volume_padding = torch.zeros(memory.shape[1], 1, device=memory.device, dtype=torch.bool)
            subject_padding = torch.cat([tgt_key_padding_mask, volume_padding], dim=1)
            subject_embedding = subject_embedding.masked_fill(subject_padding[..., None], 0)
        if not torch.isfinite(subject_embedding).all():
            raise ValueError('Visible subject embeddings must contain only finite values')
        return subject_embedding

    def _known_poses(self, memory, tgt_key_padding_mask, known_camera_poses, known_pose_mask):
        if known_camera_poses is None and known_pose_mask is None:
            return None, None
        if known_camera_poses is None or known_pose_mask is None:
            raise ValueError('known_camera_poses and known_pose_mask must be supplied together')
        expected = (memory.shape[1], self.seq_length, self.output_dim)
        if known_camera_poses.shape != expected or known_pose_mask.shape != expected[:2]:
            raise ValueError('Known poses and their mask must match the decoded batch and frames')
        known_pose_mask = known_pose_mask.to(device=memory.device, dtype=torch.bool)
        if tgt_key_padding_mask is not None:
            known_pose_mask = known_pose_mask & ~tgt_key_padding_mask
        visible = known_camera_poses.to(device=memory.device, dtype=self.embedding.weight.dtype)
        visible = visible.masked_fill(~known_pose_mask[..., None], 0)
        if not torch.isfinite(visible).all():
            raise ValueError('Visible camera poses must contain only finite values')
        return visible, known_pose_mask

    def single_step_decode(self, memory, subject_embedding, tgt_key_padding_mask=None,
                           known_camera_poses=None, known_pose_mask=None,
                           memory_presence_mask=None):
        attention_memory, memory_padding = self.prepare_attention_memory(memory, memory_presence_mask)
        subject_embedding = self._validate_context(memory, subject_embedding, tgt_key_padding_mask)
        known_camera_poses, known_pose_mask = self._known_poses(
            memory, tgt_key_padding_mask, known_camera_poses, known_pose_mask)
        decoder_input = torch.zeros(memory.shape[1], self.seq_length, self.output_dim,
                                    device=memory.device, dtype=self.embedding.weight.dtype)
        if known_camera_poses is not None:
            decoder_input = known_camera_poses
        embedded, padded_mask = self.prepare_decoder_inputs_with_positioning(
            decoder_input, subject_embedding, tgt_key_padding_mask, known_pose_mask)
        output = self.transformer_decoder(
            tgt=embedded, memory=attention_memory, tgt_key_padding_mask=padded_mask,
            memory_key_padding_mask=memory_padding,
        )
        return self.output_projection(output.transpose(0, 1)[:, -self.seq_length:, :])

    def autoregressive_decode(self, memory, subject_embedding, target=None,
                              teacher_forcing_ratio=0.5, tgt_key_padding_mask=None,
                              feedback_transform=None, known_camera_poses=None,
                              known_pose_mask=None, memory_presence_mask=None):
        attention_memory, memory_padding = self.prepare_attention_memory(memory, memory_presence_mask)
        subject_embedding = self._validate_context(memory, subject_embedding, tgt_key_padding_mask)
        known_camera_poses, known_pose_mask = self._known_poses(
            memory, tgt_key_padding_mask, known_camera_poses, known_pose_mask)
        batch_size, device = memory.shape[1], memory.device
        shape = (batch_size, self.seq_length, self.output_dim)
        if target is not None and target.shape != shape:
            raise ValueError(f'target must have shape {shape}')
        output_trajectory = torch.zeros(shape, device=device, dtype=self.embedding.weight.dtype)
        decoder_input = torch.zeros_like(output_trajectory)
        subject_length = subject_embedding.shape[1]
        length = subject_length + self.seq_length
        causal_mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
        for timestep in range(self.seq_length):
            embedded, padded_mask = self.prepare_decoder_inputs_with_positioning(
                decoder_input, subject_embedding, tgt_key_padding_mask)
            decoded = self.transformer_decoder(
                tgt=embedded, memory=attention_memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=padded_mask, memory_key_padding_mask=memory_padding,
            )
            current = self.output_projection(decoded[subject_length + timestep])
            output_trajectory[:, timestep, :] = current
            if timestep < self.seq_length - 1:
                use_target = target is not None and torch.rand(1).item() < teacher_forcing_ratio
                if use_target:
                    feedback = target[:, timestep, :]
                else:
                    feedback = feedback_transform(current) if feedback_transform is not None else current
                if known_pose_mask is not None:
                    feedback = torch.where(known_pose_mask[:, timestep, None],
                                           known_camera_poses[:, timestep], feedback)
                updated_input = decoder_input.clone()
                updated_input[:, timestep, :] = feedback
                decoder_input = updated_input
        return output_trajectory

    def forward(self, memory, subject_embedding, decode_mode='single_step', target=None,
                teacher_forcing_ratio=0.0, tgt_key_padding_mask=None, feedback_transform=None,
                known_camera_poses=None, known_pose_mask=None, memory_presence_mask=None):
        if decode_mode == 'single_step':
            return self.single_step_decode(memory, subject_embedding, tgt_key_padding_mask,
                                           known_camera_poses, known_pose_mask, memory_presence_mask)
        if decode_mode == 'autoregressive':
            return self.autoregressive_decode(
                memory, subject_embedding, target, teacher_forcing_ratio, tgt_key_padding_mask,
                feedback_transform, known_camera_poses, known_pose_mask, memory_presence_mask,
            )
        raise ValueError(f'Unknown decode_mode: {decode_mode}')
