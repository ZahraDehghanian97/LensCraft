"""Frozen semantic metric judge, with no camera generation or decoder path.

The encoder and positional encoding reproduce the fixed semantic098 checkpoint
architecture. Its former decoder/merger weights are intentionally never loaded.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral
from types import SimpleNamespace

import torch
from torch import nn

from data.simulation.constants import NumericFeature, cinematography_struct, simulation_struct

FROZEN_ENCODER_SOURCE_SHA256 = '500774072f28ac14e88563d9f11de7a0317168eb08de27486021d4d9b8747c63'
FROZEN_POSITION_SOURCE_SHA256 = '0dbd86fe66486c1d0f055cfbc51db570d9aaa75476dbcd12c360f05c952d3c36'

class _FrozenPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(_FrozenPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class _FrozenSemanticEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nhead, num_encoder_layers, dim_feedforward, dropout_rate, num_query_tokens):
        super(_FrozenSemanticEncoder, self).__init__()

        self.input_projection = nn.Linear(input_dim, latent_dim)
        self.pos_encoder = _FrozenPositionalEncoding(latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, 1, latent_dim))

    def forward(
        self,
        src,
        subject_embedded,
        src_key_padding_mask=None,
        subject_key_padding_mask=None,
    ):
        src_embedded = self.input_projection(src)
        src_embedded = torch.cat([subject_embedded, src_embedded], dim=1)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = src_embedded.permute(1, 0, 2)

        query_tokens = self.query_tokens.repeat(1, src_embedded.shape[1], 1)
        src_with_queries = torch.cat([query_tokens, src_embedded], dim=0)

        batch_size, frame_count = src.shape[:2]
        if subject_embedded.shape[:2] != (batch_size, frame_count + 1):
            raise ValueError(
                "subject_embedded must contain one token per frame followed "
                "by exactly one volume token"
            )

        if (
            src_key_padding_mask is not None
            or subject_key_padding_mask is not None
        ):
            def normalize_mask(mask, name):
                if mask is None:
                    return torch.zeros(
                        (batch_size, frame_count),
                        dtype=torch.bool,
                        device=src.device,
                    )
                mask = mask.to(device=src.device, dtype=torch.bool)
                if mask.shape != (batch_size, frame_count):
                    raise ValueError(
                        f"{name} must have shape "
                        f"{(batch_size, frame_count)}, got {tuple(mask.shape)}"
                    )
                return mask

            camera_mask = normalize_mask(
                src_key_padding_mask, "src_key_padding_mask"
            )
            subject_mask = normalize_mask(
                subject_key_padding_mask, "subject_key_padding_mask"
            )
            query_mask = torch.zeros(
                (batch_size, self.query_tokens.shape[0]),
                dtype=torch.bool,
                device=src.device,
            )
            volume_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=src.device
            )
            # Token order is queries, subject frames, subject volume, camera
            # frames. Keeping the volume mask separate avoids shifting the
            # subject mask by one and accidentally hiding the volume whenever
            # the final trajectory frame is padded/masked.
            src_key_padding_mask = torch.cat(
                [
                    query_mask,
                    subject_mask,
                    volume_mask,
                    camera_mask,
                ],
                dim=1,
            )

        memory = self.transformer_encoder(
            src_with_queries, src_key_padding_mask=src_key_padding_mask)

        return memory[:self.query_tokens.shape[0]]


class FrozenSemanticEvaluator(nn.Module):
    """Metric-only 46-slot trajectory encoder for the existing frozen judge."""
    def __init__(self, input_dim=6, subject_dim=6, nhead=8, num_encoder_layers=4,
                 dim_feedforward=2048, dropout_rate=0.3, seq_length=30,
                 latent_dim=512, num_query_tokens=46):
        super().__init__()
        if num_query_tokens != 46:
            raise ValueError('The frozen semantic judge requires all 46 historical query slots')
        self.subject_trajectory_projection = nn.Linear(subject_dim, latent_dim)
        self.subject_volume_projection = nn.Linear(3, latent_dim)
        self.encoder = _FrozenSemanticEncoder(input_dim, latent_dim, nhead,
            num_encoder_layers, dim_feedforward, dropout_rate, num_query_tokens)
        self.memory_tokens_count = num_query_tokens
        self.num_query_tokens = num_query_tokens
        # Only temporal metadata is exposed under this historical access path.
        # There is no decoder module, generation method, or alternate model mode.
        self.decoder = SimpleNamespace(seq_length=int(seq_length))
        self.clip_embeddings = None
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def load_metric_checkpoint(self, checkpoint):
        state = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state, dict):
            raise ValueError('A frozen semantic checkpoint must contain a state_dict')
        selected, ignored = {}, []
        for name, value in state.items():
            while name.startswith(('model.', '_orig_mod.')):
                name = name.split('.', 1)[1]
            name = name.replace('subject_projection_loc_rot', 'subject_trajectory_projection')
            name = name.replace('subject_projection_vol', 'subject_volume_projection')
            if name.startswith(('encoder.', 'subject_trajectory_projection.', 'subject_volume_projection.')):
                if name in selected:
                    raise ValueError('Duplicate frozen evaluator checkpoint key: ' + name)
                selected[name] = value
            else:
                ignored.append(name)
        # Missing/extra encoder weights or any projection shape mismatch fail.
        self.load_state_dict(selected, strict=True)
        digest = hashlib.sha256()
        for name, value in sorted(selected.items()):
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.checkpoint_load_info = {'strict_encoder_and_projection_load': True,
            'loaded_tensor_count': len(selected), 'ignored_nonencoder_tensor_count': len(ignored),
            'encoder_projection_sha256': digest.hexdigest()}
        self.eval().requires_grad_(False)
        return self

    @torch.no_grad()
    def embed_trajectory(self, camera_trajectory, subject_trajectory, subject_volume,
                         src_key_mask=None, subject_key_padding_mask=None):
        device = next(self.parameters()).device
        if subject_trajectory is None:
            subject_trajectory = camera_trajectory.new_zeros(
                camera_trajectory.shape[0], camera_trajectory.shape[1],
                self.subject_trajectory_projection.in_features)
        if subject_volume is None:
            subject_volume = camera_trajectory.new_zeros(camera_trajectory.shape[0], 1, 3)
        subject_embedding = torch.cat([
            self.subject_trajectory_projection(subject_trajectory.to(device)),
            self.subject_volume_projection(subject_volume.to(device)),
        ], 1)
        memory = self.encoder(camera_trajectory.to(device), subject_embedding, src_key_mask,
            subject_key_padding_mask=(src_key_mask if subject_key_padding_mask is None
                                      else subject_key_padding_mask))
        return memory[:self.memory_tokens_count]


def _flatten_fields(struct, prefix=''):
    result = []
    for name, kind in struct:
        name = prefix + '_' + name if prefix else name
        result.extend(_flatten_fields(kind, name) if isinstance(kind, list) else [(name, kind)])
    return result


_METRIC_FIELDS = _flatten_fields(cinematography_struct) + _flatten_fields(simulation_struct)
if len(_METRIC_FIELDS) != 46:
    raise RuntimeError('The frozen semantic metric schema must retain its 46 historical fields')


def _metric_numeric_embedding(value, feature, width):
    # Exact historical Fourier representation; this is a frozen metric target,
    # never a generator input or a learned conditioning mode.
    bounded = min(max(value, feature.minimum), feature.maximum)
    interval = feature.maximum - feature.minimum
    if interval <= 0:
        raise ValueError('NumericFeature.maximum must be greater than minimum')
    shifted = bounded - feature.minimum
    normalized = math.log1p(shifted) / math.log1p(interval) if feature.logarithmic else shifted / interval
    frequency = torch.arange(1, (width + 1) // 2 + 1, dtype=torch.float32)
    phase = frequency * (math.pi * normalized)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1).flatten()[:width]


def _metric_rows(batch):
    high = batch.get('cinematography_prompt_parameters')
    if high is None:
        ids = batch.get('cinematography_ids')
        if not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[1] != 10:
            raise ValueError('Semantic metrics require raw cinematography metadata or explicit ten-field IDs')
        if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError('Metric category IDs must have a signed integer dtype')
        high = []
        for sample in ids.detach().cpu().tolist():
            row = []
            for (name, kind), index in zip(_METRIC_FIELDS[:10], sample):
                vocabulary = [value.value for value in kind]
                if name == 'movement_type':
                    vocabulary += ['nativePan', 'nativeOrbit']
                if index < -1 or index >= len(vocabulary):
                    raise ValueError('Metric category ID lies outside its field vocabulary')
                row.append((name, vocabulary[index] if index >= 0 else None, index, None))
            high.append(row)
    if not isinstance(high, (list, tuple)):
        raise TypeError('Metric parameter metadata must be a sequence of sample rows')
    low = batch.get('simulation_instruction_parameters')
    if low is not None and len(low) not in (0, len(high)):
        raise ValueError('Metric simulation metadata and cinematography batch sizes differ')
    rows = []
    for index, entries in enumerate(high):
        entries = list(entries)
        if len(entries) == 10 and low:
            entries += list(low[index])
        if len(entries) not in (10, 46):
            raise ValueError('Semantic metric metadata must contain ten film fields or all 46 fields')
        for (expected, _), entry in zip(_METRIC_FIELDS, entries):
            if not isinstance(entry, (list, tuple)) or len(entry) < 3 or entry[0] != expected:
                raise ValueError('Semantic metric metadata field order differs from the frozen schema')
        entries += [(name, None, -1, None) for name, _ in _METRIC_FIELDS[len(entries):]]
        rows.append(entries)
    return rows


def metric_conditioning_from_batch(batch, clip_embeddings, *, device=None):
    """Build frozen [46,B,512] metric targets separately from generator tokens.

    Raw observed metadata determines targets. Native classes outside the fixed
    metric vocabulary remain unavailable unless an explicit metric extension is
    supplied. No nearest-category remapping or reading categorical probabilities
    is performed. Returns independent boolean presence [B,46].
    """
    if not isinstance(clip_embeddings, dict) or not clip_embeddings:
        raise ValueError('The frozen semantic metric vocabulary is required')
    try:
        sample_vector = next(vector for family in clip_embeddings.values() for vector in family.values())
    except StopIteration:
        raise ValueError('The frozen semantic metric vocabulary has no vectors') from None
    width = torch.as_tensor(sample_vector).numel()
    rows = _metric_rows(batch)
    if device is None:
        for key in ('camera_trajectory', 'cinematography_ids', 'subject_trajectory'):
            if torch.is_tensor(batch.get(key)):
                device = batch[key].device
                break
    target = torch.zeros((46, len(rows), width), dtype=torch.float32, device=device)
    presence = torch.zeros((len(rows), 46), dtype=torch.bool, device=device)
    for sample, entries in enumerate(rows):
        for field, ((name, kind), entry) in enumerate(zip(_METRIC_FIELDS, entries)):
            raw, index = entry[1], entry[2]
            if torch.is_tensor(index):
                if index.ndim != 0 or index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                    raise TypeError('Metric metadata IDs must be signed integer scalars')
                index = index.item()
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError('Metric metadata IDs must be integers')
            if index < -1:
                raise ValueError('Only -1 denotes unknown metric metadata')
            if index < 0:
                continue
            if isinstance(kind, NumericFeature):
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(raw, bool) or not math.isfinite(value):
                    continue
                vector = _metric_numeric_embedding(value, kind, width)
            else:
                family = 'boolean' if kind is bool else kind.__name__
                values = clip_embeddings.get(family, {})
                if raw not in values:
                    # The frozen base movement vocabulary predates these two
                    # native classes. They cannot silently become other labels.
                    if name == 'movement_type' and raw in ('nativePan', 'nativeOrbit'):
                        continue
                    raise ValueError('Frozen metric vocabulary lacks observed target ' + name + '=' + str(raw))
                vector = torch.as_tensor(values[raw], dtype=torch.float32).flatten()
            if vector.shape != (width,) or not torch.isfinite(vector).all():
                raise ValueError('Invalid frozen semantic metric target vector')
            target[field, sample] = vector.to(device=target.device)
            presence[sample, field] = True
    for key, count in (('structured_presence_mask', 46), ('prompt_none_mask', 10)):
        mask = batch.get(key)
        if mask is None:
            continue
        # Full legacy masks remain useful when comparing frozen metric inputs;
        # new categorical data always supplies separate 10- and 46-field masks.
        if key == 'prompt_none_mask' and torch.is_tensor(mask) and mask.ndim == 2 and mask.shape[1] == 46:
            count = 46
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2
                or mask.shape[0] != len(rows) or mask.shape[1] < count):
            raise ValueError(key + ' has an invalid semantic metric presence shape')
        presence[:, :count] &= mask[:, :count].to(device=target.device)
    target = target.masked_fill(~presence.T[..., None], 0)
    return target, presence
