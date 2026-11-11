"""Training-only removal of structured fields with inference-compatible fills.

The clean caption remains the camera encoder's supervision target. This module
only changes the decoder's conditions; it never invents values for missing
fields or copies camera poses into a prompt. An optional mask bank must come
from training annotations, not the evaluation cohort.
"""

from dataclasses import dataclass
import json
import math
from numbers import Integral
from pathlib import Path

import torch

from models.categorical_latent import FIELD_NAMES, TOKEN_WIDTH, categorical_presence


@dataclass
class CaptionMissingnessConfig:
    enabled: bool = True
    probability: float = 0.75
    min_keep_ratio: float = 0.15
    max_keep_ratio: float = 0.45
    endpoint_drop_probability: float = 0.75
    speed_drop_probability: float = 0.0
    preserve_movement_type: bool = True
    fill: str = 'zero'
    mask_bank_path: str | None = None
    profile: str = 'independent'

    def __post_init__(self):
        for name in ('probability', 'min_keep_ratio', 'max_keep_ratio',
                     'endpoint_drop_probability', 'speed_drop_probability'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        if self.min_keep_ratio > self.max_keep_ratio:
            raise ValueError('min_keep_ratio must not exceed max_keep_ratio')
        if self.fill != 'zero':
            raise ValueError('Missing categorical fields use zero fill')
        if self.profile not in ('independent', 'high_level_only'):
            raise ValueError('caption missingness profile must be independent or high_level_only')


@dataclass
class CaptionMissingnessValidation:
    enabled: bool = True
    max_batches: int = 4
    seed: int = 31415
    score_weight: float = 1.0

    def __post_init__(self):
        if isinstance(self.max_batches, bool) or not isinstance(self.max_batches, Integral) or self.max_batches < 0:
            raise ValueError('missingness validation max_batches must be a nonnegative integer (0 means all batches)')
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**63:
            raise ValueError('missingness validation seed must be an integer in [0, 2**63)')
        if not math.isfinite(self.score_weight) or self.score_weight <= 0:
            raise ValueError('missingness validation score_weight must be finite and positive')


class CaptionMissingness:
    def __init__(self, config):
        self.config = CaptionMissingnessConfig(**dict(config))
        self.field_names = list(FIELD_NAMES)
        self.mask_bank = None
        if self.config.mask_bank_path:
            payload = json.loads(Path(self.config.mask_bank_path).read_text())
            if payload.get('field_names') != self.field_names:
                raise ValueError('Caption mask bank field_names must match the ten cinematography fields')
            rows = payload.get('presence_masks')
            if (not isinstance(rows, list) or not rows or any(
                not isinstance(row, list) or len(row) != len(self.field_names)
                or any(not isinstance(value, bool) for value in row)
                for row in rows
            )):
                raise ValueError('Caption mask bank presence_masks must be nonempty boolean rows')
            self.mask_bank = torch.tensor(rows, dtype=torch.bool)

        self.endpoint_indices = torch.tensor([
            i for i, name in enumerate(self.field_names)
            if name.startswith(('initial_', 'final_'))
        ], dtype=torch.long)
        self.movement_indices = [self.field_names.index('movement_type')]
        self.speed_indices = [self.field_names.index('movement_speed')]

    def sample_keep_mask(self, batch_size, *, sample_seeds=None, force=False):
        """Draw per-example masks; explicit seeds do not advance training RNG."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError('Caption missingness batch_size must be a positive integer')
        if sample_seeds is not None:
            if len(sample_seeds) != batch_size:
                raise ValueError('One caption missingness seed is required per sample')
            if any(isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63
                   for seed in sample_seeds):
                raise ValueError('Caption missingness seeds must be integers in [0, 2**63)')
            return torch.cat([
                self._sample(1, torch.Generator().manual_seed(int(seed)), force)
                for seed in sample_seeds
            ], dim=0)
        return self._sample(batch_size, None, force)

    def _sample(self, batch_size, generator, force):
        cfg = self.config
        if self.mask_bank is not None:
            indices = torch.randint(len(self.mask_bank), (batch_size,), generator=generator)
            keep = self.mask_bank[indices].clone()
        else:
            ratio = cfg.min_keep_ratio + (cfg.max_keep_ratio - cfg.min_keep_ratio) * torch.rand(
                batch_size, 1, generator=generator,
            )
            keep = torch.rand(batch_size, len(self.field_names), generator=generator) < ratio
            drop_endpoints = torch.rand(batch_size, 1, generator=generator) < cfg.endpoint_drop_probability
            keep[:, self.endpoint_indices] &= ~drop_endpoints
            if cfg.preserve_movement_type:
                keep[:, self.movement_indices] = True
        if self.mask_bank is None and cfg.profile == 'high_level_only':
            keep[:, self.speed_indices] = True
        if not force:
            selected = torch.rand(batch_size, 1, generator=generator) < cfg.probability
            keep |= ~selected
        return self._drop_speed(keep, generator)

    def _drop_speed(self, keep, generator):
        # Speed omission is independent of the sparse-caption selection.
        if self.config.speed_drop_probability > 0:
            drop = torch.rand(len(keep), 1, generator=generator) < self.config.speed_drop_probability
            keep[:, self.speed_indices] &= ~drop
        return keep

    def apply(self, caption, presence_mask=None, *, sample_seeds=None, force=False):
        if caption.ndim != 3 or caption.shape[0] != len(self.field_names):
            raise ValueError('Caption missingness requires [structured fields, batch, embedding] tokens')
        if caption.shape[2] != TOKEN_WIDTH:
            raise ValueError('Categorical caption width must be 21')
        batch_size = caption.shape[1]
        if presence_mask is None:
            present = categorical_presence(caption)
        elif (not torch.is_tensor(presence_mask) or presence_mask.dtype != torch.bool
              or presence_mask.shape != (batch_size, caption.shape[0])):
            raise ValueError('Caption presence_mask must be boolean [batch, structured fields]')
        else:
            present = presence_mask.to(caption.device)
        keep = self.sample_keep_mask(batch_size, sample_seeds=sample_seeds, force=force).to(caption.device)
        present = present & categorical_presence(caption)
        retained = present & keep
        return caption.masked_fill(~retained.T[..., None], 0), retained
