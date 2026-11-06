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

from data.simulation.constants import NumericFeature
from data.simulation.utils import CLIP_PARAMETERS, cinematography_struct_parameters


@dataclass
class CaptionMissingnessConfig:
    enabled: bool = True
    probability: float = 0.75
    min_keep_ratio: float = 0.15
    max_keep_ratio: float = 0.45
    endpoint_drop_probability: float = 0.75
    preserve_movement_type: bool = True
    fill: str = 'mean'
    mask_bank_path: str | None = None
    # Keep the historical sampler and random draws unless explicitly opted in.
    profile: str = 'alias_grouped'

    def __post_init__(self):
        for name in ('probability', 'min_keep_ratio', 'max_keep_ratio', 'endpoint_drop_probability'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        if self.min_keep_ratio > self.max_keep_ratio:
            raise ValueError('min_keep_ratio must not exceed max_keep_ratio')
        if self.fill not in ('mean', 'zero'):
            raise ValueError('caption missingness fill must be mean or zero')
        if self.profile not in ('alias_grouped', 'high_level_only'):
            raise ValueError('caption missingness profile must be alias_grouped or high_level_only')


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
    def __init__(self, config, clip_embeddings):
        self.config = CaptionMissingnessConfig(**dict(config))
        self.field_names = [name for name, _ in CLIP_PARAMETERS]
        self.high_level_count = len(cinematography_struct_parameters)
        self.alias_groups = self._alias_groups()
        self.mask_bank = None
        if self.config.mask_bank_path:
            payload = json.loads(Path(self.config.mask_bank_path).read_text())
            if payload.get('field_names') != self.field_names:
                raise ValueError('Caption mask bank field_names must match CLIP_PARAMETERS in order')
            rows = payload.get('presence_masks')
            if (not isinstance(rows, list) or not rows or any(
                not isinstance(row, list) or len(row) != len(self.field_names)
                or any(not isinstance(value, bool) for value in row)
                for row in rows
            )):
                raise ValueError('Caption mask bank presence_masks must be nonempty boolean rows')
            self.mask_bank = torch.tensor(rows, dtype=torch.bool)
            # A bank describes an upper bound on available information. Close
            # alias groups by removal, never by adding constraints absent from
            # the training annotation. Native ET captions are not modified.
            if self.config.profile == 'alias_grouped':
                for indices in self.alias_groups:
                    self.mask_bank[:, indices] = self.mask_bank[:, indices].all(dim=1, keepdim=True)
            else:
                # High-level descriptions need not supply a simulator program.
                # A bank remains an upper bound: do not restore movement/speed
                # or endpoint fields that its original row did not provide.
                self.mask_bank[:, self.high_level_count:] = False

        self.unknown_tokens = self._unknown_tokens(clip_embeddings)
        self.endpoint_indices = torch.tensor([
            i for i, name in enumerate(self.field_names)
            if name.startswith(('initial_', 'final_', 'setup_', 'dynamic_complementSetup_'))
        ], dtype=torch.long)
        self.movement_indices = next(
            indices for indices in self.alias_groups
            if self.field_names.index('movement_type') in indices
        )

    def _alias_groups(self):
        """Group overlapping high/low-level constraints before sampling.

        setup.kind can swap the initial/final meaning of setup/complementSetup.
        Masks alone cannot resolve that value, so endpoint aliases include both
        ends conservatively. This affects synthetic removal only.
        """
        groups = []
        for field in ('cameraAngle', 'shotSize', 'subjectView', 'subjectFraming'):
            low_field = 'subjectFraming_position' if field == 'subjectFraming' else field
            groups.append((
                f'initial_{field}', f'final_{field}',
                f'setup_config_{low_field}', f'dynamic_complementSetup_{low_field}',
            ))
        groups.extend((
            ('setup_config_subjectFraming_dutchAngleScale',
             'dynamic_complementSetup_subjectFraming_dutchAngleScale'),
            ('movement_type', 'dynamic_type', 'dynamic_direction', 'dynamic_movementMode'),
            ('movement_speed', 'dynamic_easing'),
        ))
        return [tuple(self.field_names.index(name) for name in group) for group in groups]

    def _unknown_tokens(self, clip_embeddings):
        if not clip_embeddings:
            raise ValueError('Caption missingness requires structured CLIP vocabulary embeddings')
        example = next(vector for values in clip_embeddings.values() for vector in values.values())
        width = torch.as_tensor(example).numel()
        rows = []
        for name, kind in CLIP_PARAMETERS:
            if self.config.fill == 'zero' or isinstance(kind, NumericFeature):
                rows.append(torch.zeros(width, dtype=torch.float32))
                continue
            key = 'boolean' if kind is bool else kind.__name__
            values = clip_embeddings.get(key)
            if not values:
                raise ValueError(f'Missing CLIP vocabulary for field {name}: {key}')
            rows.append(torch.stack([
                torch.as_tensor(vector, dtype=torch.float32).detach().cpu().flatten()
                for vector in values.values()
            ]).mean(dim=0))
        result = torch.stack(rows)
        if result.shape != (len(self.field_names), width) or not torch.isfinite(result).all():
            raise ValueError('Caption missingness fill embeddings have invalid shape or values')
        return result

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
        if cfg.profile == 'high_level_only':
            return self._sample_high_level(batch_size, generator, force)
        if self.mask_bank is not None:
            indices = torch.randint(len(self.mask_bank), (batch_size,), generator=generator)
            keep = self.mask_bank[indices].clone()
        else:
            ratio = cfg.min_keep_ratio + (cfg.max_keep_ratio - cfg.min_keep_ratio) * torch.rand(
                batch_size, 1, generator=generator,
            )
            keep = torch.rand(batch_size, len(self.field_names), generator=generator) < ratio
            # One draw per semantic constraint preserves the requested keep
            # ratio; independently sampling then intersecting aliases would
            # make four-way groups survive with probability ratio**4.
            for indices in self.alias_groups:
                keep[:, indices] = keep[:, indices[0], None].clone()
            # A sparse caption often omits both endpoint descriptions and their
            # simulation aliases together. Independent dropout misses this case.
            drop_endpoints = torch.rand(batch_size, 1, generator=generator) < cfg.endpoint_drop_probability
            keep[:, self.endpoint_indices] &= ~drop_endpoints
            if cfg.preserve_movement_type:
                keep[:, self.movement_indices] = True
        if not force:
            selected = torch.rand(batch_size, 1, generator=generator) < cfg.probability
            keep |= ~selected
        return keep

    def _sample_high_level(self, batch_size, generator, force):
        """Omit simulator instructions without removing their high-level facts.

        Endpoint fields are sampled independently. Observed movement speed is
        retained; preserve_movement_type controls retention of the movement
        type as in the original sampler. apply() intersects with input presence
        so this never reconstructs absent values or invalidated domain claims.
        """
        cfg = self.config
        if self.mask_bank is not None:
            indices = torch.randint(len(self.mask_bank), (batch_size,), generator=generator)
            keep = self.mask_bank[indices].clone()
        else:
            ratio = cfg.min_keep_ratio + (cfg.max_keep_ratio - cfg.min_keep_ratio) * torch.rand(
                batch_size, 1, generator=generator,
            )
            keep = torch.zeros(batch_size, len(self.field_names), dtype=torch.bool)
            keep[:, :self.high_level_count] = torch.rand(
                batch_size, self.high_level_count, generator=generator,
            ) < ratio
            drop_endpoints = torch.rand(batch_size, 1, generator=generator) < cfg.endpoint_drop_probability
            high_endpoints = self.endpoint_indices[self.endpoint_indices < self.high_level_count]
            keep[:, high_endpoints] &= ~drop_endpoints
            keep[:, self.field_names.index('movement_speed')] = True
            if cfg.preserve_movement_type:
                keep[:, self.field_names.index('movement_type')] = True
        if not force:
            selected = torch.rand(batch_size, 1, generator=generator) < cfg.probability
            keep |= ~selected
        return keep

    def apply(self, caption, presence_mask=None, *, sample_seeds=None, force=False):
        if caption.ndim != 3 or caption.shape[0] != len(self.field_names):
            raise ValueError('Caption missingness requires [structured fields, batch, embedding] tokens')
        if caption.shape[2] != self.unknown_tokens.shape[1]:
            raise ValueError('Caption width does not match the CLIP vocabulary')
        batch_size = caption.shape[1]
        if presence_mask is None:
            present = torch.ones(batch_size, caption.shape[0], dtype=torch.bool, device=caption.device)
        elif (not torch.is_tensor(presence_mask) or presence_mask.dtype != torch.bool
              or presence_mask.shape != (batch_size, caption.shape[0])):
            raise ValueError('Caption presence_mask must be boolean [batch, structured fields]')
        else:
            present = presence_mask.to(caption.device)
        keep = self.sample_keep_mask(batch_size, sample_seeds=sample_seeds, force=force).to(caption.device)
        # Existing missing fields retain their dataset-provided fill. Only
        # newly removed observed fields are changed; the input is never mutated.
        dropped = present & ~keep
        unknown = self.unknown_tokens.to(device=caption.device, dtype=caption.dtype)[:, None, :]
        return torch.where(dropped.T[..., None], unknown, caption), present & keep
