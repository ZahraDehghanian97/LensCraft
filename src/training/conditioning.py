"""Explicit, persistent training modes matching the evaluation contract."""

from dataclasses import dataclass, field
import math
from numbers import Integral

import torch

from data.sim_format import MEMORY_TEACHER_FORCING_BY_MODE, build_keyframing_mask


TRAINING_MODES = ('prompt_generation', 'reconstruction', 'key_framing', 'key_framing+prompt')


def validate_keyframe_counts(counts):
    counts = tuple(counts)
    if (not counts or len(set(counts)) != len(counts) or any(
        isinstance(k, bool) or not isinstance(k, Integral) or k < 1 for k in counts
    )):
        raise ValueError('keyframe_counts must contain distinct positive integers')
    return counts


@dataclass
class ConditioningPolicy:
    enabled: bool = True
    mode_weights: dict = field(default_factory=lambda: dict.fromkeys(TRAINING_MODES, 1.0))
    keyframe_counts: tuple = (1, 2, 4, 8, 26)

    def __post_init__(self):
        if set(self.mode_weights) != set(TRAINING_MODES):
            raise ValueError(f'mode_weights must specify all modes: {TRAINING_MODES}')
        self.weights = torch.tensor([float(self.mode_weights[m]) for m in TRAINING_MODES], dtype=torch.float64)
        if not torch.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError('Every conditioning mode must have a finite positive weight')
        self.keyframe_counts = validate_keyframe_counts(self.keyframe_counts)

    def sample(self):
        mode = TRAINING_MODES[torch.multinomial(self.weights, 1).item()]
        count = None
        if mode.startswith('key_framing'):
            count = self.keyframe_counts[torch.randint(len(self.keyframe_counts), ()).item()]
        return mode, count


@dataclass
class ConditioningValidation:
    enabled: bool = True
    keyframe_counts: tuple = (1, 2, 4, 8, 26)
    max_batches: int = 4
    seed: int = 42
    known_pose_weight: float = 1.0

    def __post_init__(self):
        self.keyframe_counts = validate_keyframe_counts(self.keyframe_counts)
        if isinstance(self.max_batches, bool) or not isinstance(self.max_batches, Integral) or self.max_batches < 1:
            raise ValueError('validation max_batches must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**63:
            raise ValueError('validation seed must be an integer in [0, 2**63)')
        if not math.isfinite(self.known_pose_weight) or self.known_pose_weight < 0:
            raise ValueError('known_pose_weight must be finite and nonnegative')

    def cases(self):
        yield 'prompt_generation', 'prompt_generation', None
        yield 'reconstruction', 'reconstruction', None
        for mode in ('key_framing', 'key_framing+prompt'):
            for count in self.keyframe_counts:
                yield f'{mode}_k{count}', mode, count


def prepare_conditioning(camera, padding_mask, mode, keyframe_count=None, sample_seeds=None):
    """Return clean source, hidden-source mask, supplied-pose mask and text ratio."""
    if mode not in TRAINING_MODES:
        raise ValueError(f'Unknown conditioning mode: {mode}')
    padding = torch.zeros(camera.shape[:2], dtype=torch.bool, device=camera.device) if padding_mask is None else padding_mask
    if padding.dtype != torch.bool or padding.shape != camera.shape[:2]:
        raise ValueError('padding_mask must be boolean [batch, frames]')
    if mode.startswith('key_framing'):
        if keyframe_count is None:
            raise ValueError('Keyframe conditioning requires a keyframe_count')
        source_mask = build_keyframing_mask(len(camera), camera.device, camera.shape[1],
            num_keyframes=keyframe_count, padding_mask=padding, sample_seeds=sample_seeds)
    else:
        if keyframe_count is not None:
            raise ValueError('Only keyframe modes accept keyframe_count')
        source_mask = padding
    # Prompt retains the clean camera for encoder alignment supervision only;
    # its decoder uses alpha=1 and receives no known poses.
    known = torch.zeros_like(padding) if mode == 'prompt_generation' else ~source_mask
    return camera.masked_fill(source_mask[..., None], 0), source_mask, known, MEMORY_TEACHER_FORCING_BY_MODE[mode]
