"""Checkpoint-compatible volume removal for the scheduled ablation study.

Every variant uses this class. A disabled volume leaves one constant token in
the existing mask layout, but no volume dimensions reach any model pathway.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.camera_trajectory_model import LensCraft


class VolumeProjection(nn.Linear):
    @classmethod
    def from_linear(cls, original, *, enabled):
        # Register the existing parameters without running Linear.reset_parameters:
        # removing volume must not change initialization or consume extra RNG.
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features = original.in_features
        result.out_features = original.out_features
        result.register_parameter("weight", original.weight)
        result.register_parameter("bias", original.bias)
        result.enabled = enabled
        result.train(original.training)
        return result

    def forward(self, inputs):
        if not self.enabled:
            inputs = torch.zeros_like(inputs)
        return F.linear(inputs, self.weight, self.bias)


class AblationLensCraft(LensCraft):
    def __init__(self, *args, use_subject_volume=True, **kwargs):
        if not isinstance(use_subject_volume, bool):
            raise TypeError("use_subject_volume must be boolean")
        super().__init__(*args, **kwargs)
        self.use_subject_volume = use_subject_volume
        self.subject_volume_projection = VolumeProjection.from_linear(
            self.subject_volume_projection, enabled=use_subject_volume,
        )
