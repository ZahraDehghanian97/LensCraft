"""Observed geometry, kept separate from the categorical semantic bottleneck."""
from __future__ import annotations

import torch
from torch import nn


class GeometricContext(nn.Module):
    """Encode calibrated optics and distinguish unknown subject features.

    Intrinsics are focal length in millimetres and image width/height. NaN
    denotes an unobserved component; positive finite values are measurements.
    Availability masks describe observations, independently of dataset names.
    """

    def __init__(self, width):
        super().__init__()
        self.optics_projection = nn.Linear(4, width, bias=False)
        self.unknown_orientation = nn.Parameter(torch.randn(width) / width ** 0.5)
        self.unknown_dimensions = nn.Parameter(torch.randn(width) / width ** 0.5)

    @staticmethod
    def availability(value, shape, device, name, *, default=True):
        if value is None:
            return torch.full(shape, default, dtype=torch.bool, device=device)
        if not torch.is_tensor(value) or value.dtype != torch.bool:
            raise ValueError(f'{name} must be a boolean tensor')
        value = value.to(device=device)
        if value.shape == (shape[0],):
            value = value[:, None]
        if value.shape not in (shape, (shape[0], 1)):
            raise ValueError(f'{name} must have shape {shape} or [{shape[0]}]')
        return value.expand(shape)

    def forward(self, subject, volume, intrinsics=None, orientation_available=None,
                dimensions_available=None, padding_mask=None):
        if subject.ndim != 3 or subject.shape[-1] != 6:
            raise ValueError('subject_trajectory must have shape [batch, frames, 6]')
        batch, frames = subject.shape[:2]
        if volume.shape == (batch, 3):
            volume = volume[:, None, :]
        if volume.shape != (batch, 1, 3):
            raise ValueError('subject_volume must have shape [batch, 1, 3]')
        shape = (batch, frames)
        orientation = self.availability(orientation_available, shape, subject.device,
                                        'subject_orientation_available')
        dimensions = self.availability(dimensions_available, (batch, 1), subject.device,
                                       'subject_dimensions_available')
        if padding_mask is None:
            padding_mask = torch.zeros(shape, device=subject.device, dtype=torch.bool)
        elif (not torch.is_tensor(padding_mask) or padding_mask.dtype != torch.bool
              or padding_mask.shape != shape):
            raise ValueError('padding_mask must be boolean [batch, frames]')
        padding_mask = padding_mask.to(device=subject.device)
        # Hidden placeholders may be NaN and must never reach a projection.
        positions = subject[..., :3].masked_fill(padding_mask[..., None], 0)
        angles = subject[..., 3:].masked_fill((~orientation | padding_mask)[..., None], 0)
        subject = torch.cat((positions, angles), dim=-1)
        volume = volume.masked_fill(~dimensions[..., None], 0)
        if not torch.isfinite(subject).all() or not torch.isfinite(volume).all():
            raise ValueError('Observed subject features must contain only finite values')

        optics = subject.new_zeros(batch, frames, 2)
        known = torch.zeros(batch, frames, 2, device=subject.device, dtype=torch.bool)
        if intrinsics is not None:
            if not torch.is_tensor(intrinsics) or not intrinsics.is_floating_point():
                raise ValueError('camera_intrinsics must be a floating-point tensor')
            intrinsics = intrinsics.to(device=subject.device)
            if intrinsics.shape == (batch, 2):
                intrinsics = intrinsics[:, None, :]
            if intrinsics.shape not in ((batch, frames, 2), (batch, 1, 2)):
                raise ValueError('camera_intrinsics must have shape [batch, frames, 2] or [batch, 2]')
            intrinsics = intrinsics.expand(batch, frames, 2)
            active = ~padding_mask[..., None]
            if ((torch.isinf(intrinsics) | (intrinsics <= 0)) & active).any():
                raise ValueError('Observed camera_intrinsics must be finite and positive; use NaN for unknown')
            known = torch.isfinite(intrinsics) & active
            # A fixed numerical reference keeps the logarithm dimensionless;
            # it does not supply a lens value when calibration is missing.
            reference = intrinsics.new_tensor((50., 1.))
            safe = torch.where(known, intrinsics, reference)
            optics = torch.log(safe.float() / reference.float()).to(subject.dtype)
            optics = optics.masked_fill(~known, 0)
        features = torch.cat((optics, known.to(subject.dtype)), dim=-1)
        optics_embedding = self.optics_projection(features)
        orientation_embedding = (~orientation & ~padding_mask)[..., None].to(subject.dtype) * self.unknown_orientation
        dimensions_embedding = (~dimensions)[..., None].to(volume.dtype) * self.unknown_dimensions
        return subject, volume, optics_embedding + orientation_embedding, dimensions_embedding
