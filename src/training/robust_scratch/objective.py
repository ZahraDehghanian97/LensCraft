"""Opt-in experimental robust training component; full-run validation is pending.

Loss-only semantic anchoring for the isolated random-initialized run.

Generation runs before the clean-camera supervision branch is evaluated.  The
branch therefore cannot supply target poses or embeddings to the decoder.
Unlike the preceding continuation, cycle encoding sees every generated valid
frame and its target cannot move to meet the generated encoding.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from training.base_trainer import BaseTrainer


POLICY = 'clean-target-encoder-and-detached-caption-cycle-v1'
CAPTION_MODES = frozenset(('prompt_generation', 'key_framing+prompt'))
CAMERA_MODES = frozenset(('reconstruction', 'key_framing'))


def _temporal_padding(camera, padding):
    if padding is None:
        return None
    if (not torch.is_tensor(padding) or padding.dtype != torch.bool
            or padding.shape != camera.shape[:2]):
        raise ValueError('Temporal padding must be boolean [batch, frames]')
    return padding.to(device=camera.device)


def _without_padding(camera, padding):
    return camera if padding is None else camera.masked_fill(padding[..., None], 0)


class RobustForward(BaseTrainer):
    """Use the frozen generation path, then construct supervision-only tensors."""

    def _calculate_schedule_parameters(self) -> Dict[str, float]:
        return {
            'noise_std': 0.0, 'mask_ratio': 0.0, 'memory_mask_ratio': 0.0,
            'memory_teacher_forcing_ratio': 1.0,
            'trajectory_teacher_forcing_ratio': 0.0,
        }

    def _forward_step(
        self,
        camera_trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor,
        subject_volume: torch.Tensor,
        caption_embedding: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor],
        is_training: bool = False,
        decode_mode: str = 'single_step',
        compute_cycle_embeddings: bool = False,
        conditioning_mode: Optional[str] = None,
        keyframe_count: Optional[int] = None,
        keyframe_sample_seeds=None,
        caption_presence_mask: Optional[torch.Tensor] = None,
        caption_missingness_seeds=None,
        force_caption_missingness: bool = False,
    ) -> Dict[str, torch.Tensor]:
        padding = _temporal_padding(camera_trajectory, tgt_key_padding_mask)
        if (is_training and conditioning_mode is None
                and self.conditioning is not None and self.conditioning.enabled):
            # Make the chosen mode explicit before constructing source input.
            # This consumes exactly the draw the parent would otherwise make.
            conditioning_mode, keyframe_count = self.conditioning.sample()
        generator_camera = (torch.zeros_like(camera_trajectory)
                            if conditioning_mode == 'prompt_generation'
                            else camera_trajectory)
        # Parent owns the established conditioning and missingness contracts.
        # In particular, no clean-target branch runs until generation is done.
        output = super()._forward_step(
            generator_camera, subject_trajectory, subject_volume,
            caption_embedding, padding, is_training=is_training,
            decode_mode=decode_mode, compute_cycle_embeddings=False,
            conditioning_mode=conditioning_mode, keyframe_count=keyframe_count,
            keyframe_sample_seeds=keyframe_sample_seeds,
            caption_presence_mask=caption_presence_mask,
            caption_missingness_seeds=caption_missingness_seeds,
            force_caption_missingness=force_caption_missingness,
        )
        if is_training:
            clean_camera = _without_padding(camera_trajectory.detach(), padding)
            output['embeddings'] = self.model.encoder(
                clean_camera, output['subject_embedding'], padding,
                subject_key_padding_mask=padding,
            )

        if compute_cycle_embeddings:
            generated = _without_padding(output['reconstructed'], padding)
            output['cycle_embeddings'] = self.model.encoder(
                generated, output['subject_embedding'], padding,
                subject_key_padding_mask=padding,
            )
            mode = output.get('conditioning_mode', conditioning_mode or 'scheduled')
            if mode in CAPTION_MODES:
                caption_conditioned = True
            elif mode in CAMERA_MODES:
                caption_conditioned = False
            elif mode == 'scheduled':
                caption_conditioned = (
                    caption_embedding is not None
                    and self._calculate_schedule_parameters()['memory_teacher_forcing_ratio'] > 0
                )
            else:
                raise ValueError('Unknown conditioning mode for anchored cycle: ' + str(mode))
            if caption_conditioned:
                if caption_embedding is None:
                    raise ValueError('Caption-conditioned cycle requires supplied caption embeddings')
                output['anchored_cycle_target'] = caption_embedding.detach()
                output['anchored_cycle_mask'] = output.get('cycle_prompt_mask')
                output['cycle_target_kind'] = 'supplied_caption'
            else:
                output['anchored_cycle_target'] = output['embeddings'].detach()
                output['anchored_cycle_mask'] = None
                output['cycle_target_kind'] = 'detached_clean_camera'
        return output


def anchored_cycle_loss(prediction, target, mask=None, *, encoder_loss_function='clip'):
    """Match known fields only; unknown values have exactly zero gradient.

    Reduction matches the existing ClipLoss: average each field over examples
    where it is supplied, then average over all structured fields.  Values in
    unavailable fields are sanitized *before* cosine/MSE, including NaN fills.
    Targets are detached defensively, even if a caller bypasses RobustForward.
    """
    if (not torch.is_tensor(prediction) or not torch.is_tensor(target)
            or prediction.ndim != 3 or target.shape != prediction.shape):
        raise ValueError('Cycle prediction and target must match [fields,batch,width]')
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError('Cycle embeddings must be floating point')
    fields, batch, _ = prediction.shape
    if mask is None:
        valid = torch.ones((fields, batch), dtype=torch.bool, device=prediction.device)
    else:
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                or mask.shape != (batch, fields)):
            raise ValueError('Cycle presence mask must be boolean [batch,fields]')
        valid = mask.to(device=prediction.device).transpose(0, 1)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
        pred = torch.where(valid[..., None], prediction.to(dtype=dtype), 0.0)
        truth = torch.where(
            valid[..., None], target.detach().to(device=prediction.device, dtype=dtype), 0.0,
        )
        if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
            raise FloatingPointError('Known cycle embeddings contain nonfinite values')
        if encoder_loss_function == 'clip':
            values = 1.0 - torch.nn.functional.cosine_similarity(pred, truth, dim=-1)
        elif encoder_loss_function == 'mse':
            values = (pred - truth).square().mean(dim=-1)
        else:
            raise ValueError('Unsupported encoder loss: ' + str(encoder_loss_function))
        counts = valid.sum(dim=1)
        per_field = values.masked_fill(~valid, 0.0).sum(dim=1) / counts.clamp_min(1)
        per_field = torch.where(counts > 0, per_field, torch.zeros_like(per_field))
        total = per_field.mean() if fields else pred.sum() * 0.0
    return total, list(per_field.unbind())


class RobustObjective:
    """Preserve the frozen objective except for the explicitly anchored cycle."""

    def __init__(self, base_loss):
        self.base_loss = base_loss
        self.losses_list = dict(base_loss.losses_list)
        if getattr(base_loss, 'weighted_clip_loss', False):
            raise ValueError('This recipe requires the frozen unweighted structured CLIP loss')

    def __getattr__(self, name):
        return getattr(self.base_loss, name)

    def __call__(self, model_output, camera_trajectory, clip_target, batch,
                 tgt_key_padding_mask=None, known_mask=None):
        # Leave base weights and caller tensors untouched.  Removing the cycle
        # tensor makes the inherited cycle exactly zero, avoiding double count.
        base_output = dict(model_output)
        base_output.pop('cycle_embeddings', None)
        kwargs = {} if known_mask is None else {'known_mask': known_mask}
        total, original = self.base_loss(
            base_output, camera_trajectory, clip_target, batch,
            tgt_key_padding_mask, **kwargs,
        )
        losses = dict(original)
        cycle_weight = float(self.losses_list.get('cycle', 0.0))
        if cycle_weight:
            if model_output.get('cycle_embeddings') is None:
                cycle = total.new_zeros(())
                elements = []
            else:
                if model_output.get('anchored_cycle_target') is None:
                    raise ValueError('Generated cycle embeddings require an explicitly anchored target')
                cycle, elements = anchored_cycle_loss(
                    model_output['cycle_embeddings'], model_output['anchored_cycle_target'],
                    model_output.get('anchored_cycle_mask'),
                    encoder_loss_function=getattr(self.base_loss, 'encoder_loss_function', 'clip'),
                )
            total = total + cycle_weight * cycle
            losses['cycle'] = cycle
            losses['cycle_elements'] = dict(enumerate(elements))
        if not torch.isfinite(total):
            raise FloatingPointError('Robust objective is nonfinite')
        losses['total'] = total.detach().item()
        return total, losses
