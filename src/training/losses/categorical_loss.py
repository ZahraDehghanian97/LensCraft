"""Categorical supervision and semantic cycle KL with unchanged geometry."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.categorical_latent import FIELD_COUNT, FIELD_NAMES, FIELD_WIDTHS, TOKEN_WIDTH
from .losses import (
    CameraTrajectoryLoss, KEYFRAME_LOSS_KEYS, OPTIONAL_POSITION_LOSS_KEYS,
    TRAJECTORY_LOSS_KEYS,
)


class CategoricalCameraTrajectoryLoss(CameraTrajectoryLoss):
    """Known-field CE plus KL(original.detach() || regenerated), with geometry.

    Each semantic term averages known samples within a field, then averages
    fields that have labels. The cycle additionally honors its supplied-field
    mask. Only each field's valid categories enter CE and KL.
    """

    def __init__(self, losses_list=None, rotation_weight=1.0, *, probability_epsilon=1e-8):
        requested = {"categorical": 1.0, "cycle": 1.0, **dict(losses_list or {})}
        allowed = set(TRAJECTORY_LOSS_KEYS + KEYFRAME_LOSS_KEYS + OPTIONAL_POSITION_LOSS_KEYS) | {"categorical", "cycle"}
        unknown = set(requested) - allowed
        if unknown:
            raise ValueError("Unknown categorical objective terms: " + ", ".join(sorted(unknown)))
        if not math.isfinite(probability_epsilon) or not 0 < probability_epsilon < 1:
            raise ValueError("probability_epsilon must lie strictly between zero and one")
        super().__init__(losses_list=requested, rotation_weight=rotation_weight)
        self.cycle_target_detach = True
        self.probability_epsilon = probability_epsilon

    @staticmethod
    def _check_semantic(values, name, batch_size=None):
        if (not torch.is_tensor(values) or not values.is_floating_point()
                or values.ndim != 3 or values.shape[0] != FIELD_COUNT
                or values.shape[2] != TOKEN_WIDTH):
            raise ValueError(f"{name} must be floating-point [10,batch,21]")
        if batch_size is not None and values.shape[1] != batch_size:
            raise ValueError(f"{name} batch size must match semantic predictions")

    @staticmethod
    def _presence_mask(mask, batch_size, device, name):
        if mask is None:
            return torch.ones((batch_size, FIELD_COUNT), dtype=torch.bool, device=device)
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2
                or mask.shape[0] != batch_size or mask.shape[1] < FIELD_COUNT):
            raise ValueError(f"{name} must be boolean [batch,10 or more fields]")
        return mask[:, :FIELD_COUNT].to(device=device)

    def _available_labels(self, target, batch):
        nonzero = torch.stack([
            target[field, :, :width].ne(0).any(dim=-1)
            for field, width in enumerate(FIELD_WIDTHS)
        ], dim=1)
        supplied = self._presence_mask(
            batch.get("prompt_none_mask"), target.shape[1], target.device, "prompt_none_mask",
        )
        return nonzero & supplied

    @staticmethod
    def _macro_average(field_losses, counts):
        losses = torch.stack(field_losses)
        counts = torch.stack(counts)
        active = counts > 0
        mean = losses.masked_fill(~active, 0).sum() / active.sum().clamp_min(1)
        return mean, {FIELD_NAMES[i]: value for i, value in enumerate(losses.unbind())}

    def _categorical_term(self, prediction, target, available, logits=None):
        if logits is not None:
            self._check_semantic(logits, "embedding_logits", prediction.shape[1])
            if logits.device != prediction.device:
                raise ValueError("embedding_logits and semantic predictions must share a device")
        dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
        fields, counts = [], []
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            for field, width in enumerate(FIELD_WIDTHS):
                known = available[:, field]
                if logits is None:
                    # Mask before taking logs, so unavailable fields cannot
                    # introduce NaN gradients. Padded categories are not read.
                    probabilities = prediction[field, :, :width].to(dtype)
                    probabilities = torch.where(known[:, None], probabilities, 1.0)
                    field_logits = probabilities.clamp_min(self.probability_epsilon).log()
                else:
                    field_logits = logits[field, :, :width].to(dtype)
                    field_logits = torch.where(known[:, None], field_logits, 0.0)
                indices = target[field, :, :width].argmax(dim=-1)
                rows = F.cross_entropy(field_logits, indices, reduction="none", label_smoothing=0.0)
                count = known.sum()
                fields.append(rows.masked_fill(~known, 0).sum() / count.clamp_min(1))
                counts.append(count)
        return self._macro_average(fields, counts)

    def _cycle_term(self, original, cycle, available):
        self._check_semantic(cycle, "cycle_embeddings", original.shape[1])
        if cycle.device != original.device:
            raise ValueError("Original and cycle semantic predictions must share a device")
        dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
        target = original.detach()
        fields, counts = [], []
        with torch.autocast(device_type=original.device.type, enabled=False):
            for field, width in enumerate(FIELD_WIDTHS):
                known = available[:, field]
                p = target[field, :, :width].to(dtype)
                q = cycle[field, :, :width].to(dtype)
                p = torch.where(known[:, None], p, 0.0)
                q = torch.where(known[:, None], q, 1.0)
                # Valid categories alone enter KL. Clamping BEFORE log keeps
                # exact one-hot zeros finite in both the value and backward.
                log_p = p.clamp_min(self.probability_epsilon).log()
                log_q = q.clamp_min(self.probability_epsilon).log()
                rows = (p * (log_p - log_q)).sum(dim=-1)
                count = known.sum()
                fields.append(rows.masked_fill(~known, 0).sum() / count.clamp_min(1))
                counts.append(count)
        return self._macro_average(fields, counts)

    def __call__(self, model_output, camera_trajectory, semantic_target, batch,
                 tgt_key_padding_mask=None, known_mask=None):
        return self.compute_total_loss(
            model_output["reconstructed_raw_matrix"],
            self._euler_traj_to_matrix(camera_trajectory),
            model_output["embeddings"], semantic_target, batch,
            model_output.get("cycle_embeddings"),
            tgt_key_padding_mask=tgt_key_padding_mask,
            projected_pred=model_output.get("reconstructed_rot_matrix"),
            known_mask=(model_output.get("known_mask") if known_mask is None else known_mask),
            cycle_prompt_mask=model_output.get("cycle_prompt_mask"),
            embedding_logits=model_output.get("embedding_logits"),
        )

    def compute_total_loss(
        self, trajectory_pred, trajectory_target, semantic_pred, semantic_target, batch,
        cycle_embeddings=None, tgt_key_padding_mask=None, projected_pred=None,
        known_mask=None, cycle_prompt_mask=None, *, embedding_logits=None,
    ):
        loss_dict = {}
        if self.losses_list.get("categorical", 0) or self.losses_list.get("cycle", 0):
            self._check_semantic(semantic_pred, "embeddings")
            self._check_semantic(semantic_target, "Categorical target", semantic_pred.shape[1])
            if semantic_target.device != semantic_pred.device:
                raise ValueError("Categorical targets and predictions must share a device")
            available = self._available_labels(semantic_target, batch)

            if self.losses_list.get("categorical", 0):
                loss_dict["categorical"], loss_dict["categorical_elements"] = self._categorical_term(
                    semantic_pred, semantic_target, available, embedding_logits,
                )
            if self.losses_list.get("cycle", 0):
                if cycle_embeddings is None:
                    # An empty view supplies a connected exact zero even if
                    # other entries contain padding infinities or NaNs.
                    loss_dict["cycle"] = semantic_pred[..., :0].sum()
                else:
                    cycle_available = available & self._presence_mask(
                        cycle_prompt_mask, semantic_pred.shape[1], semantic_pred.device, "cycle_prompt_mask",
                    )
                    loss_dict["cycle"], loss_dict["cycle_elements"] = self._cycle_term(
                        semantic_pred, cycle_embeddings, cycle_available,
                    )

        if any(self.losses_list.get(key, 0) for key in
               TRAJECTORY_LOSS_KEYS + KEYFRAME_LOSS_KEYS + OPTIONAL_POSITION_LOSS_KEYS):
            loss_dict.update(self.compute_trajectory_losses(
                trajectory_pred, trajectory_target,
                tgt_key_padding_mask=tgt_key_padding_mask,
                projected_pred=projected_pred,
                known_mask=known_mask,
            ))

        dtype = torch.float64 if trajectory_pred.dtype == torch.float64 else torch.float32
        total_loss = trajectory_pred.new_zeros((), dtype=dtype)
        for key, weight in self.losses_list.items():
            if weight and key in loss_dict:
                total_loss = total_loss + weight * loss_dict[key]
        loss_dict["total"] = total_loss.detach().item()
        return total_loss, loss_dict
