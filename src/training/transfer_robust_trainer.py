"""Simulation-only adaptation to incomplete and subject-relative inputs.

Native ET/CCDM test data never enter this trainer. Proxy validation applies
fixed transforms to the simulation validation split and scores all four
conditioning families, alongside the clean/sparse simulation objective.
The additional ``val_transfer_prompt_score`` gives equal weight to clean,
sparse, ET-shaped and CCDM-shaped SIM prompt generation. Explicitly selecting
that monitor leaves the legacy all-family objective available as a diagnostic.
``val_ccdm_prompt_score`` selects only the held-out CCDM-shaped simulation
prompt view; the legacy aggregate scores retain their existing definitions.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import torch

from training.autoencoder_trainer import LightningLensCraft
from training.conditioning import TRAINING_MODES
from training.conditioning_metrics import METRIC_NAMES, conditioning_pose_score, pose_metric_totals
from training.domain_augmentation import DomainAugmentation


@dataclass
class DomainValidation:
    enabled: bool = True
    max_batches: int = 4
    seed: int = 27182
    keyframe_count: int = 4
    score_weight: float = 1.0

    def __post_init__(self):
        for name in ('max_batches', 'keyframe_count'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'Domain validation {name} must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**62:
            raise ValueError('Domain validation seed must be an integer in [0, 2**62)')
        if not math.isfinite(self.score_weight) or self.score_weight <= 0:
            raise ValueError('Domain validation score_weight must be finite and positive')


class TransferRobustTrainer(LightningLensCraft):
    def __init__(self, *args, domain_augmentation=None,
                 validation_domain_augmentation=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dataset_mode != 'simulation':
            raise ValueError('Transfer robustness uses only the simulation train/validation splits')
        if not self._validate_conditioning or not self._validate_caption_missingness:
            raise ValueError('Transfer robustness requires clean conditioning and sparse caption validation')
        if not getattr(self.model, 'caption_presence_conditioning', False):
            raise ValueError('Transfer robustness requires caption_presence_conditioning=true')
        if self.loss_module.losses_list.get('contrastive', 0):
            raise ValueError('Domain augmentation does not support metadata-based contrastive supervision')
        if not domain_augmentation or not domain_augmentation.get('enabled', True):
            raise ValueError('Transfer robustness requires enabled domain_augmentation')
        self.domain_augmentation = DomainAugmentation(
            domain_augmentation, self.loss_module.clip_embeddings,
        )
        self.validation_domain_augmentation = DomainValidation(
            **dict(validation_domain_augmentation or {}),
        )
        if not self.validation_domain_augmentation.enabled:
            raise ValueError('Transfer robustness requires domain validation for checkpoint selection')
        self._transfer_base_score = None
        self._transfer_clean_prompt_score = None
        self._transfer_sparse_prompt_score = None

    def log(self, name, value, *args, **kwargs):
        if name == 'val_robustness_score':
            self._transfer_base_score = value.detach()
        elif name == 'val_conditioning/prompt_generation/score':
            self._transfer_clean_prompt_score = value.detach()
        elif name == 'val_caption_missingness_score':
            self._transfer_sparse_prompt_score = value.detach()
        return super().log(name, value, *args, **kwargs)

    def _step(self, batch, batch_idx, stage):
        if stage == 'train':
            batch = self.domain_augmentation.apply(batch)
            views = batch['domain_augmentation_view']
            for index, name in enumerate(('original', 'et', 'ccdm')):
                self.log(f'train_domain_fraction/{name}', (views == index).float().mean(),
                         on_step=False, on_epoch=True, batch_size=len(views))
            ccdm_rows = views == 2
            presence = batch.get('prompt_none_mask')
            if presence is not None:
                movement = self.domain_augmentation.field_names.index('movement_type')
                count = ccdm_rows.sum()
                observed = (ccdm_rows & presence[:, movement]).sum()
                self.log('train_ccdm_motion_present_fraction', observed.float() / count.clamp_min(1),
                         on_step=False, on_epoch=True, batch_size=max(int(count), 1))
                self.log('train_ccdm_motion_sample_count', count,
                         on_step=False, on_epoch=True, reduce_fx='sum', batch_size=len(views))
                self.log('train_ccdm_motion_present_count', observed,
                         on_step=False, on_epoch=True, reduce_fx='sum', batch_size=len(views))
        loss = super()._step(batch, batch_idx, stage)
        if stage == 'val':
            self._validate_domain_batch(batch, batch_idx)
        return loss

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._transfer_base_score = None
        self._transfer_clean_prompt_score = None
        self._transfer_sparse_prompt_score = None
        self._domain_sample_offset = 0
        self._domain_totals = {
            (domain, mode): {
                key: torch.zeros(2, dtype=torch.float64, device=self.device)
                for key in METRIC_NAMES
            }
            for domain in ('et', 'ccdm') for mode in TRAINING_MODES
        }

    def _validate_domain_batch(self, batch, batch_idx):
        cfg = self.validation_domain_augmentation
        if batch_idx >= cfg.max_batches:
            return
        size = len(batch['camera_trajectory'])
        seeds = [cfg.seed + self._domain_sample_offset + i for i in range(size)]
        self._domain_sample_offset += size
        for domain in ('et', 'ccdm'):
            view = self.domain_augmentation.apply(batch, sample_seeds=seeds, force_domain=domain)
            camera = view['camera_trajectory']
            padding = view.get('padding_mask')
            caption = self._prepare_clip_embeddings(view)
            for mode in TRAINING_MODES:
                output = self._forward_step(
                    camera, view['subject_trajectory'], view['subject_volume'], caption, padding,
                    is_training=False, decode_mode=self.decode_mode,
                    conditioning_mode=mode,
                    keyframe_count=cfg.keyframe_count if mode.startswith('key_framing') else None,
                    keyframe_sample_seeds=seeds,
                    caption_presence_mask=view.get('prompt_none_mask'),
                    caption_missingness_seeds=seeds,
                    force_caption_missingness=True,
                )
                totals = pose_metric_totals(output['reconstructed'], camera, padding, output['known_mask'])
                for key, pair in totals.items():
                    self._domain_totals[domain, mode][key] += pair

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self._transfer_base_score is None:
            raise RuntimeError('Clean/sparse simulation validation did not produce a selection score')
        scores = []
        prompt_scores = {
            'clean_sim': self._transfer_clean_prompt_score,
            'sparse_sim': self._transfer_sparse_prompt_score,
        }
        for domain in ('et', 'ccdm'):
            mode_scores = []
            for mode in TRAINING_MODES:
                means = self._log_pose_metric_means(
                    f'val_domain/{domain}/{mode}', self._domain_totals[domain, mode],
                    log_counts=False,
                )
                if 'position_error_normalized' not in means:
                    raise RuntimeError(f'No valid proxy validation poses for {domain}/{mode}')
                score = conditioning_pose_score(means, self.validation_conditioning.motion_score_weight)
                if mode.startswith('key_framing'):
                    score = score + self.validation_conditioning.known_pose_weight * (
                        means['known_position_error_normalized'] + means['known_rotation_error_deg'] / 180.
                    )
                mode_scores.append(score)
                self.log(f'val_domain/{domain}/{mode}/score', score,
                         on_step=False, on_epoch=True, sync_dist=False)
                if mode == 'prompt_generation':
                    prompt_scores[f'{domain}_proxy'] = score
            domain_score = torch.stack(mode_scores).mean()
            if not torch.isfinite(domain_score):
                raise RuntimeError(f'Nonfinite {domain} proxy validation score')
            scores.append(domain_score)
            self.log(f'val_domain/{domain}/score', domain_score,
                     on_step=False, on_epoch=True, sync_dist=False)
        weight = self.validation_domain_augmentation.score_weight
        score = (self._transfer_base_score + weight * torch.stack(scores).sum()) / (1 + 2 * weight)
        if not torch.isfinite(score):
            raise RuntimeError('Nonfinite transfer validation score')
        self.log('val_transfer_score', score, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=False)
        # No reference-conditioned result or configurable legacy family weight
        # contributes to this objective. Each view uses the same pose/motion
        # score and only held-out simulation examples; lower is better.
        for name, component in prompt_scores.items():
            if component is None:
                raise RuntimeError(f'No {name} prompt validation score')
            if not torch.isfinite(component):
                raise RuntimeError(f'Nonfinite {name} prompt validation score')
            self.log(f'val_transfer_prompt/{name}_score', component,
                     on_step=False, on_epoch=True, sync_dist=False)
        prompt_score = torch.stack(list(prompt_scores.values())).mean()
        if not torch.isfinite(prompt_score):
            raise RuntimeError('Nonfinite prompt transfer validation score')
        self.log('val_transfer_prompt_score', prompt_score, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=False)
        self.log('val_ccdm_prompt_score', prompt_scores['ccdm_proxy'],
                 on_step=False, on_epoch=True, sync_dist=False)
        count = torch.tensor(self._domain_sample_offset, device=self.device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
        self.log('val_domain_sample_count', count, on_step=False, on_epoch=True, sync_dist=False)
