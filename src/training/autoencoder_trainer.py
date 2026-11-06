from typing import Optional, Dict, Any
import torch

from training.base_trainer import BaseTrainer, NoiseConfig, MaskConfig, TeacherForcingConfig
from data.convertor.convertor import convert_to_target
from data.simulation.utils import structured_conditioning_from_batch
from training.conditioning import ConditioningValidation, TRAINING_MODES
from training.conditioning_metrics import (
    METRIC_NAMES, conditioning_pose_score, deterministic_caption_shuffle,
    pose_metric_totals, rotation_baseline_totals,
)
from training.caption_missingness import CaptionMissingnessValidation


class LightningLensCraft(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_module: torch.nn.Module,
        noise: NoiseConfig,
        mask: MaskConfig,
        teacher_forcing_schedule: TeacherForcingConfig,
        compile_mode: str = "default",
        compile_enabled: bool = True,
        dataset_mode: str = 'simulation',
        use_merged_memory: bool = True,
        decode_mode: str = 'single_step',
        use_cycle_consistency: bool = False,
        conditioning: Optional[Dict[str, Any]] = None,
        validation_conditioning: Optional[Dict[str, Any]] = None,
        caption_missingness: Optional[Dict[str, Any]] = None,
        validation_caption_missingness: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model,
            optimizer,
            lr_scheduler,
            loss_module,
            noise,
            mask,
            teacher_forcing_schedule,
            compile_mode,
            compile_enabled,
            use_merged_memory,
            conditioning=conditioning,
            caption_missingness=caption_missingness,
        )
        self.dataset_mode = dataset_mode
        self.decode_mode = decode_mode
        self.use_cycle_consistency = use_cycle_consistency
        self.validation_conditioning = (
            ConditioningValidation(**dict(validation_conditioning))
            if validation_conditioning is not None else None
        )
        self.validation_caption_missingness = (
            CaptionMissingnessValidation(**dict(validation_caption_missingness))
            if validation_caption_missingness is not None else None
        )
        if self._validate_caption_missingness and self.caption_missingness is None:
            raise ValueError('Sparse caption validation requires enabled caption_missingness')

    def _prepare_clip_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        caption = structured_conditioning_from_batch(batch)
        if caption is None:
            caption = batch['caption_feat']
            if caption.dim() == 2:
                caption = caption.unsqueeze(0)
        return caption

    def _convert_to_simulation(
        self, batch: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        target_seq_len = self.model.decoder.seq_length

        camera_trajectory, subject_trajectory, subject_volume, padding_mask = convert_to_target(
            source,
            "simulation",
            batch["camera_trajectory"],
            batch["subject_trajectory"],
            batch.get("subject_volume"),
            batch.get("padding_mask"),
            target_len=target_seq_len,
        )

        new_batch = dict(batch)
        new_batch["camera_trajectory"] = camera_trajectory
        new_batch["subject_trajectory"] = subject_trajectory
        new_batch["subject_volume"] = subject_volume
        new_batch["padding_mask"] = padding_mask
        return new_batch

    def _step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        if self.dataset_mode in ('et', 'ccdm'):
            batch = self._convert_to_simulation(batch, self.dataset_mode)

        camera_trajectory = batch['camera_trajectory']
        subject_trajectory = batch['subject_trajectory']
        subject_volume = batch['subject_volume']
        tgt_key_padding_mask = batch.get("padding_mask", None)

        caption_embedding = self._prepare_clip_embeddings(batch)

        compute_cycle = self.use_cycle_consistency and self.dataset_mode in (
            'simulation', 'et', 'ccdm'
        )

        output = self._forward_step(
            camera_trajectory,
            subject_trajectory,
            subject_volume,
            caption_embedding,
            tgt_key_padding_mask,
            is_training=(stage == "train"),
            decode_mode=self.decode_mode,
            compute_cycle_embeddings=compute_cycle,
            conditioning_mode=('prompt_generation' if stage != 'train' and self._validate_conditioning else None),
            caption_presence_mask=batch.get('prompt_none_mask'),
        )

        loss, loss_dict = self.loss_module(
            output,
            camera_trajectory,
            caption_embedding,
            batch,
            tgt_key_padding_mask
        )

        self._log_metrics(stage, loss, loss_dict, len(camera_trajectory))

        if stage == 'train' and output['conditioning_mode'] != 'scheduled':
            for mode in TRAINING_MODES:
                self.log(f'train_mode_fraction/{mode}', float(output['conditioning_mode'] == mode),
                         on_step=False, on_epoch=True, batch_size=len(camera_trajectory), sync_dist=True)
            if output['keyframe_count'] is not None:
                self.log('train_keyframe_count', float(output['keyframe_count']),
                         on_step=False, on_epoch=True, batch_size=len(camera_trajectory))
        if stage == 'val' and self._validate_conditioning:
            self._validate_conditioning_batch(batch, batch_idx, output)
        if stage == 'val' and self._validate_caption_missingness:
            self._validate_caption_missingness_batch(batch, batch_idx)

        return loss

    @property
    def _validate_conditioning(self):
        return self.validation_conditioning is not None and self.validation_conditioning.enabled

    @property
    def _validate_caption_missingness(self):
        return self.validation_caption_missingness is not None and self.validation_caption_missingness.enabled

    def on_validation_epoch_start(self):
        if self._validate_caption_missingness:
            self._caption_missingness_sample_offset = 0
            self._caption_missingness_totals = {
                key: torch.zeros(2, dtype=torch.float64, device=self.device) for key in METRIC_NAMES
            }
        if self._validate_conditioning:
            self._conditioning_sample_offset = 0
            self._conditioning_totals = {
                name: {key: torch.zeros(2, dtype=torch.float64, device=self.device) for key in METRIC_NAMES}
                for name, _, _ in self.validation_conditioning.cases()
            }
            diagnostics = {}
            if self.validation_conditioning.rotation_baselines:
                rotation_names = [key for key in METRIC_NAMES if 'rotation' in key
                                  and not key.startswith(('known_', 'hidden_'))]
                for name in ('sim_zero', 'static_first_pose_oracle'):
                    diagnostics[f'rotation_baseline/{name}'] = rotation_names
            if self.validation_conditioning.caption_shuffle:
                for name in ('correct', 'shuffled'):
                    diagnostics[f'caption_shuffle/{name}'] = METRIC_NAMES
            self._conditioning_diagnostic_totals = {
                name: {key: torch.zeros(2, dtype=torch.float64, device=self.device) for key in keys}
                for name, keys in diagnostics.items()
            }

    def _validate_conditioning_batch(self, batch, batch_idx, prompt_output):
        cfg = self.validation_conditioning
        if cfg.max_batches and batch_idx >= cfg.max_batches:
            return
        camera = batch['camera_trajectory']
        padding = batch.get('padding_mask')
        caption = self._prepare_clip_embeddings(batch)
        # Stable across epochs and independent of training RNG. Within a fixed
        # distributed sampler layout, each rank gets a disjoint seed interval.
        start = cfg.seed + self._conditioning_sample_offset + self.global_rank * 10**9
        seeds = [start + i for i in range(len(camera))]
        self._conditioning_sample_offset += len(camera)
        for name, mode, count in cfg.cases():
            output = prompt_output if mode == 'prompt_generation' else self._forward_step(
                camera, batch['subject_trajectory'], batch['subject_volume'], caption, padding,
                is_training=False, decode_mode=self.decode_mode, compute_cycle_embeddings=False,
                conditioning_mode=mode, keyframe_count=count, keyframe_sample_seeds=seeds,
                caption_presence_mask=batch.get('prompt_none_mask'),
            )
            totals = pose_metric_totals(output['reconstructed'], camera, padding, output['known_mask'])
            for key, pair in totals.items():
                self._conditioning_totals[name][key] += pair
        if cfg.rotation_baselines:
            for name, totals in rotation_baseline_totals(camera, padding).items():
                for key, pair in totals.items():
                    self._conditioning_diagnostic_totals[f'rotation_baseline/{name}'][key] += pair
        if cfg.caption_shuffle:
            shuffled = deterministic_caption_shuffle(caption, batch.get('prompt_none_mask'), start)
            if shuffled is not None:
                shuffled_caption, shuffled_mask = shuffled
                shuffled_output = self._forward_step(
                    camera, batch['subject_trajectory'], batch['subject_volume'], shuffled_caption, padding,
                    is_training=False, decode_mode=self.decode_mode, compute_cycle_embeddings=False,
                    conditioning_mode='prompt_generation', caption_presence_mask=shuffled_mask,
                )
                # Paired correct/shuffled totals use exactly the same clips.
                for name, output in (('correct', prompt_output), ('shuffled', shuffled_output)):
                    totals = pose_metric_totals(output['reconstructed'], camera, padding, output['known_mask'])
                    for key, pair in totals.items():
                        self._conditioning_diagnostic_totals[f'caption_shuffle/{name}'][key] += pair

    def _log_pose_metric_means(self, prefix, totals, *, log_counts=True):
        """Reduce additive metrics without mutating the epoch accumulators."""
        means = {}
        for key, pair in totals.items():
            pair = pair.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(pair)
            total, count = pair
            if log_counts:
                self.log(f'{prefix}/{key}_count', count,
                         on_step=False, on_epoch=True, sync_dist=False)
            if count > 0:
                means[key] = total / count
                self.log(f'{prefix}/{key}', means[key],
                         on_step=False, on_epoch=True, sync_dist=False)
        return means

    def _log_conditioning_diagnostics(self):
        scores = {}
        for name, totals in self._conditioning_diagnostic_totals.items():
            means = self._log_pose_metric_means(f'val_{name}', totals)
            if name.startswith('caption_shuffle/') and 'position_error_normalized' in means:
                # A diagnostic batch can contain only singleton trajectories;
                # log its pose result without fabricating motion evidence.
                weight = (self.validation_conditioning.motion_score_weight
                          if 'rotation_path_error_deg' in means else 0.)
                scores[name] = conditioning_pose_score(means, weight)
        if len(scores) == 2:
            self.log('val_caption_shuffle/score_gap_shuffled_minus_correct',
                     scores['caption_shuffle/shuffled'] - scores['caption_shuffle/correct'],
                     on_step=False, on_epoch=True, sync_dist=False)

    def _validate_caption_missingness_batch(self, batch, batch_idx):
        cfg = self.validation_caption_missingness
        if cfg.max_batches and batch_idx >= cfg.max_batches:
            return
        camera, padding = batch['camera_trajectory'], batch.get('padding_mask')
        start = cfg.seed + self._caption_missingness_sample_offset + self.global_rank * 10**9
        seeds = [start + i for i in range(len(camera))]
        self._caption_missingness_sample_offset += len(camera)
        output = self._forward_step(
            camera, batch['subject_trajectory'], batch['subject_volume'],
            self._prepare_clip_embeddings(batch), padding,
            is_training=False, decode_mode=self.decode_mode, conditioning_mode='prompt_generation',
            caption_presence_mask=batch.get('prompt_none_mask'),
            caption_missingness_seeds=seeds, force_caption_missingness=True,
        )
        totals = pose_metric_totals(output['reconstructed'], camera, padding, output['known_mask'])
        for key, pair in totals.items():
            self._caption_missingness_totals[key] += pair

    def _log_caption_missingness_metrics(self):
        if not self._validate_caption_missingness:
            return None
        means = self._log_pose_metric_means(
            'val_caption_missingness', self._caption_missingness_totals,
        )
        if 'position_error_normalized' not in means:
            raise RuntimeError('Sparse caption validation needs at least one valid pose')
        motion_weight = self.validation_conditioning.motion_score_weight if self._validate_conditioning else 0.
        score = conditioning_pose_score(means, motion_weight)
        self.log('val_caption_missingness_score', score,
                 on_step=False, on_epoch=True, sync_dist=False)
        return score

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        sparse_score = self._log_caption_missingness_metrics()
        if not self._validate_conditioning:
            if sparse_score is not None:
                self.log('val_robustness_score', sparse_score,
                         on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            return
        families = {mode: [] for mode in TRAINING_MODES}
        for name, mode, _ in self.validation_conditioning.cases():
            means = self._log_pose_metric_means(
                f'val_conditioning/{name}',
                {key: self._conditioning_totals[name][key] for key in METRIC_NAMES},
            )
            if 'position_error_normalized' not in means:
                raise RuntimeError('Conditioning validation needs at least one valid pose')
            score = conditioning_pose_score(means, self.validation_conditioning.motion_score_weight)
            if mode.startswith('key_framing'):
                if 'known_position_error_normalized' not in means:
                    raise RuntimeError('Keyframe validation produced no supplied valid poses')
                score = score + self.validation_conditioning.known_pose_weight * (
                    means['known_position_error_normalized'] + means['known_rotation_error_deg'] / 180.
                )
            families[mode].append(score)
        # Each family has equal weight; additional K values do not overweight
        # the keyframe families. Lower score is better, in normalized pose units.
        family_scores = []
        for mode, scores in families.items():
            score = torch.stack(scores).mean()
            family_scores.append(score)
            self.log(f'val_conditioning/{mode}/score', score, on_step=False, on_epoch=True, sync_dist=False)
        conditioning_score = torch.stack(family_scores).mean()
        self.log('val_conditioning_score', conditioning_score,
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        if sparse_score is not None:
            weight = self.validation_caption_missingness.score_weight
            self.log('val_robustness_score', (conditioning_score + weight * sparse_score) / (1 + weight),
                     on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        count = torch.tensor(self._conditioning_sample_offset, dtype=torch.float64, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
        self.log('val_conditioning_sample_count', count, on_step=False, on_epoch=True, sync_dist=False)
        self._log_conditioning_diagnostics()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "test")

    def _get_total_steps(self) -> int:
        return self.trainer.max_epochs * len(
            self.trainer.datamodule.train_dataloader()
        )
