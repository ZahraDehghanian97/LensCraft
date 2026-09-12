from typing import Optional, Dict, Any
import torch

from training.base_trainer import BaseTrainer, NoiseConfig, MaskConfig, TeacherForcingConfig
from data.convertor.convertor import convert_to_target
from data.simulation.utils import structured_conditioning_from_batch
from training.conditioning import ConditioningValidation, TRAINING_MODES
from training.conditioning_metrics import METRIC_NAMES, pose_metric_totals


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
        )
        self.dataset_mode = dataset_mode
        self.decode_mode = decode_mode
        self.use_cycle_consistency = use_cycle_consistency
        self.validation_conditioning = (
            ConditioningValidation(**dict(validation_conditioning))
            if validation_conditioning is not None else None
        )

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

        return loss

    @property
    def _validate_conditioning(self):
        return self.validation_conditioning is not None and self.validation_conditioning.enabled

    def on_validation_epoch_start(self):
        if self._validate_conditioning:
            self._conditioning_sample_offset = 0
            self._conditioning_totals = {
                name: {key: torch.zeros(2, dtype=torch.float64, device=self.device) for key in METRIC_NAMES}
                for name, _, _ in self.validation_conditioning.cases()
            }

    def _validate_conditioning_batch(self, batch, batch_idx, prompt_output):
        cfg = self.validation_conditioning
        if batch_idx >= cfg.max_batches:
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
            )
            totals = pose_metric_totals(output['reconstructed'], camera, padding, output['known_mask'])
            for key, pair in totals.items():
                self._conditioning_totals[name][key] += pair

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if not self._validate_conditioning:
            return
        families = {mode: [] for mode in TRAINING_MODES}
        for name, mode, _ in self.validation_conditioning.cases():
            means = {}
            for key in METRIC_NAMES:
                pair = self._conditioning_totals[name][key].clone()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(pair)
                total, count = pair
                self.log(f'val_conditioning/{name}/{key}_count', count,
                         on_step=False, on_epoch=True, sync_dist=False)
                if count > 0:
                    means[key] = total / count
                    self.log(f'val_conditioning/{name}/{key}', means[key],
                             on_step=False, on_epoch=True, sync_dist=False)
            if 'position_error_normalized' not in means:
                raise RuntimeError('Conditioning validation needs at least one valid pose')
            score = means['position_error_normalized'] + means['rotation_error_deg'] / 180.
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
        self.log('val_conditioning_score', torch.stack(family_scores).mean(),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        count = torch.tensor(self._conditioning_sample_offset, dtype=torch.float64, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
        self.log('val_conditioning_sample_count', count, on_step=False, on_epoch=True, sync_dist=False)

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
