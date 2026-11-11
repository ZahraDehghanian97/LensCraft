from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass

import lightning as L
import torch

from utils.augmentation import apply_mask_and_noise, linear_increase
from training.conditioning import ConditioningPolicy, prepare_conditioning
from training.caption_missingness import CaptionMissingness


@dataclass
class NoiseConfig:
    initial_std: float
    final_std: float

@dataclass
class MaskConfig:
    initial_ratio: float
    final_ratio: float
    memory_ratio: float = 0.0

@dataclass
class TeacherForcingConfig:
    memory_initial_ratio: float
    memory_final_ratio: float
    trajectory_initial_ratio: float
    trajectory_final_ratio: float


class BaseTrainer(L.LightningModule):
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
        moving_avg_window: int = 10,
        conditioning: Optional[Dict[str, Any]] = None,
        caption_missingness: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.noise = noise
        self.mask = mask
        self.teacher_forcing_schedule = teacher_forcing_schedule
        self.compile_mode = compile_mode
        self.compiled = not compile_enabled
        self.loss_module = loss_module
        self.conditioning = ConditioningPolicy(**dict(conditioning)) if conditioning is not None else None
        self.caption_missingness = (
            CaptionMissingness(caption_missingness)
            if caption_missingness is not None and caption_missingness.get('enabled', True) else None
        )

        self.moving_avg_window = moving_avg_window
        self.metric_history = {"ff": [], "sp": [], "re": [], "ce": [], "cy": []}

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.compiled:
            self.model = torch.compile(self.model, mode=self.compile_mode)
            self.compiled = True

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Persist training scales alongside weights for independent inference."""
        from data.simulation.dataset import SimulationDataset

        datamodule = getattr(self._trainer, "datamodule", None)
        state = datamodule.state_dict() if datamodule is not None else {}
        parameters = state.get("normalization")
        if parameters is None:
            parameters = SimulationDataset._normalization_parameters
        if parameters is not None:
            checkpoint["normalization"] = {
                group: {statistic: torch.as_tensor(value).detach().cpu().clone()
                        for statistic, value in values.items()}
                for group, values in parameters.items()
            }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore the geometric schema and the scales used by these weights."""
        from data.simulation.dataset import SimulationDataset
        from utils.load_lens_craft import migrate_geometric_context_state

        state = checkpoint.get("state_dict")
        if state is not None:
            has_context = any("geometric_context." in key for key in state)
            needs_context = any("geometric_context." in key for key in self.model.state_dict())
            if needs_context and not has_context and checkpoint.get("optimizer_states"):
                raise ValueError(
                    "Cannot resume optimizer state from a checkpoint predating geometric context. "
                    "Use initialize_from_checkpoint=<path> and resume_checkpoint=null to continue "
                    "training with a fresh optimizer."
                )
            checkpoint["state_dict"] = migrate_geometric_context_state(self.model, state, prefix="model.")
        parameters = checkpoint.get("normalization")
        if parameters is not None:
            SimulationDataset.set_current_normalization_parameters(parameters)
            datamodule = getattr(self._trainer, "datamodule", None)
            if datamodule is not None:
                datamodule.load_state_dict({"normalization": parameters})

    def _calculate_schedule_parameters(self) -> Dict[str, float]:
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs

        return {
            'noise_std': linear_increase(
                self.noise.initial_std,
                self.noise.final_std,
                current_epoch,
                max_epochs
            ),
            'mask_ratio': linear_increase(
                self.mask.initial_ratio,
                self.mask.final_ratio,
                current_epoch,
                max_epochs
            ),
            'memory_mask_ratio': self.mask.memory_ratio,
            'memory_teacher_forcing_ratio': linear_increase(
                self.teacher_forcing_schedule.memory_initial_ratio,
                self.teacher_forcing_schedule.memory_final_ratio,
                current_epoch,
                max_epochs
            ),
            'trajectory_teacher_forcing_ratio': linear_increase(
                self.teacher_forcing_schedule.trajectory_initial_ratio,
                self.teacher_forcing_schedule.trajectory_final_ratio,
                current_epoch,
                max_epochs
            )
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
        geometry_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        ratios = self._calculate_schedule_parameters()

        if is_training and conditioning_mode is None and self.conditioning is not None and self.conditioning.enabled:
            conditioning_mode, keyframe_count = self.conditioning.sample()

        if (caption_embedding is None and (conditioning_mode in ('prompt_generation', 'key_framing+prompt')
                or (not is_training and conditioning_mode is None))):
            raise ValueError(
                "Validation/test require caption embeddings for the "
                "caption-only conditioning path"
            )

        valid_len = (
            (~tgt_key_padding_mask).sum(dim=1)
            if tgt_key_padding_mask is not None else None
        )

        if conditioning_mode is not None:
            model_input, src_key_mask, known_mask, memory_ratio = prepare_conditioning(
                camera_trajectory, tgt_key_padding_mask, conditioning_mode,
                keyframe_count, keyframe_sample_seeds,
            )
        elif is_training:
            model_input, src_key_mask = apply_mask_and_noise(
                camera_trajectory,
                valid_len,
                ratios['mask_ratio'],
                ratios['noise_std'],
                self.device
            )
        else:
            # Validation/test use clean input but optimize exactly the same loss
            # terms (including cycle consistency). Padding still has to reach
            # the encoder.
            model_input = camera_trajectory
            src_key_mask = tgt_key_padding_mask

        if conditioning_mode is None:
            memory_ratio = ratios['memory_teacher_forcing_ratio'] if is_training else 1.0
            known_mask = torch.zeros(camera_trajectory.shape[:2], dtype=torch.bool, device=camera_trajectory.device)

        decoder_caption = caption_embedding
        effective_caption_mask = None
        if caption_embedding is not None and caption_presence_mask is not None:
            if (not torch.is_tensor(caption_presence_mask)
                    or caption_presence_mask.dtype != torch.bool
                    or caption_embedding.ndim != 3
                    or caption_presence_mask.shape != (caption_embedding.shape[1], caption_embedding.shape[0])):
                raise ValueError('Caption presence_mask must be boolean [batch, caption fields]')
            effective_caption_mask = caption_presence_mask.to(caption_embedding.device)
        if self.caption_missingness is not None and caption_embedding is not None and memory_ratio > 0:
            if is_training or force_caption_missingness:
                decoder_caption, effective_caption_mask = self.caption_missingness.apply(
                    caption_embedding, effective_caption_mask,
                    sample_seeds=caption_missingness_seeds,
                    force=force_caption_missingness,
                )

        # Native ET missingness matters even without synthetic dropout. A
        # caption-conditioned cycle must not match unsupplied constraints from
        # the clean camera encoding. Camera-only reconstruction, however, has
        # the camera's constraints available and keeps its full cycle target.
        cycle_prompt_mask = effective_caption_mask if memory_ratio > 0 else None
        caption_kwargs = (
            {'caption_presence_mask': effective_caption_mask}
            if decoder_caption is not None else {}
        )
        # Only keyframe modes pin supplied poses. Reconstruction must still
        # predict every valid frame, even though its full source is visible.
        keyframe_kwargs = (
            {'known_camera_mask': known_mask}
            if conditioning_mode in ('key_framing', 'key_framing+prompt') else {}
        )

        output = self.model(
            src=model_input,
            subject_trajectory=subject_trajectory,
            subject_volume=subject_volume,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_mask=src_key_mask,
            target=camera_trajectory if is_training and conditioning_mode is None else None,
            caption_embedding=decoder_caption,
            # Evaluation must measure the caption-conditioned generator, not
            # a moving blend with an encoding of the ground-truth trajectory.
            # A fixed ratio of one makes validation/test leakage-free and
            # comparable across epochs and Optuna trials.
            memory_teacher_forcing_ratio=memory_ratio,
            trajectory_teacher_forcing_ratio=(
                ratios['trajectory_teacher_forcing_ratio']
                if is_training and conditioning_mode is None else 0.0
            ),
            mask_memory_prob=(
                ratios['memory_mask_ratio'] if is_training and conditioning_mode is None else 0.0
            ),
            decode_mode=decode_mode,
            **caption_kwargs,
            **keyframe_kwargs,
            **(geometry_context or {}),
        )
        output['known_mask'] = known_mask
        output['conditioning_mode'] = conditioning_mode or 'scheduled'
        output['keyframe_count'] = keyframe_count
        if cycle_prompt_mask is not None:
            # Keep the caller's clean caption for encoder supervision, while
            # cycle alignment only judges constraints available to the decoder.
            output['cycle_prompt_mask'] = cycle_prompt_mask

        if compute_cycle_embeddings:
            if conditioning_mode is not None:
                # Judge the generated path, including filled frames. Reusing
                # the source observation mask would leave only fixed poses
                # and remove every decoder prediction from the cycle loss.
                cycle_key_mask = tgt_key_padding_mask
                cycle_input = output['reconstructed']
                if cycle_key_mask is not None:
                    cycle_input = cycle_input.masked_fill(cycle_key_mask[..., None], 0)
            elif is_training:
                cycle_input, cycle_key_mask = apply_mask_and_noise(
                    output["reconstructed"],
                    valid_len,
                    ratios['mask_ratio'],
                    ratios['noise_std'],
                    self.device
                )
            else:
                cycle_input = output["reconstructed"]
                cycle_key_mask = tgt_key_padding_mask
            cycle_embeddings = self.model.encoder(
                cycle_input,
                output["subject_embedding"],
                cycle_key_mask,
                subject_key_padding_mask=tgt_key_padding_mask,
            )
            output['cycle_embeddings'] = cycle_embeddings

        return output

    def _log_metrics(self, stage: str, loss: torch.Tensor, loss_dict: Dict[str, Any], batch_size: int) -> None:
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        progress_metrics = {}
        if stage == "train":
            progress_metrics["ts"] = loss
        elif stage == "val":
            progress_metrics["ve"] = loss

        metric_keys = {"first_frame": "ff", "speed": "sp", "relative": "re", "categorical": "ce", "cycle": "cy"}
        for key, p_key in metric_keys.items():
            if key in loss_dict:
                current_value = loss_dict[key] if isinstance(loss_dict[key], float) else loss_dict[key].item()

                self.metric_history[p_key].append(current_value)

                if len(self.metric_history[p_key]) > self.moving_avg_window:
                    self.metric_history[p_key] = self.metric_history[p_key][-self.moving_avg_window:]

                if self.metric_history[p_key]:
                    moving_avg = sum(self.metric_history[p_key]) / len(self.metric_history[p_key])
                    progress_metrics[p_key] = moving_avg

        self.log_dict(progress_metrics, prog_bar=True, logger=True, batch_size=batch_size)

        for key, value in loss_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.log(f"{stage}_{key}_{subkey}", subvalue, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
            else:
                self.log(f"{stage}_{key}", value, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List, List]]:
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is not None:
            total_steps = self._get_total_steps()

            scheduler = self.lr_scheduler(
                optimizer=optimizer,
                total_steps=total_steps
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def _get_total_steps(self) -> int:
        """Should be implemented by child classes to return total training steps."""
        raise NotImplementedError

    def lr_scheduler_step(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metric: Any
    ) -> None:
        scheduler.step(self.global_step)

    def on_before_optimizer_step(self, optimizer) -> None:
        checks = [torch.isfinite(p.grad).all()
                for p in self.parameters() if p.grad is not None]
        if checks and not torch.stack(checks).all():
            self.log("grad_nan_skip", 1.0, on_step=True, prog_bar=True)
            for p in self.parameters():
                p.grad = None

    def on_train_epoch_end(self) -> None:
        optimizer = self.trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            self.log('learning_rate', lr, on_step=False, on_epoch=True)

        train_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        if train_loss is not None:
            self.log('te', train_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            val_loss = self.trainer.callback_metrics.get('val_loss_epoch')
        if val_loss is not None:
            self.log('ve', val_loss, on_step=False, on_epoch=True, prog_bar=True)
