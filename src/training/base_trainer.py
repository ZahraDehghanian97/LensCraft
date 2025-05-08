from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass

import lightning as L
import torch

from utils.augmentation import apply_mask_and_noise, linear_increase


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
        use_merged_memory: bool = True,
        moving_avg_window: int = 10
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
        self.use_merged_memory = use_merged_memory
        
        self.moving_avg_window = moving_avg_window
        self.metric_history = {"ff": [], "sp": [], "re": [], "cl": [], "cy": [], "co": []}

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.compiled:
            self.model = torch.compile(self.model, mode=self.compile_mode)
            self.compiled = True

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
        compute_cycle_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        if not is_training:
            return self.model(
                camera_trajectory,
                subject_trajectory,
                subject_volume,
                tgt_key_padding_mask,
                caption_embedding=caption_embedding,
                memory_teacher_forcing_ratio=0.5,
                trajectory_teacher_forcing_ratio=0.0,
                decode_mode=decode_mode
            )

        ratios = self._calculate_schedule_parameters()
        
        valid_len = (
            (~tgt_key_padding_mask).sum(dim=1)
            if tgt_key_padding_mask is not None else None
        )
        
        noisy_masked_trajectory, src_key_mask = apply_mask_and_noise(
            camera_trajectory,
            valid_len,
            ratios['mask_ratio'],
            ratios['noise_std'],
            self.device
        )

        output = self.model(
            noisy_masked_trajectory,
            subject_trajectory,
            subject_volume,
            tgt_key_padding_mask,
            src_key_mask,
            camera_trajectory,
            caption_embedding,
            ratios['memory_teacher_forcing_ratio'],
            ratios['trajectory_teacher_forcing_ratio'],
            ratios['memory_mask_ratio'],
            decode_mode
        )
            
        if compute_cycle_embeddings:
            noisy_masked_trajectory, src_key_mask = apply_mask_and_noise(
                output["reconstructed"],
                valid_len,
                ratios['mask_ratio'],
                ratios['noise_std'],
                self.device
            )
            cycle_embeddings = self.model.encoder(
                noisy_masked_trajectory,
                output["subject_embedding_loc_rot_vol"],
                src_key_mask
            )
            output['cycle_embeddings'] = cycle_embeddings
        
        return output
    
    def _log_metrics(self, stage: str, loss: torch.Tensor, loss_dict: Dict[str, Any], batch_size: int) -> None:
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        
        progress_metrics = {}
        progress_metrics["ts" if stage == "train" else "ve"] = loss
        
        metric_keys = {"first_frame": "ff", "speed": "sp", "relative": "re", "clip": "cl", "cycle": "cy", "contrastive": "co"}
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

    def on_train_epoch_end(self) -> None:
        optimizer = self.trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            self.log('learning_rate', lr, on_step=False, on_epoch=True)
        
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        if train_loss is not None:
            self.log('te', train_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get('val_loss_epoch')
        if val_loss is not None:
            self.log('ve', val_loss, on_step=False, on_epoch=True, prog_bar=True)
