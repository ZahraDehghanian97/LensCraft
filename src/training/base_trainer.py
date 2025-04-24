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
        use_merged_memory: bool = True
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
        decode_mode: str = 'single_step'
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

        return self.model(
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
            decode_mode,
        )

    def _log_metrics(self, stage: str, loss: torch.Tensor, loss_dict: Dict[str, Any], batch_size: int) -> None:
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            batch_size=batch_size
        )
        
        if stage == "train":
            self.log(
                "ts",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=batch_size
            )
            self.log(
                "te",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size
            )
        elif stage == "val":
            self.log(
                "ve",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size
            )
        
        if "trajectory" in loss_dict:
            trajectory_val = loss_dict["trajectory"] if isinstance(loss_dict["trajectory"], float) else loss_dict["trajectory"].item()
            self.log(
                "tr",
                trajectory_val,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=batch_size
            )
        
        if "average_clip" in loss_dict:
            clip_val = loss_dict["average_clip"] if isinstance(loss_dict["average_clip"], float) else loss_dict["average_clip"].item()
            self.log(
                "cl",
                clip_val,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=batch_size
            )
        elif "clip" in loss_dict and isinstance(loss_dict["clip"], dict):
            clip_values = list(loss_dict["clip"].values())
            if clip_values:
                clip_avg = sum(clip_values) / len(clip_values)
                clip_avg = clip_avg if isinstance(clip_avg, float) else clip_avg.item()
                self.log(
                    "cl",
                    clip_avg,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch_size
                )
        
        for key, value in loss_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.log(
                        f"{stage}_{key}_{subkey}",
                        subvalue,
                        on_step=True,
                        on_epoch=True,
                        logger=True,
                        batch_size=batch_size
                    )
            else:
                self.log(
                    f"{stage}_{key}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size
                )

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
