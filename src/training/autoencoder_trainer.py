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
    initial_ratio: float
    final_ratio: float

class LightningMultiTaskAutoencoder(L.LightningModule):
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
        use_merged_memory: bool = True
    ):
        super().__init__()
        # self.save_hyperparameters(ignore=['model', 'loss_module'])
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.noise = noise
        self.mask = mask
        self.teacher_forcing_schedule = teacher_forcing_schedule
        self.compile_mode = compile_mode
        self.compiled = not compile_enabled
        self.dataset_mode = dataset_mode
        self.loss_module = loss_module
        self.use_merged_memory = use_merged_memory

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.compiled:
            self.model = torch.compile(self.model, mode=self.compile_mode)
            self.compiled = True

    def _get_current_ratios(self) -> Dict[str, float]:
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
            'teacher_forcing_ratio': linear_increase(
                self.teacher_forcing_schedule.initial_ratio,
                self.teacher_forcing_schedule.final_ratio,
                current_epoch,
                max_epochs
            )
        }

    def _prepare_clip_embeddings(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        if self.dataset_mode == 'et' or self.use_merged_memory:
            return [torch.stack([batch['caption_feat']]), torch.stack([])]
        
        return [batch['cinematography_prompt'], batch['simulation_instruction']]

    def _forward_step(
        self,
        camera_trajectory: torch.Tensor,
        subject_trajectory_loc_rot: torch.Tensor,
        subject_volume: torch.Tensor,
        caption_embedding: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor],
        is_training: bool = False
    ) -> Dict[str, torch.Tensor]:
        if not is_training:
            return self.model(
                camera_trajectory,
                subject_trajectory_loc_rot,
                subject_volume,
                tgt_key_padding_mask,
                caption_embedding=caption_embedding,
                teacher_forcing_ratio=0.5
            )

        ratios = self._get_current_ratios()
        
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
            subject_trajectory_loc_rot,
            subject_volume,
            tgt_key_padding_mask,
            src_key_mask,
            camera_trajectory,
            caption_embedding,
            ratios['teacher_forcing_ratio'],
            ratios['memory_mask_ratio'],
        )

    def _step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        camera_trajectory = batch['camera_trajectory']
        subject_trajectory_loc_rot = batch['subject_trajectory_loc_rot']
        subject_volume = batch['subject_volume']
        tgt_key_padding_mask = batch.get("padding_mask", None)
        
        [caption_embedding, additional_embeddings] = self._prepare_clip_embeddings(batch)
        
        output = self._forward_step(
            camera_trajectory,
            subject_trajectory_loc_rot,
            subject_volume,
            caption_embedding,
            tgt_key_padding_mask,
            is_training=(stage == "train")
        )
        
        merge_embeddings = torch.cat([caption_embedding, additional_embeddings], dim=0)
        
        loss, loss_dict = self.loss_module(
            output,
            camera_trajectory,
            merge_embeddings,
            tgt_key_padding_mask
        )

        self._log_metrics(stage, loss, loss_dict)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "test")

    def _log_metrics(self, stage: str, loss: torch.Tensor, loss_dict: Dict[str, Any]) -> None:
        batch_size = self.trainer.datamodule.batch_size
        
        self.log(
            f"{stage}_loss", 
            loss, 
            on_step=True, 
            on_epoch=True,
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
            total_steps = self.trainer.max_epochs * len(
                self.trainer.datamodule.train_dataloader()
            )
            scheduler = self.lr_scheduler(
                optimizer=optimizer, 
                total_steps=total_steps
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        return optimizer
    
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
