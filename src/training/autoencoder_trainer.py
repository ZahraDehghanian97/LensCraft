from typing import Optional, Dict, Any, List, Union
import torch

from training.base_trainer import BaseTrainer, NoiseConfig, MaskConfig, TeacherForcingConfig


class LightningMultiTaskAutoencoder(BaseTrainer):
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
        use_cycle_consistency: bool = False
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
            use_merged_memory
        )
        self.dataset_mode = dataset_mode
        self.decode_mode = decode_mode
        self.use_cycle_consistency = use_cycle_consistency

    def _prepare_clip_embeddings(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        if self.dataset_mode == 'et' or self.use_merged_memory:
            return [torch.stack([batch['caption_feat']]), torch.stack([])]
        
        return [batch['cinematography_prompt'], batch['simulation_instruction']]

    def _step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        camera_trajectory = batch['camera_trajectory']
        subject_trajectory = batch['subject_trajectory']
        subject_volume = batch['subject_volume']
        tgt_key_padding_mask = batch.get("padding_mask", None)
        
        [caption_embedding, additional_embeddings] = self._prepare_clip_embeddings(batch)
        
        compute_cycle = self.use_cycle_consistency and stage == "train" and self.dataset_mode == 'simulation'
        
        output = self._forward_step(
            camera_trajectory,
            subject_trajectory,
            subject_volume,
            caption_embedding,
            tgt_key_padding_mask,
            is_training=(stage == "train"),
            decode_mode=self.decode_mode,
            compute_cycle_embeddings=compute_cycle
        )
        
        merge_embeddings = torch.cat([caption_embedding, additional_embeddings], dim=0)
        
        loss, loss_dict = self.loss_module(
            output,
            camera_trajectory,
            merge_embeddings,
            batch,
            tgt_key_padding_mask
        )

        self._log_metrics(stage, loss, loss_dict, len(camera_trajectory))

        return loss

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
