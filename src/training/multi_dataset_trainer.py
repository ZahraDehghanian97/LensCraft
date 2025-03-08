from typing import Optional, Dict, Any, Tuple, List, Any
import torch

from training.base_trainer import BaseTrainer, NoiseConfig, MaskConfig, TeacherForcingConfig


class MultiDatasetTrainer(BaseTrainer):
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
        sim_weight: float = 0.6,
        ccdm_weight: float = 0.4,
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
        
        self.sim_weight = sim_weight
        self.ccdm_weight = ccdm_weight
        
        self.train_step_count = 0

    def _prepare_sim_clip_embeddings(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        return [batch['cinematography_prompt'], batch['simulation_instruction']]
    
    def _process_sim_batch(self, batch: Dict[str, Any], stage: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        camera_trajectory = batch['camera_trajectory']
        subject_trajectory_loc_rot = batch['subject_trajectory_loc_rot']
        subject_volume = batch['subject_volume']
        tgt_key_padding_mask = batch.get("padding_mask", None)
        
        [caption_embedding, additional_embeddings] = self._prepare_sim_clip_embeddings(batch)
        
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
        
        return loss, loss_dict

    def _process_ccdm_batch(self, batch: Dict[str, Any], stage: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        camera_trajectory = batch['camera_trajectory']
        subject_trajectory_loc_rot = batch['subject_trajectory_loc_rot']
        subject_volume = batch['subject_volume']
        tgt_key_padding_mask = batch.get("padding_mask", None)
        
        output = self._forward_step(
            camera_trajectory,
            subject_trajectory_loc_rot,
            subject_volume,
            caption_embedding=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            is_training=(stage == "train"),
            teacher_forcing_ratio=0.0
        )
        
        trajectory_loss = self.loss_module.compute_trajectory_loss(
            output['reconstructed'],
            camera_trajectory
        )
        
        loss_dict = {
            "trajectory": trajectory_loss.item(),
            "total": trajectory_loss.item()
        }
        
        return trajectory_loss, loss_dict

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        sim_batch = batch['simulation']
        ccdm_batch = batch['ccdm']
        
        sim_loss, sim_loss_dict = self._process_sim_batch(sim_batch, "train")
        
        ccdm_loss, ccdm_loss_dict = self._process_ccdm_batch(ccdm_batch, "train")
        
        combined_loss = (self.sim_weight * sim_loss) + (self.ccdm_weight * ccdm_loss)
        
        self._log_metrics("train_sim", sim_loss, sim_loss_dict, len(sim_batch['camera_trajectory']))
        self._log_metrics("train_ccdm", ccdm_loss, ccdm_loss_dict, len(ccdm_batch['camera_trajectory']))
        
        combined_loss_dict = {
            "total": combined_loss.item(),
            "sim_contribution": (self.sim_weight * sim_loss).item(),
            "ccdm_contribution": (self.ccdm_weight * ccdm_loss).item()
        }
        self._log_metrics("train", combined_loss, combined_loss_dict, 
                        len(sim_batch['camera_trajectory']) + len(ccdm_batch['camera_trajectory']))
        
        self.train_step_count += 1
        return combined_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if dataloader_idx == 0:
            loss, loss_dict = self._process_sim_batch(batch, "val")
            self._log_metrics("val_sim", loss, loss_dict, len(batch['camera_trajectory']))
            return loss
        else:
            loss, loss_dict = self._process_ccdm_batch(batch, "val")
            self._log_metrics("val_ccdm", loss, loss_dict, len(batch['camera_trajectory']))
            return loss

    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        
        if isinstance(outputs[0], list):
            sim_outputs = outputs[0]
            ccdm_outputs = outputs[1]
            
            sim_avg_loss = torch.stack([x for x in sim_outputs]).mean()
            ccdm_avg_loss = torch.stack([x for x in ccdm_outputs]).mean()
            
            combined_loss = (self.sim_weight * sim_avg_loss) + (self.ccdm_weight * ccdm_avg_loss)
            
            self.log("val_loss", combined_loss, prog_bar=True)
        else:
            avg_loss = torch.stack([x for x in outputs]).mean()
            self.log("val_loss", avg_loss, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if dataloader_idx == 0:
            loss, loss_dict = self._process_sim_batch(batch, "test")
            self._log_metrics("test_sim", loss, loss_dict, len(batch['camera_trajectory']))
            return loss
        else:
            loss, loss_dict = self._process_ccdm_batch(batch, "test")
            self._log_metrics("test_ccdm", loss, loss_dict, len(batch['camera_trajectory']))
            return loss
    
    def _get_total_steps(self) -> int:
        sim_steps = len(self.trainer.train_dataloader['simulation'])
        ccdm_steps = len(self.trainer.train_dataloader['ccdm'])
        return max(sim_steps, ccdm_steps) * self.trainer.max_epochs
