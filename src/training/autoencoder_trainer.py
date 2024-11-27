import lightning as L
import torch
from torch.nn.functional import mse_loss
from utils.augmentation import apply_mask_and_noise, linear_increase


class LightningMultiTaskAutoencoder(L.LightningModule):
    def __init__(self, model, optimizer, lr_scheduler, noise, mask, teacher_forcing_schedule, compile_mode="default", compile_enabled=True, dataset_mode='simulation'):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.noise = noise
        self.mask = mask
        self.teacher_forcing_schedule = teacher_forcing_schedule
        self.compile_mode = compile_mode
        self.compiled = not compile_enabled
        self.dataset_mode = dataset_mode

    def setup(self, stage=None):
        if not self.compiled:
            self.model = torch.compile(self.model, mode=self.compile_mode)
            self.compiled = True

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def _forward_step(self, camera_trajectory, subject_trajectory, tgt_key_padding_mask, is_training=False):
        if not is_training:
            return self.model(camera_trajectory, subject_trajectory, tgt_key_padding_mask)

        current_noise_std = linear_increase(
            initial_value=self.noise.initial_std,
            final_value=self.noise.final_std,
            current_epoch=self.current_epoch,
            total_epochs=self.trainer.max_epochs
        )

        current_mask_ratio = linear_increase(
            initial_value=self.mask.initial_ratio,
            final_value=self.mask.final_ratio,
            current_epoch=self.current_epoch,
            total_epochs=self.trainer.max_epochs
        )

        current_teacher_forcing_ratio = linear_increase(
            initial_value=self.teacher_forcing_schedule.initial_ratio,
            final_value=self.teacher_forcing_schedule.final_ratio,
            current_epoch=self.current_epoch,
            total_epochs=self.trainer.max_epochs
        )

        valid_len = (~tgt_key_padding_mask).sum(dim=1) if tgt_key_padding_mask is not None else None
        noisy_trajectory, src_key_mask = apply_mask_and_noise(
            camera_trajectory,
            valid_len,
            current_mask_ratio,
            current_noise_std,
            self.device
        )

        return self.model(
            noisy_trajectory, 
            subject_trajectory, 
            tgt_key_padding_mask, 
            src_key_mask,
            camera_trajectory, 
            current_teacher_forcing_ratio
        )

    def _step(self, batch, batch_idx, stage):
        camera_trajectory, subject_trajectory, tgt_key_padding_mask = batch['camera_trajectory'], batch['subject_trajectory'], batch.get("padding_mask", None)
        
        output = self._forward_step(
            camera_trajectory, 
            subject_trajectory, 
            tgt_key_padding_mask, 
            is_training=(stage == "train")
        )
        
        clip_targets = {'cls': batch['caption_feat']} if self.dataset_mode == 'et' else {
            'movement': batch['movement_clip'],
            'easing': batch['easing_clip'],
            'camera_angle': batch['angle_clip'],
            'shot_type': batch['shot_clip']
        }

        loss, loss_dict = self.compute_loss(
            output, camera_trajectory, clip_targets, tgt_key_padding_mask)

        self._log_metrics(stage, loss, loss_dict)

        return loss

    def _log_metrics(self, stage, loss, loss_dict):
        self.log(f"{stage}_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        for key, value in loss_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.log(f"{stage}_{key}_{subkey}", subvalue,
                             on_step=True, on_epoch=True, logger=True)
            else:
                self.log(f"{stage}_{key}", value, on_step=True,
                         on_epoch=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def compute_loss(self, model_output, camera_trajectory, clip_targets, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed'].flatten(0, 1)
        camera_trajectory = camera_trajectory.flatten(0, 1)

        if tgt_key_padding_mask is not None:
            valid_mask = (~tgt_key_padding_mask).reshape(-1)
            reconstructed = reconstructed[valid_mask]
            camera_trajectory = camera_trajectory[valid_mask]

        clip_embeddings = {
            k: model_output[f'{k}_embedding'] for k in clip_targets.keys()
        }

        return self.total_loss(reconstructed, camera_trajectory, clip_embeddings, clip_targets)

    def total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target):
        trajectory_loss = self.combined_trajectory_loss(
            trajectory_pred, trajectory_target)

        clip_losses = {}
        for key in clip_pred.keys():
            clip_losses[key] = self.clip_similarity_loss(
                clip_pred[key], clip_target[key])

        total_clip_loss = sum(clip_losses.values())

        total_loss = trajectory_loss + total_clip_loss
        loss_dict = {
            'trajectory': trajectory_loss.item(),
            'clip': {k: v.item() for k, v in clip_losses.items()},
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def combined_trajectory_loss(self, pred, target):
        pred_position = pred[:, :4]
        target_position = target[:, :4]

        pred_angle = pred[:, 4:]
        target_angle = target[:, 4:]

        position_loss = mse_loss(pred_position, target_position)
        angle_loss = self.circular_distance_loss(pred_angle, target_angle)

        return position_loss + angle_loss

    def circular_distance_loss(self, pred, target):
        pred_rad = pred * torch.pi / 180.0
        target_rad = target * torch.pi / 180.0

        sin_diff = torch.sin(target_rad) - torch.sin(pred_rad)
        cos_diff = torch.cos(target_rad) - torch.cos(pred_rad)

        return torch.mean(sin_diff.pow(2) + cos_diff.pow(2))

    def clip_similarity_loss(self, pred_embedding, target_embedding):
        similarity = torch.nn.functional.cosine_similarity(
            pred_embedding, target_embedding)
        return 1 - similarity.mean()

    def on_train_epoch_end(self):
        optimizer = self.trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            self.log('learning_rate', lr, on_step=False, on_epoch=True)
