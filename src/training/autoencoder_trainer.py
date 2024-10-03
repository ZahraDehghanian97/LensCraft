import lightning as L
import torch
import numpy as np
from utils.augmentation import apply_mask_and_noise, get_noise_and_mask_values


class LightningMultiTaskAutoencoder(L.LightningModule):
    def __init__(self, model, optimizer_config):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch, batch_idx, stage):
        camera_trajectory = batch['camera_trajectory']
        subject = batch['subject']
        clip_targets = {
            'movement': batch['movement_clip'],
            'easing': batch['easing_clip'],
            'camera_angle': batch['angle_clip'],
            'shot_type': batch['shot_clip']
        }

        if stage == "train":
            current_noise_std, current_mask_ratio = get_noise_and_mask_values(
                self.current_epoch, self.trainer.max_epochs, self.optimizer_config)
            current_teacher_forcing_ratio = self.optimizer_config['init_teacher_forcing_ratio'] * (
                1 - self.current_epoch / self.trainer.max_epochs)

            noisy_trajectory, mask, src_key_padding_mask = apply_mask_and_noise(
                camera_trajectory, current_mask_ratio, current_noise_std, self.device)

            output = self.model(noisy_trajectory, subject, src_key_padding_mask,
                                camera_trajectory, current_teacher_forcing_ratio)
        else:
            output = self.model(camera_trajectory, subject)

        loss, loss_dict = self.compute_loss(
            output, camera_trajectory, clip_targets)

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

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.optimizer_config['lr'],
                                weight_decay=self.optimizer_config['weight_decay'])

    def compute_loss(self, model_output, camera_trajectory, clip_targets, mask=None):
        reconstructed = model_output['reconstructed']
        if mask is not None:
            reconstructed = reconstructed[mask]
            camera_trajectory = camera_trajectory[mask]

        clip_embeddings = {k: model_output[f'{k}_embedding']
                           for k in clip_targets.keys()}
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

        position_loss = torch.nn.functional.mse_loss(
            pred_position, target_position)
        angle_loss = self.circular_distance_loss(pred_angle, target_angle)

        return position_loss + angle_loss

    def circular_distance_loss(self, pred, target):
        pred = torch.clamp(pred, min=-np.pi, max=np.pi)
        target = torch.clamp(target, min=-np.pi, max=np.pi)

        distance = torch.abs(pred - target)
        distance = torch.where(distance > np.pi, 2 *
                               np.pi - distance, distance)

        return torch.mean(distance ** 2)

    def clip_similarity_loss(self, pred_embedding, target_embedding):
        similarity = torch.nn.functional.cosine_similarity(
            pred_embedding, target_embedding)
        return 1 - similarity.mean()
