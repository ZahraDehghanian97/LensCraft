from torch.nn.functional import mse_loss, cosine_similarity
from .angle_loss import AngleLoss

class CameraTrajectoryLoss:
    def __init__(self):
        self.angle_loss = AngleLoss()

    def __call__(self, model_output, camera_trajectory, clip_targets, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']

        # if tgt_key_padding_mask is not None: TODO: fix dimentions
        #     valid_mask = (~tgt_key_padding_mask).reshape(-1)
        #     reconstructed_flat = reconstructed_flat[valid_mask]
        #     camera_trajectory_flat = camera_trajectory_flat[valid_mask]

        clip_embeddings = {
            k: model_output[f'{k}_embedding'] for k in clip_targets.keys()
        }

        return self.compute_total_loss(reconstructed, camera_trajectory, clip_embeddings, clip_targets)

    def compute_total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target):
        trajectory_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)
        
        speed_loss = self.compute_speed_loss(trajectory_pred, trajectory_target)

        clip_losses = {}
        for key in clip_pred.keys():
            clip_losses[key] = self.compute_clip_loss(
                clip_pred[key], clip_target[key])

        total_clip_loss = sum(clip_losses.values())

        total_loss = trajectory_loss + total_clip_loss + speed_loss
        loss_dict = {
            'trajectory': trajectory_loss.item(),
            'speed': speed_loss.item(),
            'clip': {k: v.item() for k, v in clip_losses.items()},
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def compute_speed_loss(self, pred, target):
        pred_velocity = pred[:, 1:] - pred[:, :-1]
        target_velocity = target[:, 1:] - target[:, :-1]
        
        pred_pos_velocity = pred_velocity[..., :4]
        target_pos_velocity = target_velocity[..., :4]
        
        pred_rot_velocity = pred_velocity[..., 4:]
        target_rot_velocity = target_velocity[..., 4:]
        
        pos_velocity_loss = mse_loss(pred_pos_velocity, target_pos_velocity)
        rot_velocity_loss = self.angle_loss(pred_rot_velocity, target_rot_velocity)
        
        return pos_velocity_loss + rot_velocity_loss

    def compute_trajectory_loss(self, pred, target):
        pred_position = pred[..., :4]
        target_position = target[..., :4]

        pred_angle = pred[..., 4:]
        target_angle = target[..., 4:]

        position_loss = mse_loss(pred_position, target_position)
        angle_loss = self.angle_loss(pred_angle, target_angle)
                
        return position_loss + angle_loss

    @staticmethod
    def compute_clip_loss(pred_embedding, target_embedding):
        similarity = cosine_similarity(pred_embedding, target_embedding)
        return 1 - similarity.mean()