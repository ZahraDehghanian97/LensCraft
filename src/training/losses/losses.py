from torch.nn.functional import mse_loss, cosine_similarity
from .angle_loss import AngleLoss

class CameraTrajectoryLoss:
    def __init__(self):
        self.angle_loss = AngleLoss()
        self.position_slice = slice(0, 4)
        self.rotation_slice = slice(4, None)

    def __call__(self, model_output, camera_trajectory, clip_targets, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']
        
        # if tgt_key_padding_mask is not None: TODO: fix dimensions
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
        
        clip_losses = {
            key: self.compute_clip_loss(clip_pred[key], clip_target[key])
            for key in clip_pred.keys()
        }
        total_clip_loss = sum(clip_losses.values())

        total_loss = trajectory_loss + total_clip_loss + speed_loss
        loss_dict = {
            'trajectory': trajectory_loss.item(),
            'speed': speed_loss.item(),
            'clip': {k: v.item() for k, v in clip_losses.items()},
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def compute_component_losses(self, pred, target):
        position_loss = mse_loss(
            pred[..., self.position_slice], 
            target[..., self.position_slice]
        )
        rotation_loss = self.angle_loss(
            pred[..., self.rotation_slice], 
            target[..., self.rotation_slice]
        )
        return position_loss + rotation_loss

    def compute_velocity(self, trajectory):
        return trajectory[:, 1:] - trajectory[:, :-1]

    def compute_speed_loss(self, pred, target):
        pred_velocity = self.compute_velocity(pred)
        target_velocity = self.compute_velocity(target)
        return self.compute_component_losses(pred_velocity, target_velocity)

    def compute_trajectory_loss(self, pred, target):
        relative_loss = self.compute_component_losses(pred[:, 1:] - pred[:, 0:1], target[:, 1:] - target[:, 0:1])
        first_frame_loss = self.compute_component_losses(pred[:, 0:1], target[:, 0:1])
        
        return relative_loss + first_frame_loss

    @staticmethod
    def compute_clip_loss(pred_embedding, target_embedding):
        similarity = cosine_similarity(pred_embedding, target_embedding)
        return 1 - similarity.mean()