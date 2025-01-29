import torch
from torch.nn.functional import mse_loss, cosine_similarity, cross_entropy
from .angle_loss import AngleLoss

class CameraTrajectoryLoss:
    def __init__(self, clip_embeddings=[], contrastive=False):
        self.angle_loss = AngleLoss()
        self.position_slice = slice(0, 4)
        self.rotation_slice = slice(4, None)
        self.contrastive = contrastive

        # self.clip_embeddings = clip_embeddings
        # self.all_categories = {}
        # if self.contrastive:
        #     device = next(iter(clip_embeddings['movement'].values())).device
        #     for cat in ['movement', 'easing', 'angle', 'shot']:
        #         keys = list(self.clip_embeddings[cat].keys())
        #         embeds = torch.stack([self.clip_embeddings[cat][k].squeeze(0) for k in keys]).to(device)
        #         embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        #         self.all_categories[cat] = (keys, embeds)

    def __call__(self, model_output, camera_trajectory, clip_targets, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']
        clip_pred = model_output['embeddings']
        
        # if tgt_key_padding_mask is not None: TODO: fix dimensions
        #     # Apply padding mask to both reconstructed and target trajectories
        #     valid_mask = ~tgt_key_padding_mask
        #     reconstructed = reconstructed * valid_mask.unsqueeze(-1)
        #     camera_trajectory = camera_trajectory * valid_mask.unsqueeze(-1)

        return self.compute_total_loss(
            reconstructed,
            camera_trajectory,
            clip_pred,
            clip_targets
        )

    def compute_total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target):
        trajectory_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)
        
        clip_losses = []
        clip_loss = 0
        if self.contrastive:
            clip_loss, clip_losses = self.compute_contrastive_loss(clip_pred, clip_target)
        else:            
            for i in range(clip_pred.shape[0]):
                current_loss = self.compute_clip_loss(clip_pred[i], clip_target[i])
                clip_losses.append(current_loss)
                clip_loss += current_loss
        
        total_loss = trajectory_loss + clip_loss / clip_pred.shape[1] * 100
        loss_dict = {
            'trajectory': trajectory_loss.item(),
            'clip': {i: clip_losses[i] for i in range(len(clip_losses))},
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

    def compute_trajectory_loss(self, pred, target):
        first_frame_loss = self.compute_component_losses(pred[:, 0:1], target[:, 0:1])
        relative_loss = self.compute_component_losses(pred[:, 1:] - pred[:, 0:1], target[:, 1:] - target[:, 0:1])
        
        return relative_loss * 10 + first_frame_loss

    def compute_contrastive_loss(self, clip_pred, clip_target):
        pass

    @staticmethod
    def _find_label_indices(all_embeds, target_embeds):
        distances = (target_embeds.unsqueeze(1) - all_embeds.unsqueeze(0)).pow(2).sum(-1)
        label_idx = distances.argmin(dim=1)
        return label_idx
    
    @staticmethod
    def compute_clip_loss(pred_embedding, target_embedding):
        similarity = cosine_similarity(pred_embedding, target_embedding)
        return 1 - similarity.mean()