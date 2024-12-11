import torch
from torch.nn.functional import mse_loss, cosine_similarity, cross_entropy
from .angle_loss import AngleLoss

class CameraTrajectoryLoss:
    def __init__(self, clip_embeddings, convtrastive=False):
        self.angle_loss = AngleLoss()
        self.position_slice = slice(0, 4)
        self.rotation_slice = slice(4, None)
        self.convtrastive = convtrastive

        self.clip_embeddings = clip_embeddings
        self.all_categories = {}
        if self.convtrastive:
            device = next(iter(clip_embeddings['movement'].values())).device
            for cat in ['movement', 'easing', 'angle', 'shot']:
                keys = list(self.clip_embeddings[cat].keys())
                embeds = torch.stack([self.clip_embeddings[cat][k].squeeze(0) for k in keys]).to(device)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                self.all_categories[cat] = (keys, embeds)


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
        if self.convtrastive:
            clip_loss, clip_losses = self.compute_contrastive_loss(clip_pred, clip_target)
        else:
            clip_losses = {
                key: self.compute_clip_loss(clip_pred[key], clip_target[key]) * 100
                for key in clip_pred.keys()
            }
            clip_loss = sum(clip_losses.values())
        total_loss = trajectory_loss + clip_loss
        loss_dict = {
            'trajectory': trajectory_loss.item(),
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

    def compute_trajectory_loss(self, pred, target):
        # return self.compute_component_losses(pred, target)
        first_frame_loss = self.compute_component_losses(pred[:, 0:1], target[:, 0:1])
        relative_loss = self.compute_component_losses(pred[:, 1:] - pred[:, 0:1], target[:, 1:] - target[:, 0:1])
        
        return relative_loss + first_frame_loss

    def compute_contrastive_loss(self, clip_pred, clip_target):
        losses = {}
        
        for cat in ['movement', 'easing', 'angle', 'shot']:
            pred = clip_pred[cat]
            target_emb = clip_target[cat]
            keys, class_embeds = self.all_categories[cat]
            pred = pred / pred.norm(dim=-1, keepdim=True)
            label_idx = self._find_label_indices(class_embeds, target_emb)
            similarity = torch.matmul(pred, class_embeds.T)
            
            losses[cat] = cross_entropy(similarity, label_idx)

        total_loss = sum(losses.values())
        return total_loss, losses

    @staticmethod
    def _find_label_indices(all_embeds, target_embeds):
        distances = (target_embeds.unsqueeze(1) - all_embeds.unsqueeze(0)).pow(2).sum(-1)
        label_idx = distances.argmin(dim=1)
        return label_idx
    
    @staticmethod
    def compute_clip_loss(pred_embedding, target_embedding):
        similarity = cosine_similarity(pred_embedding, target_embedding)
        return 1 - similarity.mean()