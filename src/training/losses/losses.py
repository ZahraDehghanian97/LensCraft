import torch
from torch.nn.functional import mse_loss, cosine_similarity, cross_entropy
from .angle_loss import AngleLoss

class CameraTrajectoryLoss:
    def __init__(self, clip_embeddings=[], contrastive=False):
        self.angle_loss = AngleLoss()
        self.position_slice = slice(0, 4)
        self.rotation_slice = slice(4, None)
        self.contrastive = contrastive

        self.clip_embeddings = clip_embeddings
        self.all_categories = {}
        if self.contrastive:
            device = next(iter(clip_embeddings['movement'].values())).device
            for cat in ['movement', 'easing', 'angle', 'shot']:
                keys = list(self.clip_embeddings[cat].keys())
                embeds = torch.stack([self.clip_embeddings[cat][k].squeeze(0) for k in keys]).to(device)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                self.all_categories[cat] = (keys, embeds)

    def __call__(self, model_output, camera_trajectory, clip_targets, embedding_masks=None, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']
        
        # if tgt_key_padding_mask is not None: TODO: fix dimensions
        #     # Apply padding mask to both reconstructed and target trajectories
        #     valid_mask = ~tgt_key_padding_mask
        #     reconstructed = reconstructed * valid_mask.unsqueeze(-1)
        #     camera_trajectory = camera_trajectory * valid_mask.unsqueeze(-1)

        clip_embeddings = {
            k: model_output[f'{k}_embedding'] for k in clip_targets.keys()
        }

        return self.compute_total_loss(
            reconstructed, 
            camera_trajectory, 
            clip_embeddings, 
            clip_targets,
            embedding_masks
        )

    def compute_total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target, embedding_masks=None):
        trajectory_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)
        
        if self.contrastive:
            clip_loss, clip_losses = self.compute_contrastive_loss(clip_pred, clip_target, embedding_masks)
        else:
            clip_losses = {}
            total_clip_loss = 0
            num_valid_embeddings = 0
            
            for key in clip_pred.keys():
                if embedding_masks is None or key not in embedding_masks:
                    current_loss = self.compute_clip_loss(clip_pred[key], clip_target[key]) * 100
                    clip_losses[key] = current_loss
                    total_clip_loss += current_loss
                    num_valid_embeddings += 1
                else:
                    mask = embedding_masks[key]
                    masked_pred = clip_pred[key][mask]
                    masked_target = clip_target[key][mask]
                    if masked_pred.shape[0] != 0:
                        current_loss = self.compute_clip_loss(masked_pred, masked_target) * 100
                        clip_losses[key] = current_loss
                        total_clip_loss += current_loss
                        num_valid_embeddings += 1
            
            clip_loss = total_clip_loss / max(num_valid_embeddings, 1)

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
        first_frame_loss = self.compute_component_losses(pred[:, 0:1], target[:, 0:1])
        relative_loss = self.compute_component_losses(pred[:, 1:] - pred[:, 0:1], target[:, 1:] - target[:, 0:1])
        
        return relative_loss * 10 + first_frame_loss

    def compute_contrastive_loss(self, clip_pred, clip_target, embedding_masks=None):
        losses = {}
        total_loss = 0
        num_valid_categories = 0
        
        for cat in ['movement', 'easing', 'angle', 'shot']:
            if embedding_masks is not None and cat in embedding_masks and not embedding_masks[cat].any():
                losses[cat] = torch.tensor(0.0, device=clip_pred[cat].device)
                continue
                
            pred = clip_pred[cat]
            target_emb = clip_target[cat]
            
            if embedding_masks is not None and cat in embedding_masks:
                mask = embedding_masks[cat]
                pred = pred[mask]
                target_emb = target_emb[mask]
                
                if len(pred) == 0:
                    losses[cat] = torch.tensor(0.0, device=clip_pred[cat].device)
                    continue
            
            keys, class_embeds = self.all_categories[cat]
            pred = pred / pred.norm(dim=-1, keepdim=True)
            label_idx = self._find_label_indices(class_embeds, target_emb)
            similarity = torch.matmul(pred, class_embeds.T)
            
            current_loss = cross_entropy(similarity, label_idx)
            losses[cat] = current_loss
            total_loss += current_loss
            num_valid_categories += 1

        total_loss = total_loss / max(num_valid_categories, 1)
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