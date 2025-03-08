import torch
from torch.nn.functional import mse_loss, cosine_similarity
from .angle_loss import AngleLoss

class CameraTrajectoryLoss:
    def __init__(self, 
                 contrastive_loss_margin: int=5, 
                 n_clip_embs: int=28, 
                 losses_list: list=[], 
                 weighted_clip_loss: bool=False,
                 weight_power: int=1,
                 clip_weights: dict=None,
                 clip_loss_scaling_factor: float=37500,
                 trajectory_loss_ratio: float=10,
                 contrastive_loss_scaling_factor: float=0.1,
                 angle_loss_scaling_factor: float=180
                 ):
        self.angle_loss = AngleLoss(angle_loss_scaling_factor)
        self.position_slice = slice(0, 4)
        self.rotation_slice = slice(4, None)
        self.contrastive_loss_margin = contrastive_loss_margin
        self.n_clip_embs = n_clip_embs
        self.losses_list = losses_list
        self.weighted_clip_loss = weighted_clip_loss
        self.weight_power = weight_power
        self.clip_weights = clip_weights
        self.sum_clip_weights = 0
        self.clip_loss_scaling_factor = clip_loss_scaling_factor
        self.trajectory_loss_ratio = trajectory_loss_ratio
        self.contrastive_loss_scaling_factor = contrastive_loss_scaling_factor
        
        if self.weighted_clip_loss:
            for embedding, weight in self.clip_weights.items():
                if self.weight_power > 1:
                    self.clip_weights[embedding] = weight ** self.weight_power
                self.sum_clip_weights += self.clip_weights[embedding]

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
        if len(self.losses_list) == 0:
            raise ValueError("The losses list cannot be empty; it must contain at least one of the following: 'trajectory', 'clip', or 'contrastive'.")
        
        total_loss = 0
        loss_dict = dict()

        if "trajectory" in self.losses_list:
            trajectory_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)
            total_loss += trajectory_loss
            loss_dict["trajectory"] = trajectory_loss.item()
        
        if "contrastive" in self.losses_list:
            contrastive_loss, _ = self.compute_contrastive_loss(clip_target, clip_pred, self.n_clip_embs, self.contrastive_loss_margin)
            total_loss += contrastive_loss * self.contrastive_loss_scaling_factor
            loss_dict["contrastive"] = contrastive_loss.item()

        if "clip" in self.losses_list:
            total_clip_loss_weighted, clip_losses, total_clip_loss = self.compute_clip_loss(clip_target, clip_pred, self.n_clip_embs, self.weighted_clip_loss, self.clip_weights, self.sum_clip_weights)
            if self.weighted_clip_loss:
                total_loss += total_clip_loss_weighted / clip_pred.shape[1] * self.clip_loss_scaling_factor
            else:
                total_loss += total_clip_loss / clip_pred.shape[1] * self.clip_loss_scaling_factor
            loss_dict["clip"] = {i: clip_losses[i] for i in range(self.n_clip_embs)}
            loss_dict["average_clip"] = total_clip_loss

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
    
    def compute_trajectory_only_loss(self, model_output, camera_trajectory, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']
        trajectory_loss = self.compute_trajectory_loss(reconstructed, camera_trajectory)
        
        loss_dict = {
            "trajectory": trajectory_loss.item(),
            "total": trajectory_loss.item()
        }
        
        return trajectory_loss, loss_dict

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
        
        return relative_loss * self.trajectory_loss_ratio + first_frame_loss

    @staticmethod
    def compute_contrastive_loss(clip_target, clip_pred, n_clip_embs, margin):
        contrastive_losses = []
        total_contrastive_loss = 0
        for i in range(n_clip_embs): # Target
            current_loss = 0
            for j in range(n_clip_embs): # Predicted
                if i == j:
                    current_loss += max(0, margin - torch.sqrt(mse_loss(clip_target[i], clip_pred[j]))) ** 2
                else:
                    current_loss += mse_loss(clip_target[i], clip_pred[j])
            contrastive_losses.append(current_loss)
            total_contrastive_loss += current_loss
        return total_contrastive_loss, contrastive_losses

    
    @staticmethod
    def compute_clip_loss(clip_target, clip_pred, n_clip_embs, weighted_clip_loss, clip_weights, sum_clip_weights):
        clip_losses = []
        total_clip_loss = 0
        total_clip_loss_weighted = 0
        for i in range(n_clip_embs):
            similarity = cosine_similarity(clip_target[i], clip_pred[i])
            current_loss = 1 - similarity.mean()
            clip_losses.append(current_loss) 
            if weighted_clip_loss:
                total_clip_loss_weighted += current_loss * clip_weights[f"clip_{i}"]
            total_clip_loss += current_loss
        if weighted_clip_loss:
            total_clip_loss_weighted = total_clip_loss_weighted / sum_clip_weights
        total_clip_loss = total_clip_loss / n_clip_embs
        return total_clip_loss_weighted, clip_losses, total_clip_loss
    

    @staticmethod
    def _find_label_indices(all_embeds, target_embeds):
        distances = (target_embeds.unsqueeze(1) - all_embeds.unsqueeze(0)).pow(2).sum(-1)
        label_idx = distances.argmin(dim=1)
        return label_idx