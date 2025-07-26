import torch
from torch.nn.functional import mse_loss, cosine_similarity


class ClipLoss:
    def __init__(self, clip_weights=None, weight_power=1):
        self.clip_weights = clip_weights
        self.weight_power = weight_power
        self.sum_clip_weights = 0
        
        if self.clip_weights:
            for embedding, weight in self.clip_weights.items():
                if self.weight_power > 1:
                    self.clip_weights[embedding] = weight ** self.weight_power
                self.sum_clip_weights += self.clip_weights[embedding]
    
    def compute(self,
                clip_target,
                clip_pred,
                weighted_clip_loss,
                prompt_none_mask=None,
                encoder_loss_function="clip"):
        clip_losses = []
        total_clip_loss = 0
        total_clip_loss_weighted = 0
        
        for i in range(clip_target.shape[0]):
            if encoder_loss_function == "clip":
                similarity = cosine_similarity(clip_target[i], clip_pred[i])
                if prompt_none_mask is not None:
                    similarity = similarity[prompt_none_mask[:, i]]
                
                if len(similarity) != 0:
                    current_loss = 1 - similarity.mean()
                else:
                    current_loss = 0
            elif encoder_loss_function == "mse":
                current_loss = mse_loss(clip_target[i], clip_pred[i])
            
            clip_losses.append(current_loss)
            
            if weighted_clip_loss and self.clip_weights:
                total_clip_loss_weighted += current_loss * self.clip_weights[f"clip_{i}"]
            
            total_clip_loss += current_loss

        
        if weighted_clip_loss and self.sum_clip_weights > 0:
            total_clip_loss_weighted = total_clip_loss_weighted / self.sum_clip_weights
            return clip_losses, total_clip_loss_weighted

        total_clip_loss = total_clip_loss / clip_target.shape[0]
        return clip_losses, total_clip_loss
