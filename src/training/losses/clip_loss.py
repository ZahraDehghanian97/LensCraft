from torch.nn.functional import cosine_similarity


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
        applied_clip_weight = 0
        
        for i in range(clip_target.shape[0]):
            if encoder_loss_function == "clip":
                similarity = cosine_similarity(clip_target[i], clip_pred[i])
                if prompt_none_mask is not None:
                    similarity = similarity[prompt_none_mask[:, i]]
                
                if len(similarity) != 0:
                    current_loss = 1 - similarity.mean()
                else:
                    current_loss = clip_pred[i].new_zeros(())
            elif encoder_loss_function == "mse":
                per_sample_loss = (
                    (clip_target[i] - clip_pred[i]) ** 2
                ).mean(dim=-1)
                if prompt_none_mask is not None:
                    per_sample_loss = per_sample_loss[prompt_none_mask[:, i]]
                current_loss = (
                    per_sample_loss.mean()
                    if len(per_sample_loss) != 0
                    else clip_pred[i].new_zeros(())
                )
            else:
                raise ValueError(
                    f"Unsupported encoder_loss_function: {encoder_loss_function}"
                )
            
            clip_losses.append(current_loss)
            
            if weighted_clip_loss and self.clip_weights:
                # New serialized constraints add tokens beyond legacy weight
                # files. Give unspecified tokens a neutral weight instead of
                # dropping them or failing with KeyError.
                weight = self.clip_weights.get(f"clip_{i}", 1.0)
                total_clip_loss_weighted += current_loss * weight
                applied_clip_weight += weight
            
            total_clip_loss += current_loss

        
        if weighted_clip_loss and applied_clip_weight > 0:
            total_clip_loss_weighted = total_clip_loss_weighted / applied_clip_weight
            return clip_losses, total_clip_loss_weighted

        total_clip_loss = total_clip_loss / clip_target.shape[0]
        return clip_losses, total_clip_loss
