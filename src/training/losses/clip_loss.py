import torch
from torch.nn.functional import cosine_similarity


class ClipLoss:
    def __init__(self, clip_weights=None, weight_power=1):
        self.clip_weights = clip_weights
        self.weight_power = weight_power

        if self.clip_weights:
            for embedding, weight in self.clip_weights.items():
                if self.weight_power > 1:
                    self.clip_weights[embedding] = weight ** self.weight_power

    def compute(self,
                clip_target,
                clip_pred,
                weighted_clip_loss,
                prompt_none_mask=None,
                encoder_loss_function="clip"):
        if encoder_loss_function == "clip":
            values = cosine_similarity(clip_target, clip_pred, dim=-1)
        elif encoder_loss_function == "mse":
            values = ((clip_target - clip_pred) ** 2).mean(dim=-1)
        else:
            raise ValueError(
                f"Unsupported encoder_loss_function: {encoder_loss_function}"
            )

        if prompt_none_mask is None:
            if values.shape[1] == 0:
                losses = clip_pred.new_zeros(clip_target.shape[0])
            else:
                means = values.mean(dim=1)
                losses = 1 - means if encoder_loss_function == "clip" else means
        else:
            valid = prompt_none_mask.transpose(0, 1)
            counts = valid.sum(dim=1)
            means = values.masked_fill(~valid, 0.0).sum(dim=1) / counts.clamp_min(1)
            losses = 1 - means if encoder_loss_function == "clip" else means
            losses = torch.where(counts > 0, losses, torch.zeros_like(losses))

        clip_losses = list(losses.unbind())
        if weighted_clip_loss and self.clip_weights:
            weights = [
                self.clip_weights.get(f"clip_{i}", 1.0)
                for i in range(clip_target.shape[0])
            ]
            applied_clip_weight = sum(weights)
            if applied_clip_weight > 0:
                return clip_losses, (
                    losses * losses.new_tensor(weights)
                ).sum() / applied_clip_weight

        return clip_losses, losses.mean()
