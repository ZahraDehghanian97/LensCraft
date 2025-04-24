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
                n_clip_embs, 
                weighted_clip_loss, 
                batch,  
                encoder_loss_function="clip"):
        none_mask = self.create_none_mask_matrix(batch, n_clip_embs)
        clip_losses = []
        total_clip_loss = 0
        total_clip_loss_weighted = 0
        
        for i in range(n_clip_embs):
            embedding_none_mask = none_mask[:, i]
            if encoder_loss_function == "clip":
                similarity = cosine_similarity(clip_target[i], clip_pred[i])[embedding_none_mask]
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
        
        total_clip_loss = total_clip_loss / n_clip_embs
        
        return total_clip_loss_weighted, clip_losses, total_clip_loss
    
    @staticmethod
    def create_none_mask_matrix(batch: dict, n_clip_embs: int):
        batch_size = len(batch["camera_trajectory"])
        none_entries = torch.ones(batch_size, n_clip_embs, dtype=torch.bool)
        
        for sample_idx in range(batch_size):
            clip_embedding_parameters = batch["cinematography_prompt_parameters"][sample_idx] +\
                                        batch["simulation_instruction_parameters"][sample_idx]
            
            for emb_idx in range(n_clip_embs):
                _, _, value_idx, _ = clip_embedding_parameters[emb_idx]
                if value_idx == -1:
                    none_entries[sample_idx, emb_idx] = False
        
        return none_entries
