import torch
from torch.nn.functional import mse_loss, cosine_similarity
from .angle_loss import AngleLoss
import random
import numpy as np
from data.simulation.constants import CLIP_PARAMETERS_DICT

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
                 angle_loss_scaling_factor: float=180,
                 clip_embeddings: dict=None
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
        self.clip_embeddings = clip_embeddings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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

    def __call__(self, model_output, camera_trajectory, clip_targets, batch, tgt_key_padding_mask=None):
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
            clip_targets,
            batch
        )

    def compute_total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target, batch):
        if len(self.losses_list) == 0:
            raise ValueError("The losses list cannot be empty; it must contain at least one of the following: 'trajectory', 'clip', or 'contrastive'.")
        
        total_loss = 0
        loss_dict = dict()

        if "trajectory" in self.losses_list:
            trajectory_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)
            total_loss += trajectory_loss
            loss_dict["trajectory"] = trajectory_loss.item()
        
        if "contrastive" in self.losses_list:
            contrastive_loss = self.compute_contrastive_loss(clip_target, clip_pred, self.n_clip_embs, self.contrastive_loss_margin, batch)
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

            print("CLIP LOSS:            {}".format(total_clip_loss))
            print("Contrastive Loss:     {}".format(contrastive_loss))

        loss_dict["total"] = total_loss.item()
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
        
        return relative_loss * self.trajectory_loss_ratio + first_frame_loss
    
    @staticmethod
    def get_embedding_name(name: str) -> str:
        embedding_name = str(name).split(".")[-1].lower()
        embedding_name = embedding_name.split("_")
        embedding_name = embedding_name[0] + "".join([item.capitalize() for item in embedding_name[1:]])
        return embedding_name
    
    def modify_sample(self, clip_embedding_parameters: list[torch.tensor], n_modification: int) -> torch.tensor:
        n_embeddings = len(clip_embedding_parameters)
        embedding_dim = len(clip_embedding_parameters[0][-1])
        indices_to_modify = np.random.choice(range(0, n_embeddings), n_modification, replace=False)
        modified_sample = torch.full((n_embeddings, embedding_dim), -1, dtype=torch.float)
        for index, (parameter, value, value_index, embedding) in enumerate(clip_embedding_parameters):
            if embedding is not None:
                if index in indices_to_modify:
                    if parameter.count("_") > 1: 
                        parameter = "_".join(parameter.split("_")[-2:])
                    embedding_type_enum = CLIP_PARAMETERS_DICT[parameter]
                    embedding_type_name = embedding_type_enum.__name__
                    n_embedding_values = len(embedding_type_enum)
                    valid_indices = np.setdiff1d(np.arange(0, n_embedding_values), [value_index])
                    random_index = np.random.choice(valid_indices)
                    random_embedding_name = list(embedding_type_enum)[random_index]
                    random_embedding_name = self.get_embedding_name(random_embedding_name)
                    random_embedding_vector = self.clip_embeddings[embedding_type_name][random_embedding_name]
                    modified_sample[index] = random_embedding_vector
                else:
                    modified_sample[index] = embedding
        return modified_sample




    def compute_contrastive_loss(self, clip_target, clip_pred, n_clip_embs, margin, batch):
        MIN_EMB_MOD_SIMILAR, MAX_EMB_MOD_SIMILAR = 1, 5
        MIN_EMB_MOD_DISSIMILAR, MAX_EMB_MOD_DISSIMILAR = 24, 28

        MIN_SIMILAR_SAMPLES, MAX_SIMILAR_SAMPLES = 1, 5
        MIN_DISSIMILAR_SAMPLES, MAX_DISSIMILAR_SAMPLES = 1, 5

        N_SIMILAR_SAMPLES = random.randint(MIN_SIMILAR_SAMPLES, MAX_SIMILAR_SAMPLES)
        N_DISSIMILAR_SAMPLES = random.randint(MIN_DISSIMILAR_SAMPLES, MAX_DISSIMILAR_SAMPLES)

        batch_size = clip_target.shape[1]
        total_contrastive_loss = 0

        for sample_idx in range(batch_size):
            clip_sample_target = clip_target[:, sample_idx, :]
            clip_sample_pred = clip_pred[:, sample_idx, :]

            clip_embedding_parameters = batch["cinematography_prompt_parameters"][sample_idx] +\
                                        batch["simulation_instruction_parameters"][sample_idx]
            
            for _ in range(N_SIMILAR_SAMPLES): # Loop to compute loss of similar samples
                n_modification = random.randint(MIN_EMB_MOD_SIMILAR, MAX_EMB_MOD_SIMILAR)
                clip_sample_similar = self.modify_sample(clip_embedding_parameters, n_modification).to(self.device)
                similarity = cosine_similarity(clip_sample_pred, clip_sample_similar).mean()
                loss = 1 - similarity
                total_contrastive_loss += loss

            for _ in range(N_DISSIMILAR_SAMPLES): # Loop to compute loss of dissimilar samples
                n_modification = random.randint(MIN_EMB_MOD_DISSIMILAR, MAX_EMB_MOD_DISSIMILAR)
                clip_sample_dissimilar = self.modify_sample(clip_embedding_parameters, n_modification).to(self.device)
                similarity = cosine_similarity(clip_sample_pred, clip_sample_dissimilar).mean()
                loss = -(1 - similarity) # FIXME: possible to get negative loss
                total_contrastive_loss += loss
        return total_contrastive_loss

    
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