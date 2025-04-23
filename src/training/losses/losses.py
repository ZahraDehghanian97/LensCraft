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
                 contrastive_loss_version: int=1,
                 clip_loss_scaling_factor: float=37500,
                 trajectory_loss_scaling_factor: float=10,
                 contrastive_loss_scaling_factor: float=0.1,
                 angle_loss_scaling_factor: float=180,
                 clip_embeddings: dict=None,
                 encoder_loss_function: str="clip"
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

        self.contrastive_loss_version = contrastive_loss_version
        if self.contrastive_loss_version == 1:
            self.compute_contrastive_loss = self.compute_contrastive_loss_v1
        elif self.contrastive_loss_version == 2:
            self.compute_contrastive_loss = self.compute_contrastive_loss_v2
        else:
            raise ValueError("Contrastive loss version should be 1 or 2, you passed {}".format(self.contrastive_loss_version))

        self.clip_loss_scaling_factor = clip_loss_scaling_factor
        self.trajectory_loss_scaling_factor = trajectory_loss_scaling_factor
        self.contrastive_loss_scaling_factor = contrastive_loss_scaling_factor

        self.encoder_loss_function = encoder_loss_function
        self.clip_embeddings = clip_embeddings

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.weighted_clip_loss:
            for embedding, weight in self.clip_weights.items():
                if self.weight_power > 1:
                    self.clip_weights[embedding] = weight ** self.weight_power
                self.sum_clip_weights += self.clip_weights[embedding]



    def __call__(self, model_output, camera_trajectory, clip_target, batch, tgt_key_padding_mask=None):
        reconstructed = model_output['reconstructed']
        clip_pred = model_output['embeddings']
        
        return self.compute_total_loss(
            reconstructed,
            camera_trajectory,
            clip_pred,
            clip_target,
            batch
        )



    def compute_total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target, batch):
        if len(self.losses_list) == 0:
            raise ValueError("The losses list cannot be empty; it must contain at least one of the following: 'trajectory', 'clip', or 'contrastive'.")
        
        total_loss = 0
        loss_dict = dict()

        if "clip" in self.losses_list:
            total_clip_loss_weighted, clip_losses, total_clip_loss = self.compute_clip_loss(
                clip_target=clip_target, 
                clip_pred=clip_pred, 
                n_clip_embs=self.n_clip_embs, 
                weighted_clip_loss=self.weighted_clip_loss, 
                clip_weights=self.clip_weights, 
                sum_clip_weights=self.sum_clip_weights,
                encoder_loss_function=self.encoder_loss_function,
                batch=batch)
            if self.weighted_clip_loss:
                total_loss += total_clip_loss_weighted / clip_pred.shape[1] * self.clip_loss_scaling_factor
            else:
                if self.encoder_loss_function == "clip":
                    total_loss += total_clip_loss / clip_pred.shape[1] * self.clip_loss_scaling_factor
                elif self.encoder_loss_function == "mse":
                    total_loss += total_clip_loss / clip_pred.shape[1] * 12800

            loss_dict["clip"] = {i: clip_losses[i] for i in range(self.n_clip_embs)}
            loss_dict["average_clip"] = total_clip_loss * 200
            print("CLIP LOSS (NET): {:.3f}   |   CLIP LOSS (SCALED): {:.3f}".format(total_clip_loss, total_clip_loss * self.clip_loss_scaling_factor / clip_pred.shape[1]))

        if "trajectory" in self.losses_list:
            trajectory_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)
            total_loss += trajectory_loss * self.trajectory_loss_scaling_factor
            loss_dict["trajectory"] = trajectory_loss.item()
            print("TRAJ LOSS (NET): {:.3f}   |   TRAJ LOSS (SCALED): {:.3f}".format(trajectory_loss, trajectory_loss * self.trajectory_loss_scaling_factor))
        
        if "contrastive" in self.losses_list:
            contrastive_loss = self.compute_contrastive_loss(clip_pred, clip_target, batch)
            total_loss += contrastive_loss * self.contrastive_loss_scaling_factor
            loss_dict["contrastive"] = contrastive_loss.item()
            print("CONT Loss (NET): {:.3f}   |   CONT Loss (SCALED): {:.3f}".format(contrastive_loss, contrastive_loss * self.contrastive_loss_scaling_factor))

        print("TOTL LOSS: {:.3f}".format(total_loss))

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
        
        return relative_loss * self.trajectory_loss_scaling_factor + first_frame_loss
    


    @staticmethod
    def get_embedding_name(name: str) -> str:
        embedding_name = str(name).split(".")[-1].lower()
        embedding_name = embedding_name.split("_")
        embedding_name = embedding_name[0] + "".join([item.capitalize() for item in embedding_name[1:]])
        return embedding_name
    


    def modify_sample(self, 
                      clip_embedding_parameters: list[torch.tensor], 
                      clip_sample_target: torch.tensor, 
                      n_modification: int) -> torch.tensor:

        n_embeddings = len(clip_embedding_parameters)
        indices_to_modify = np.random.permutation(n_embeddings)[:n_modification]
        modified_sample = clip_sample_target.clone()

        for index, (parameter, value, value_index, embedding) in enumerate(clip_embedding_parameters):
            if index in indices_to_modify and value_index != -1:

                if parameter.count("_") > 1: 
                    parameter = "_".join(parameter.split("_")[-2:])
                embedding_type_enum = CLIP_PARAMETERS_DICT[parameter]
                embedding_type_name = embedding_type_enum.__name__
                n_embedding_values = len(embedding_type_enum)
                
                random_index = random.randint(0, n_embedding_values - 1)
                if random_index == value_index:
                    random_index = (random_index + 1) % n_embedding_values

                random_embedding_name = list(embedding_type_enum)[random_index]
                random_embedding_name = self.get_embedding_name(random_embedding_name)
                modified_sample[index] = self.clip_embeddings[embedding_type_name][random_embedding_name]
        
        return modified_sample



    def compute_contrastive_loss_v1(self, clip_pred, clip_target, batch):
        N_SIMILAR_SAMPLES = 3
        N_DISSIMILAR_SAMPLES = 3

        batch_size = clip_pred.shape[1]
        contrastive_loss = 0

        for sample_idx in range(batch_size):
            clip_sample_pred = clip_pred[:, sample_idx, :]
            clip_sample_target = clip_target[:, sample_idx, :]

            clip_embedding_parameters = batch["cinematography_prompt_parameters"][sample_idx] +\
                                        batch["simulation_instruction_parameters"][sample_idx]

            for _ in range(N_SIMILAR_SAMPLES): 
                n_modification = random.randint(1, 4)
                clip_sample_similar = self.modify_sample(clip_embedding_parameters, 
                                                         clip_sample_target, 
                                                         n_modification).to(self.device)
                similarity = cosine_similarity(clip_sample_pred, clip_sample_similar).mean()
                contrastive_loss += 1 - similarity


            for _ in range(N_DISSIMILAR_SAMPLES):
                n_modification = random.randint(25, 28)
                clip_sample_dissimilar = self.modify_sample(clip_embedding_parameters, 
                                                            clip_sample_target, 
                                                            n_modification).to(self.device)
                similarity = cosine_similarity(clip_sample_pred, clip_sample_dissimilar).mean()
                contrastive_loss += similarity + 1

        return contrastive_loss
    


    def compute_contrastive_loss_v2(self, clip_pred, clip_target, batch):
        contrastive_loss = 0

        batch_size = len(batch["camera_trajectory"])
        
        for sample_idx in range(batch_size):
            clip_sample_pred = clip_pred[:, sample_idx, :]
            clip_embedding_parameters = batch["cinematography_prompt_parameters"][sample_idx] +\
                                        batch["simulation_instruction_parameters"][sample_idx]
            for emb_idx in range(len(clip_embedding_parameters)):
                prefix, data_value, value_idx, _ = clip_embedding_parameters[emb_idx]

                if value_idx == -1:
                    continue

                if prefix.count("_") > 1:
                    prefix = "_".join(prefix.split("_")[-2:])
                
                if CLIP_PARAMETERS_DICT[prefix].__name__ == "bool":
                    value_type = "boolean" 
                else:
                    value_type = CLIP_PARAMETERS_DICT[prefix].__name__
                
                for embedding_key in self.clip_embeddings[value_type].keys():
                    if embedding_key != data_value:
                        similarity = cosine_similarity(clip_sample_pred[emb_idx].unsqueeze(0).to(self.device), 
                                        self.clip_embeddings[value_type][embedding_key].unsqueeze(0).to(self.device)).mean()
                        contrastive_loss += similarity + 1
                        
        return torch.tensor(contrastive_loss).clone().detach()


    
    def compute_clip_loss(self,
                          clip_target, 
                          clip_pred, 
                          n_clip_embs, 
                          weighted_clip_loss, 
                          clip_weights, 
                          sum_clip_weights, 
                          encoder_loss_function, 
                          batch):   
        none_mask = self.create_none_mask_matrix(batch, n_clip_embs)
        clip_losses = []
        total_clip_loss = 0
        total_clip_loss_weighted = 0
        for i in range(n_clip_embs):
            embedding_none_mask = none_mask[:, i]
            if encoder_loss_function == "clip":
                similarity = cosine_similarity(clip_target[i], clip_pred[i])
                current_loss = 1 - similarity[embedding_none_mask].mean()
            elif encoder_loss_function == "mse":
                current_loss = mse_loss(clip_target[i], clip_pred[i])
            clip_losses.append(current_loss) 
            if weighted_clip_loss:
                total_clip_loss_weighted += current_loss * clip_weights[f"clip_{i}"]
            total_clip_loss += current_loss
        if weighted_clip_loss:
            total_clip_loss_weighted = total_clip_loss_weighted / sum_clip_weights
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
    


    @staticmethod
    def _find_label_indices(all_embeds, target_embeds):
        distances = (target_embeds.unsqueeze(1) - all_embeds.unsqueeze(0)).pow(2).sum(-1)
        label_idx = distances.argmin(dim=1)
        return label_idx