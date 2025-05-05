import torch
from torch.nn.functional import cosine_similarity
import random
import numpy as np
from data.simulation.constants import CLIP_PARAMETERS_DICT


class ContrastiveLoss:
    def __init__(self, clip_embeddings, device, embedding_means=None, get_embedding_name_func=None):
        self.clip_embeddings = clip_embeddings
        self.device = device
        self.embedding_means = embedding_means
        self.get_embedding_name = get_embedding_name_func
        
        if self.get_embedding_name is None:
            self.get_embedding_name = self._default_get_embedding_name
    
    @staticmethod
    def _default_get_embedding_name(name: str) -> str:
        embedding_name = str(name).split(".")[-1].lower()
        embedding_name = embedding_name.split("_")
        embedding_name = embedding_name[0] + "".join([item.capitalize() for item in embedding_name[1:]])
        return embedding_name
    
    def _modify_sample(self,
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
    
    def compute_v1(self, clip_pred, clip_target, batch):
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
                clip_sample_similar = self._modify_sample(clip_embedding_parameters,
                                                         clip_sample_target,
                                                         n_modification).to(self.device)
                similarity = cosine_similarity(clip_sample_pred, clip_sample_similar).mean()
                contrastive_loss += 1 - similarity


            for _ in range(N_DISSIMILAR_SAMPLES):
                n_modification = random.randint(25, 28)
                clip_sample_dissimilar = self._modify_sample(clip_embedding_parameters,
                                                            clip_sample_target,
                                                            n_modification).to(self.device)
                similarity = cosine_similarity(clip_sample_pred, clip_sample_dissimilar).mean()
                contrastive_loss += similarity + 1

        return contrastive_loss
    
    def compute_v2(self, clip_pred, clip_target, batch):
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

    def compute_v3(self, clip_pred, clip_target, batch):
        contrastive_loss = 0
        parameters_size = clip_pred.shape[0]
        batch_size = clip_pred.shape[1]
        
        for sample_idx in range(batch_size):
            clip_sample_pred = clip_pred[:, sample_idx, :]
            clip_embedding_parameters = batch["cinematography_prompt_parameters"][sample_idx] + \
                                    batch["simulation_instruction_parameters"][sample_idx]
            
            for emb_idx, (prefix, data_value, value_idx, _) in enumerate(clip_embedding_parameters):
                if value_idx == -1:
                    continue
                    
                if prefix.count("_") > 1:
                    prefix = "_".join(prefix.split("_")[-2:])
                    
                if CLIP_PARAMETERS_DICT[prefix].__name__ == "bool":
                    value_type = "boolean"
                else:
                    value_type = CLIP_PARAMETERS_DICT[prefix].__name__
                    
                mean_embedding = self.embedding_means[value_type].to(self.device)
                similarity = cosine_similarity(
                    clip_sample_pred[emb_idx].unsqueeze(0), 
                    mean_embedding.unsqueeze(0)
                ).mean()
                contrastive_loss += similarity + 1
                
        return torch.tensor(contrastive_loss / (batch_size * parameters_size)).clone().detach()
