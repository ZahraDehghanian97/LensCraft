from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import torch
import pickle


def get_enum_index(enum_class, value) -> int:
    if isinstance(enum_class, bool) or enum_class is bool:
        if isinstance(value, bool):
            return 1 if value else 0
        return -1
        
    if isinstance(enum_class, type(Enum)):
        try:
            return list(enum_class).index(enum_class(value))
        except (ValueError, KeyError):
            return -1
            
    return -1



def extract_cinematography_parameters(
    data: Dict,
    struct: List,
    clip_embeddings: Optional[Dict] = None,
    prefix: str = "",
    fill_none_with_mean: bool = False,
    embedding_means = None,
) -> List[Tuple[str, Any, int, Optional[torch.Tensor]]]:
    parameters = []
    
    for key, value_type in struct:
        current_prefix = f"{prefix}_{key}" if prefix else key
        data_value = data.get(key, None)
        embedding = None
        
        if isinstance(value_type, type) and issubclass(value_type, Enum):
            index = get_enum_index(value_type, data_value)

            if index == -1 and fill_none_with_mean:
                embedding = get_mean_embedding(value_type, embedding_means)

            if data_value:
                embedding = clip_embeddings[value_type.__name__][data_value]

            parameters.append((current_prefix, data_value, index, embedding))
            
        elif value_type is bool:
            if data_value is None:
                index = -1
                if fill_none_with_mean:
                    embedding = get_mean_embedding(value_type, embedding_means)

            elif isinstance(data_value, bool):
                index = 1 if data_value else 0
                embedding = clip_embeddings["boolean"][data_value]
                
            parameters.append((current_prefix, data_value, index, embedding))
            
        elif isinstance(value_type, list):
            if data_value is None:
                nested_data = {}

            elif isinstance(data_value, dict):
                nested_data = data_value

            else:
                nested_data = {}

            nested_params = extract_cinematography_parameters(
                data=nested_data,
                struct=value_type,
                clip_embeddings=clip_embeddings,
                fill_none_with_mean=fill_none_with_mean,
                prefix=current_prefix,
                embedding_means=embedding_means,
            )
            parameters.extend(nested_params)
                
    return parameters



def count_total_parameters_in_struct(struct: List) -> int:
    count = 0
    
    for _, value_type in struct:
        if isinstance(value_type, list):
            count += count_total_parameters_in_struct(value_type)
        else:
            count += 1
            
    return count



def flatten_struct_parameters(struct: list, last_value=None) -> list:
    parameter_list = list()
    for parameter, value_type in struct:
        if isinstance(value_type, list):
            parameter_list.extend(flatten_struct_parameters(value_type, parameter))
        else:
            parameter_list.append((last_value + "_" + parameter, value_type))
    return parameter_list



def convert_parameters_to_embedding_tensor(parameters: List, struct_size: int) -> torch.Tensor:
    embedding_dim = len(parameters[0][-1])
    instruction_tensor = torch.full((struct_size, embedding_dim), -1, dtype=torch.float)
    
    for param_idx, (_, _, _, embedding) in enumerate(parameters):
        if embedding is not None:
            instruction_tensor[param_idx] = embedding
            
    return instruction_tensor



def load_clip_means():
    with open("embedding_means.pkl", 'rb') as f:
        embedding_means = pickle.load(f)
    
    for key, value in embedding_means.items():
        embedding_means[key] = torch.tensor(value)

    return embedding_means



def get_mean_embedding(value_type, embedding_means):
    if value_type == bool:
        return embedding_means["boolean"]
    else:
        return embedding_means[value_type.__name__]