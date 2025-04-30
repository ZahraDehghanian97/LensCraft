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


def extract_text_prompt(cin_params):
    if not cin_params:
        return ""
    
    prompt_parts = []
    
    if "initial" in cin_params:
        initial = cin_params["initial"]
        
        if "cameraAngle" in initial:
            angle = initial["cameraAngle"]
            if angle == "low":
                prompt_parts.append("The camera shoots at low angle.")
            elif angle == "high":
                prompt_parts.append("The camera shoots at high angle.")
            elif angle == "eye":
                prompt_parts.append("The camera shoots at eye level.")
            elif angle == "overhead":
                prompt_parts.append("The camera shoots at overhead angle.")
            elif angle == "birdsEye":
                prompt_parts.append("The camera shoots at bird's eye angle.")
        
        if "shotSize" in initial:
            shot_size = initial["shotSize"]
            prompt_parts.append(f"The camera shoots in {shot_size} shot.")
        
        if "subjectView" in initial:
            view = initial["subjectView"]
            if "threeQuarter" in view:
                parts = view.replace("threeQuarter", "").split("_")
                view_desc = f"{parts[0]} {parts[1]}" if len(parts) > 1 else view
                prompt_parts.append(f"The camera shoots in {view_desc} view.")
            else:
                prompt_parts.append(f"The camera shoots in {view} view.")
        
        if "subjectFraming" in initial:
            framing = initial["subjectFraming"]
            prompt_parts.append(f"The character is at the {framing} of the screen.")
    
    if "movement" in cin_params and "type" in cin_params["movement"]:
        movement_type = cin_params["movement"]["type"]
        
        if movement_type == "static":
            prompt_parts.append("The camera is static.")
        elif movement_type == "panLeft":
            prompt_parts.append("The camera pans to the left.")
        elif movement_type == "panRight":
            prompt_parts.append("The camera pans to the right.")
        elif movement_type == "tiltUp":
            prompt_parts.append("The camera tilts up.")
        elif movement_type == "tiltDown":
            prompt_parts.append("The camera tilts down.")
        elif movement_type == "dollyIn":
            prompt_parts.append("The camera pushes in to the character.")
        elif movement_type == "dollyOut":
            prompt_parts.append("The camera pulls out from the character.")
        elif movement_type == "arcLeft":
            prompt_parts.append("The camera rotates around the character to the left.")
        elif movement_type == "arcRight":
            prompt_parts.append("The camera rotates around the character to the right.")
        elif movement_type == "follow":
            prompt_parts.append("The camera follows the character.")
        elif movement_type == "track":
            prompt_parts.append("The camera tracks the character.")
        else:
            prompt_parts.append(f"The camera {movement_type}.")
            
        if "speed" in cin_params["movement"]:
            speed = cin_params["movement"]["speed"]
            if speed == "slowToFast":
                prompt_parts.append("The camera accelerates gradually.")
            elif speed == "fastToSlow":
                prompt_parts.append("The camera decelerates gradually.")
            elif speed == "smoothStartStop":
                prompt_parts.append("The camera moves with smooth start and stop.")
    
    if "final" in cin_params and cin_params["final"]:
        final = cin_params["final"]
        initial = cin_params.get("initial", {})
        
        if "shotSize" in final and ("shotSize" not in initial or final["shotSize"] != initial.get("shotSize")):
            prompt_parts.append(f"The camera moves from {initial.get('shotSize', 'current')} shot to {final['shotSize']} shot.")
        
        if "subjectView" in final and ("subjectView" not in initial or final["subjectView"] != initial.get("subjectView")):
            init_view = initial.get("subjectView", "current")
            final_view = final["subjectView"]
            prompt_parts.append(f"The camera switches from {init_view} view to {final_view} view.")
    
    return " ".join(prompt_parts)


def create_prompt_none_mask_matrix(cinematography_prompt_parameters: list, simulation_instruction_parameters: list, n_clip_embs: int):
    batch_size = len(cinematography_prompt_parameters)
    none_entries = torch.ones(batch_size, n_clip_embs, dtype=torch.bool)
    
    for sample_idx in range(batch_size):
        clip_embedding_parameters = cinematography_prompt_parameters[sample_idx] + simulation_instruction_parameters[sample_idx]
        
        for emb_idx in range(n_clip_embs):
            _, _, value_idx, _ = clip_embedding_parameters[emb_idx]
            if value_idx == -1:
                none_entries[sample_idx, emb_idx] = False
    
    return none_entries