from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import pickle

from .constants import (
    NumericFeature,
    cinematography_struct,
    simulation_struct,
)


def get_enum_index(enum_class, value) -> int:
    if isinstance(enum_class, type(Enum)):
        try:
            return list(enum_class).index(enum_class(value))
        except (TypeError, ValueError, KeyError):
            return -1
            
    return -1


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _coerce_finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _numeric_embedding(
    value: float, feature: NumericFeature, embedding_dim: int
) -> torch.Tensor:
    """Encode one bounded scalar as deterministic Fourier features."""

    bounded = min(max(value, feature.minimum), feature.maximum)
    width = feature.maximum - feature.minimum
    if width <= 0:
        raise ValueError("NumericFeature.maximum must be greater than minimum")
    shifted = bounded - feature.minimum
    if feature.logarithmic:
        normalized = math.log1p(shifted) / math.log1p(width)
    else:
        normalized = shifted / width

    frequency = torch.arange(
        1, (embedding_dim + 1) // 2 + 1, dtype=torch.float32
    )
    phase = frequency * (math.pi * normalized)
    embedding = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1).flatten()
    return embedding[:embedding_dim]


def _infer_embedding_dim(clip_embeddings: Optional[Dict]) -> int:
    if clip_embeddings:
        for values in clip_embeddings.values():
            if isinstance(values, dict):
                for embedding in values.values():
                    if torch.is_tensor(embedding):
                        return int(embedding.numel())
    raise ValueError("embedding_dim is required when no CLIP embeddings are available")


def extract_cinematography_parameters(
    data: Dict,
    struct: List,
    clip_embeddings: Optional[Dict] = None,
    prefix: str = "",
    fill_none_with_mean: bool = False,
    embedding_means = None,
    embedding_dim: Optional[int] = None,
) -> List[Tuple[str, Any, int, Optional[torch.Tensor]]]:
    parameters = []
    clip_embeddings = clip_embeddings or {}
    if embedding_dim is None:
        embedding_dim = _infer_embedding_dim(clip_embeddings)
    
    for key, value_type in struct:
        current_prefix = f"{prefix}_{key}" if prefix else key
        data_value = data.get(key, None)
        embedding = None
        
        if isinstance(value_type, type) and issubclass(value_type, Enum):
            index = get_enum_index(value_type, data_value)

            if index == -1 and fill_none_with_mean:
                embedding = get_mean_embedding(
                    value_type, embedding_means, embedding_dim
                )

            if index != -1:
                enum_value = value_type(data_value).value
                embedding = clip_embeddings.get(value_type.__name__, {}).get(
                    enum_value
                )
                if embedding is None:
                    embedding = get_mean_embedding(
                        value_type, embedding_means, embedding_dim
                    )

            parameters.append((current_prefix, data_value, index, embedding))
            
        elif value_type is bool:
            bool_value = _coerce_bool(data_value)
            if bool_value is None:
                index = -1
                if fill_none_with_mean:
                    embedding = get_mean_embedding(
                        value_type, embedding_means, embedding_dim
                    )

            else:
                index = 1 if bool_value else 0
                embedding = clip_embeddings.get("boolean", {}).get(bool_value)
                if embedding is None:
                    embedding = get_mean_embedding(
                        value_type, embedding_means, embedding_dim
                    )
                
            parameters.append((current_prefix, bool_value, index, embedding))

        elif isinstance(value_type, NumericFeature):
            numeric_value = _coerce_finite_number(data_value)
            if numeric_value is None:
                index = -1
                embedding = torch.zeros(embedding_dim, dtype=torch.float32)
            else:
                index = 0
                embedding = _numeric_embedding(
                    numeric_value, value_type, embedding_dim
                )
            parameters.append(
                (current_prefix, numeric_value, index, embedding)
            )
            
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
                embedding_dim=embedding_dim,
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


def flatten_struct_parameters(struct: list, prefix: str = "") -> list:
    parameter_list = list()
    for parameter, value_type in struct:
        current_prefix = f"{prefix}_{parameter}" if prefix else parameter
        if isinstance(value_type, list):
            parameter_list.extend(
                flatten_struct_parameters(value_type, current_prefix)
            )
        else:
            parameter_list.append((current_prefix, value_type))
    return parameter_list


def convert_parameters_to_embedding_tensor(
    parameters: List, struct_size: int, embedding_dim: int
) -> torch.Tensor:
    # A zero row is the explicit neutral/unknown token. Presence is tracked
    # separately by prompt_none_mask, so unknown values never masquerade as a
    # real enum or numeric constraint.
    instruction_tensor = torch.zeros(
        (struct_size, embedding_dim), dtype=torch.float32
    )
    
    for param_idx, (_, _, _, embedding) in enumerate(parameters):
        if embedding is not None:
            embedding = torch.as_tensor(embedding, dtype=torch.float32).flatten()
            if embedding.numel() != embedding_dim:
                raise ValueError(
                    f"Embedding at index {param_idx} has {embedding.numel()} "
                    f"elements; expected {embedding_dim}"
                )
            if not torch.isfinite(embedding).all():
                raise ValueError(f"Embedding at index {param_idx} is not finite")
            instruction_tensor[param_idx] = embedding
            
    return instruction_tensor


def load_clip_means():
    with open("embedding_means.pkl", 'rb') as f:
        embedding_means = pickle.load(f)
    
    for key, value in embedding_means.items():
        embedding_means[key] = torch.tensor(value)

    return embedding_means


def get_mean_embedding(value_type, embedding_means, embedding_dim):
    key = "boolean" if value_type is bool else value_type.__name__
    if embedding_means is not None and key in embedding_means:
        return torch.as_tensor(embedding_means[key], dtype=torch.float32)
    return torch.zeros(embedding_dim, dtype=torch.float32)


def create_prompt_none_mask(cinematography_prompt_parameters: list, simulation_instruction_parameters: list):
    clip_embedding_parameters = cinematography_prompt_parameters + simulation_instruction_parameters
    prompt_none_entries = torch.ones(len(clip_embedding_parameters), dtype=torch.bool)
        
    for emb_idx in range(len(clip_embedding_parameters)):
        _, _, value_idx, _ = clip_embedding_parameters[emb_idx]
        if value_idx == -1:
            prompt_none_entries[emb_idx] = False
    
    return prompt_none_entries


def fix_prompts_and_instructions(
    instruction,
    prompt,
    clip_embeddings,
    fill_none_with_mean,
    embedding_means,
    embedding_dim: Optional[int] = None,
):
    if embedding_dim is None:
        embedding_dim = _infer_embedding_dim(clip_embeddings)
    simulation_instruction_parameters = extract_cinematography_parameters(
        data=instruction,
        struct=simulation_struct,
        clip_embeddings=clip_embeddings,
        fill_none_with_mean=fill_none_with_mean,
        embedding_means=embedding_means,
        embedding_dim=embedding_dim,
    )

    cinematography_prompt_parameters = extract_cinematography_parameters(
        data=prompt,
        struct=cinematography_struct,
        clip_embeddings=clip_embeddings,
        fill_none_with_mean=fill_none_with_mean,
        embedding_means=embedding_means,
        embedding_dim=embedding_dim,
    )

    simulation_instruction_tensor = convert_parameters_to_embedding_tensor(
        simulation_instruction_parameters,
        simulation_struct_size,
        embedding_dim,
    )
    
    cinematography_prompt_tensor = convert_parameters_to_embedding_tensor(
        cinematography_prompt_parameters,
        cinematography_struct_size,
        embedding_dim,
    )
        
    prompt_none_mask = create_prompt_none_mask(
        cinematography_prompt_parameters=cinematography_prompt_parameters,
        simulation_instruction_parameters=simulation_instruction_parameters,
    )
    
    return simulation_instruction_tensor, cinematography_prompt_tensor, prompt_none_mask, simulation_instruction_parameters, cinematography_prompt_parameters


def structured_conditioning_from_batch(batch: Dict[str, torch.Tensor]):
    """Return every structured token in model order, or ``None``."""

    cinematography = batch.get("cinematography_prompt")
    simulation = batch.get("simulation_instruction")
    if cinematography is None:
        return None
    if simulation is None:
        return cinematography
    return torch.cat([cinematography, simulation], dim=0)



cinematography_struct_size = count_total_parameters_in_struct(cinematography_struct)

simulation_struct_size = count_total_parameters_in_struct(simulation_struct)

cinematography_struct_parameters = flatten_struct_parameters(cinematography_struct)

simulation_struct_parameters = flatten_struct_parameters(simulation_struct)

CLIP_PARAMETERS = cinematography_struct_parameters + simulation_struct_parameters

CLIP_PARAMETERS_DICT = dict(CLIP_PARAMETERS)
