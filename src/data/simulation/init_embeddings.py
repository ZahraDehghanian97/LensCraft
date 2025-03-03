from typing import Dict, Any, List, Tuple
import pickle
import torch
import os
from .caption import enum_descriptions
from models.clip_embeddings import CLIPEmbedder
import numpy as np
import copy

def calc_embedding_mean(embeddings_data: dict, embedding_dimension: int=512) -> dict:
    means = {}
    for key, value in embeddings_data.items():
        sum = np.zeros(embedding_dimension)
        n_emb = 0
        for _, vector in value.items():
            sum += vector.numpy()
            n_emb += 1
        means[key] = sum / n_emb
    return means


def calc_embedding_std(embeddings_data: dict, means: dict, embedding_dimension: int=512) -> dict:
    stds = {}
    for key, value in embeddings_data.items():
        sum_squared_error = np.zeros(embedding_dimension)
        n_emb = 0
        for _, vector in value.items():
            sum_squared_error += (vector.numpy() - means[key]) ** 2
            n_emb += 1
        stds[key] = np.sqrt(sum_squared_error / n_emb)
    return stds


def normalize_embeddings(embeddings_data: dict, embedding_dimension: int) -> dict:
    means = calc_embedding_mean(embeddings_data, embedding_dimension)
    stds = calc_embedding_std(embeddings_data, means, embedding_dimension)
    embeddings_data_normalized = copy.deepcopy(embeddings_data)
    for key, value in embeddings_data.items():
        for key_nested, vector in value.items():
            mean = means[key]
            std = stds[key]
            vector_normalized = (vector - mean) / std
            embeddings_data_normalized[key][key_nested] = vector_normalized
    return embeddings_data_normalized


def initialize_all_clip_embeddings(
    clip_model_name: str = "openai/clip-vit-large-patch14",
    cache_file: str = "clip_embeddings_cache.pkl",
    chunk_size: int = 100,
    embedding_dimension: int = 512,
    normalize: bool = True,
) -> Dict[str, Any]:
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
    try:
        with open(cache_file, 'rb') as f:
            print(f"Loading CLIP embeddings from cache: {cache_file}")
            if normalize:
                return normalize_embeddings(pickle.load(f), embedding_dimension)
            else:
                return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        print("Generating new CLIP embeddings...")

    embedder = CLIPEmbedder(clip_model_name, chunk_size=chunk_size)
    
    all_sentences: List[str] = []
    metadata: List[Tuple[str, str]] = []  
    
    for param_type, descriptions in enum_descriptions.items():
        for key, sentence in descriptions.items():
            all_sentences.append(sentence)
            metadata.append((param_type, key))
    
    bool_sentences = ["enabled", "disabled"]
    bool_keys = [True, False]
    all_sentences.extend(bool_sentences)
    metadata.extend([("boolean", str(key)) for key in bool_keys])
    
    all_embeddings = embedder.get_embeddings(all_sentences).to('cpu')
    
    embeddings_data: Dict[str, Dict[Any, torch.Tensor]] = {
        param_type: {} for param_type in enum_descriptions.keys()
    }
    embeddings_data["boolean"] = {}
    
    for (param_type, key), embedding in zip(metadata, all_embeddings):
        if param_type == "boolean":
            embeddings_data[param_type][key == "True"] = embedding
        else:
            embeddings_data[param_type][key] = embedding

    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print(f"Saved CLIP embeddings to cache: {cache_file}")
    if normalize:
        return normalize_embeddings(embeddings_data, embedding_dimension)
    else: 
        return embeddings_data