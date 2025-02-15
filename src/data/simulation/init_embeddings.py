from typing import Dict, Any, List, Tuple
import pickle
import torch
import os
from .caption import enum_descriptions
from models.clip_embeddings import CLIPEmbedder


def initialize_all_clip_embeddings(
    clip_model_name: str = "openai/clip-vit-large-patch14",
    cache_file: str = "clip_embeddings_cache.pkl",
) -> Dict[str, Any]:
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
    try:
        with open(cache_file, 'rb') as f:
            print(f"Loading CLIP embeddings from cache: {cache_file}")
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        print("Generating new CLIP embeddings...")

    embedder = CLIPEmbedder(clip_model_name)
    
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
    return embeddings_data