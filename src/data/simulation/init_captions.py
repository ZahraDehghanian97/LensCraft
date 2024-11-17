import os
import pickle
from typing import Dict

import torch

from data.simulation.constants import movement_descriptions, easing_descriptions, angle_descriptions, shot_descriptions
from models.clip_embeddings import CLIPEmbedder


def initialize_all_clip_embeddings(
    clip_model_name: str = "openai/clip-vit-large-patch14",
    cache_file: str = "clip_embeddings_cache.pkl"
) -> Dict[str, Dict[str, torch.Tensor]]:
    if os.path.exists(cache_file):
        print(f"Loading CLIP embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Generating new CLIP embeddings...")

    embedder = CLIPEmbedder(clip_model_name)

    embeddings = {
        'movement': embedder.embed_descriptions(movement_descriptions),
        'easing': embedder.embed_descriptions(easing_descriptions),
        'angle': embedder.embed_descriptions(angle_descriptions),
        'shot': embedder.embed_descriptions(shot_descriptions)
    }

    print(f"Saving CLIP embeddings to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings
