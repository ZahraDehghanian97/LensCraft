import os
import json
import pickle
from typing import Dict

import torch

from data.simulation.constants import movement_descriptions, easing_descriptions, angle_descriptions, shot_descriptions
from models.clip_embeddings import CLIPEmbedder

def generate_caption(instruction: Dict) -> str:
    movement = movement_descriptions[instruction['cameraMovement']]
    easing = easing_descriptions[instruction['movementEasing']]
    angle = angle_descriptions[instruction.get('initialCameraAngle', 'mediumAngle')]
    shot = shot_descriptions[instruction.get('initialShotType', 'mediumShot')]
    
    return f"The camera is {movement} {easing}, {angle} {shot}."

def initialize_all_clip_embeddings(
    clip_model_name: str = "openai/clip-vit-large-patch14",
    cache_file: str = "clip_embeddings_cache.pkl",
    data_path: str = "/media/external_2T/abolghasemi/random_simulation_dataset.json",
) -> Dict[str, Dict[str, torch.Tensor]]:
    if os.path.exists(cache_file):
        print(f"Loading CLIP embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Generating new CLIP embeddings...")
    
    with open(data_path, 'r') as file:
        raw_data = json.load(file)
    
    valid_simulations = [
        sim for sim in raw_data['simulations']
        if len(sim['instructions']) == 1 and 
           sim['instructions'][0]['frameCount'] == 30 and
           len(sim['cameraFrames']) == 30
    ]

    embedder = CLIPEmbedder(clip_model_name)

    captions = [generate_caption(sim['instructions'][0]) for sim in valid_simulations]
    caption_embeddings = embedder.get_embeddings(captions)

    embeddings = {
        'movement': embedder.embed_descriptions(movement_descriptions),
        'easing': embedder.embed_descriptions(easing_descriptions),
        'angle': embedder.embed_descriptions(angle_descriptions),
        'shot': embedder.embed_descriptions(shot_descriptions),
        'caption_feat': caption_embeddings
    }

    print(f"Saving CLIP embeddings to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings
