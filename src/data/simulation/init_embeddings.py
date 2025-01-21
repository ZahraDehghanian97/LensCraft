import os
import json
import pickle
from typing import Dict, Any

from models.clip_embeddings import CLIPEmbedder
from data.simulation.caption import (
    generate_cinematography_sentences,
    generate_simulation_sentences
)


def initialize_all_clip_embeddings(
    clip_model_name: str = "openai/clip-vit-large-patch14",
    cache_file: str = "clip_embeddings_cache.pkl",
    data_path: str = "/media/external_2T/abolghasemi/p_haghighi/random_simulation_dataset_v2_static.json",
) -> Dict[str, Any]:
    if os.path.exists(cache_file):
        print(f"Loading CLIP embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Generating new CLIP embeddings...")
    
    with open(data_path, 'r') as file:
        raw_data = json.load(file)

    embedder = CLIPEmbedder(clip_model_name)

    all_sentences = []
    sentence_maps = []

    for simulation_data in raw_data:
        local_map = {}

        instruction = simulation_data["simulationInstructions"][0]
        sim_sentences = generate_simulation_sentences(instruction)
        for key in ["init_setup", "movement", "end_setup", "constraints"]:
            text = sim_sentences[key]
            if text is not None:
                local_map['sim_' + key] = len(all_sentences)
                all_sentences.append(text)
            else:
                local_map['sim_' + key] = None
        
        dynamic_conf = instruction.get("dynamic", {})
        is_interpolation = dynamic_conf.get("type") == "interpolation"

        prompt = simulation_data["cinematographyPrompts"][0]
        cin_sentences = generate_cinematography_sentences(prompt, is_interpolation)
        for key in ["init_setup", "simple_movement", "interpolation_movement", "end_setup"]:
            text = cin_sentences[key]
            if text is not None:
                local_map['cin_' + key] = len(all_sentences)
                all_sentences.append(text)
            else:
                local_map['cin_' + key] = None

        sentence_maps.append(local_map)

    all_embeddings = embedder.get_embeddings(all_sentences)

    embeddings_data = []
    for sentence_map in sentence_maps:
        embedding = {
            "simulation": {},
            "cinematography": {}
        }
        
        for key in ["init_setup", "movement", "end_setup", "constraints"]:
            idx = sentence_map['sim_' + key]
            embedding["simulation"][key] = all_embeddings[idx].to('cpu') if idx is not None else None
        
        for key in ["init_setup", "simple_movement", "interpolation_movement", "end_setup"]:
            idx = sentence_map['cin_' + key]
            embedding["cinematography"][key] = all_embeddings[idx].to('cpu') if idx is not None else None
        
        embeddings_data.append(embedding)

    print(f"Saving CLIP embeddings to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_data, f)

    return embeddings_data
