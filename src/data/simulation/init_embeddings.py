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
    data_path: str = "/media/external_2T/abolghasemi/p_haghighi/random_simulation_dataset_v2.json",
) -> Dict[str, Any]:
    if os.path.exists(cache_file):
        print(f"Loading CLIP embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Generating new CLIP embeddings...")
    
    with open(data_path, 'r') as file:
        raw_data = json.load(file)

    embedder = CLIPEmbedder(clip_model_name)

    embeddings_data = []

    for _, simulation_data in enumerate(raw_data):
        embedding = { "simulation": {}, "cinematography": {} }
        embeddings_data.append(embedding)
        
        flat_sents = []
        local_map = {}

        instruction = simulation_data["simulationInstructions"][0]
        sentence_dict = generate_simulation_sentences(instruction)
        for key in ["init_setup", "movement", "end_setup", "constraints"]:
            text = sentence_dict[key]
            if text is not None:
                local_map['sim_' + key] = len(sentence_dict)
                flat_sents.append(text)
            else:
                local_map['sim_' + key] = None

        prompt = simulation_data["cinematographyPrompts"][0]
        sentence_dict = generate_cinematography_sentences(prompt)
        for key in ["init_setup", "movement", "end_setup"]:
            text = sentence_dict[key]
            if text is not None:
                local_map['cin_' + key] = len(sentence_dict)
                flat_sents.append(text)
            else:
                local_map['cin_' + key] = None

        embeddings_list = embedder.get_embeddings(flat_sents)
        
        for key in ["init_setup", "movement", "end_setup", "constraints"]:
            emb_idx = local_map['sim_' + key]
            embedding["simulation"][key] = embeddings_list[emb_idx].to('cpu') if emb_idx is not None else None
        
        for key in ["init_setup", "movement", "end_setup"]:
            emb_idx = local_map['cin_' + key]
            embedding["cinematography"][key] = embeddings_list[emb_idx].to('cpu') if emb_idx is not None else None

    print(f"Saving CLIP embeddings to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_data, f)

    return embeddings_data
