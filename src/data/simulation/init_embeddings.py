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
    data_path: str = "/media/external_2T/abolghasemi/random_simulation_dataset_v2.json",
) -> Dict[str, Any]:
    if os.path.exists(cache_file):
        print(f"Loading CLIP embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Generating new CLIP embeddings...")
    
    with open(data_path, 'r') as file:
        raw_data = json.load(file)

    embedder = CLIPEmbedder(clip_model_name)

    embeddings_data = {
        "simulations": []
    }

    for _, simulation_data in enumerate(raw_data):

        simulation_entry = {
            "simulation": [],
            "cinematography": []
        }
        simulation_sentences = []
        cinematography_sentences = []

        instructions = simulation_data.get("simulationInstructions", [])
        for instr in instructions:
            sentence_dict = generate_simulation_sentences(instr)
            simulation_sentences.append(sentence_dict)

        prompts = simulation_data.get("cinematographyPrompts", [])
        for prompt in prompts:
            sentence_dict = generate_cinematography_sentences(prompt)
            cinematography_sentences.append(sentence_dict)

        flat_sim_sents = []
        sim_sents_map = []

        for sim_dict in simulation_sentences:
            local_map = {}
            for key in ["init_setup", "movement", "end_setup", "constraints"]:
                text = sim_dict[key]
                if text is not None:
                    local_map[key] = len(flat_sim_sents)
                    flat_sim_sents.append(text)
                else:
                    local_map[key] = None
            sim_sents_map.append(local_map)

        if len(flat_sim_sents) > 0:
            sim_embeddings_list = embedder.get_embeddings(flat_sim_sents)
        else:
            sim_embeddings_list = []

        for sim_dict, local_map in zip(simulation_sentences, sim_sents_map):
            sim_emb_dict = {}
            for key in ["init_setup", "movement", "end_setup", "constraints"]:
                if local_map[key] is not None:
                    emb_idx = local_map[key]
                    sim_emb_dict[key] = sim_embeddings_list[emb_idx]
                else:
                    sim_emb_dict[key] = None
            simulation_entry["simulation"].append(sim_emb_dict)

        flat_cine_sents = []
        cine_sents_map = []
        for cine_dict in cinematography_sentences:
            local_map = {}
            for key in ["init_setup", "movement", "end_setup"]:
                text = cine_dict[key]
                if text is not None:
                    local_map[key] = len(flat_cine_sents)
                    flat_cine_sents.append(text)
                else:
                    local_map[key] = None
            cine_sents_map.append(local_map)

        if len(flat_cine_sents) > 0:
            cine_embeddings_list = embedder.get_embeddings(flat_cine_sents)
        else:
            cine_embeddings_list = []

        for cine_dict, local_map in zip(cinematography_sentences, cine_sents_map):
            cine_emb_dict = {}
            for key in ["init_setup", "movement", "end_setup"]:
                if local_map[key] is not None:
                    emb_idx = local_map[key]
                    cine_emb_dict[key] = cine_embeddings_list[emb_idx]
                else:
                    cine_emb_dict[key] = None
            simulation_entry["cinematography"].append(cine_emb_dict)

        embeddings_data["simulations"].append(simulation_entry)

    print(f"Saving CLIP embeddings to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_data, f)

    return embeddings_data
