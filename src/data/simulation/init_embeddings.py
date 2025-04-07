from typing import Dict, Any, List, Tuple
import pickle
import torch
import os
from .caption import enum_descriptions
from models.clip_embeddings import CLIPEmbedder
import numpy as np
import copy
from sklearn.decomposition import PCA

def calc_embedding_mean(embeddings_data: dict, embedding_dimension: int=512) -> dict:
    means = {}
    for key, value in embeddings_data.items():
        sum = np.zeros(embedding_dimension, dtype=np.float32)
        n_emb = 0
        for _, vector in value.items():
            sum += vector.numpy()
            n_emb += 1
        means[key] = sum / n_emb
    return means


def calc_embedding_std(embeddings_data: dict, means: dict, embedding_dimension: int=512) -> dict:
    stds = {}
    for key, value in embeddings_data.items():
        sum_squared_error = np.zeros(embedding_dimension, dtype=np.float32)
        n_emb = 0
        for _, vector in value.items():
            sum_squared_error += (vector.numpy() - means[key]) ** 2
            n_emb += 1
        stds[key] = np.sqrt(sum_squared_error / n_emb)
    return stds


def normalize_embeddings(embeddings_data: dict, embedding_dimension: int) -> dict:
    means = calc_embedding_mean(embeddings_data, embedding_dimension)
    stds = calc_embedding_std(embeddings_data, means, embedding_dimension)
    save_means_and_stds(means, stds)
    embeddings_data_normalized = copy.deepcopy(embeddings_data)
    for key, value in embeddings_data.items():
        for key_nested, vector in value.items():
            mean = means[key]
            std = stds[key]
            vector_normalized = (vector - mean) / std
            embeddings_data_normalized[key][key_nested] = vector_normalized
    return embeddings_data_normalized



def pca_embeddings(embeddings_data: dict, n_components: int) -> tuple[dict, dict]:
    clip_emb = copy.deepcopy(embeddings_data)
    pca_map = dict()
    for key, value in clip_emb.items():
        feature_embeddings = list()
        n_feature_embeddings = len(value)

        for key_n, value_n in value.items():
            feature_embeddings.append(value_n)
        feature_embeddings = np.array(feature_embeddings)
        std = feature_embeddings.std(axis=0)

        while feature_embeddings.shape[0] < n_components:
            noise = np.random.randn(*feature_embeddings.shape) * (std * 10e-4)
            feature_embeddings = np.concatenate([feature_embeddings, 
                                                 feature_embeddings + noise], axis=0)

        pca = PCA(n_components=n_components)
        pca.fit(feature_embeddings)
        feature_embeddings_low_dim = pca.transform(feature_embeddings[:n_feature_embeddings, :])
        pca_map[key] = pca

        feature_embeddings_low_dim = torch.tensor(feature_embeddings_low_dim)
        idx = 0
        for key_n, value_n in value.items():
            clip_emb[key][key_n] = feature_embeddings_low_dim[idx, :]
            idx += 1

    return clip_emb, pca_map


def save_means_and_stds(means: dict, stds:dict) -> None:
    with open("embedding_means.pkl", "wb") as f:
        pickle.dump(means, f)

    with open("embedding_stds.pkl", "wb") as f:
        pickle.dump(stds, f)


def initialize_all_clip_embeddings(
    clip_model_name: str = "openai/clip-vit-large-patch14",
    cache_file: str = "clip_embeddings_cache.pkl",
    chunk_size: int = 100,
    embedding_dimension: int = 512,
    embedding_mode: str = "default",
    n_components_pca: int = 10
) -> Dict[str, Any]:
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
    try:
        with open(cache_file, 'rb') as f:
            print(f"Loading CLIP embeddings from cache: {cache_file}")
            embeddings = pickle.load(f)
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

        embeddings: Dict[str, Dict[Any, torch.Tensor]] = {
            param_type: {} for param_type in enum_descriptions.keys()
        }
        embeddings["boolean"] = {}

        for (param_type, key), embedding in zip(metadata, all_embeddings):
            if param_type == "boolean":
                embeddings[param_type][key == "True"] = embedding
            else:
                embeddings[param_type][key] = embedding

        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"Saved CLIP embeddings to cache: {cache_file}")

    finally:
        if embedding_mode == "default":
            return embeddings
        elif embedding_mode == "normal":
            return normalize_embeddings(embeddings, embedding_dimension)
        elif embedding_mode == "pca":
            return pca_embeddings(embeddings, n_components_pca)
