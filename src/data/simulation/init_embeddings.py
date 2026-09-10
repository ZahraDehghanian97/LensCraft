from typing import Dict, Any, List, Tuple
import hashlib
import json
import pickle
import torch
import os
from pathlib import Path
from .caption import enum_descriptions
import numpy as np
import copy


CLIP_EMBEDDING_CACHE_VERSION = 2
_BOOLEAN_DESCRIPTIONS = {True: "enabled", False: "disabled"}


def _embedding_cache_fingerprint(
    clip_model_name: str, embedding_dimension: int
) -> str:
    """Fingerprint every input that determines the cached CLIP vectors."""

    payload = {
        "version": CLIP_EMBEDDING_CACHE_VERSION,
        "clipModelName": clip_model_name,
        "embeddingDimension": embedding_dimension,
        "enumDescriptions": enum_descriptions,
        "booleanDescriptions": _BOOLEAN_DESCRIPTIONS,
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _embedding_cache_metadata_path(cache_file: str) -> Path:
    return Path(f"{cache_file}.meta.json")


def _embedding_cache_metadata_matches(
    cache_file: str, clip_model_name: str, embedding_dimension: int
) -> bool:
    try:
        with _embedding_cache_metadata_path(cache_file).open(
            "r", encoding="utf-8"
        ) as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("version") == CLIP_EMBEDDING_CACHE_VERSION
        and metadata.get("fingerprint")
        == _embedding_cache_fingerprint(clip_model_name, embedding_dimension)
    )


def _write_embedding_cache_metadata(
    cache_file: str, clip_model_name: str, embedding_dimension: int
) -> None:
    metadata = {
        "version": CLIP_EMBEDDING_CACHE_VERSION,
        "fingerprint": _embedding_cache_fingerprint(
            clip_model_name, embedding_dimension
        ),
    }
    with _embedding_cache_metadata_path(cache_file).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")


def _as_numpy_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _is_finite_vector(value: Any, embedding_dimension: int) -> bool:
    try:
        vector = _as_numpy_vector(value)
    except (RuntimeError, TypeError, ValueError):
        return False
    return vector.shape == (embedding_dimension,) and np.isfinite(vector).all()


def _cache_has_current_vocabulary(
    embeddings: Dict[str, Any], embedding_dimension: int
) -> bool:
    if not isinstance(embeddings, dict):
        return False
    expected_types = set(enum_descriptions) | {"boolean"}
    if set(embeddings) != expected_types:
        return False
    for parameter_type, descriptions in enum_descriptions.items():
        cached = embeddings.get(parameter_type)
        if not isinstance(cached, dict) or set(cached) != set(descriptions):
            return False
    boolean = embeddings.get("boolean")
    if not isinstance(boolean, dict) or set(boolean) != {True, False}:
        return False

    return all(
        _is_finite_vector(vector, embedding_dimension)
        for values in embeddings.values()
        for vector in values.values()
    )


def _statistics_match_embeddings(
    embeddings: Dict[str, Any],
    means: Any,
    stds: Any,
    embedding_dimension: int,
) -> bool:
    """Validate cached moments against every current cached embedding."""

    if not isinstance(means, dict) or not isinstance(stds, dict):
        return False
    expected_types = set(embeddings)
    if set(means) != expected_types or set(stds) != expected_types:
        return False

    try:
        expected_means = calc_embedding_mean(embeddings, embedding_dimension)
        expected_stds = calc_embedding_std(
            embeddings, expected_means, embedding_dimension
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False

    try:
        for parameter_type in expected_types:
            cached_mean = _as_numpy_vector(means[parameter_type])
            cached_std = _as_numpy_vector(stds[parameter_type])
            if (
                cached_mean.shape != (embedding_dimension,)
                or cached_std.shape != (embedding_dimension,)
                or not np.isfinite(cached_mean).all()
                or not np.isfinite(cached_std).all()
                or not np.allclose(cached_mean, expected_means[parameter_type])
                or not np.allclose(cached_std, expected_stds[parameter_type])
            ):
                return False
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def _create_clip_embedder(clip_model_name: str, chunk_size: int):
    # Keep the large transformers dependency lazy. Validating or consuming an
    # existing cache should not require downloading/importing a CLIP model.
    from models.clip_embeddings import CLIPEmbedder

    return CLIPEmbedder(clip_model_name, chunk_size=chunk_size)


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


def generate_mean_stds(embeddings_data: dict, embedding_dimension: int):
    means = calc_embedding_mean(embeddings_data, embedding_dimension)
    stds = calc_embedding_std(embeddings_data, means, embedding_dimension)
    save_means_and_stds(means, stds)
    return means, stds

def normalize_embeddings(embeddings_data: dict, embedding_dimension: int) -> dict:
    means, stds = generate_mean_stds(embeddings_data, embedding_dimension)
    embeddings_data_normalized = copy.deepcopy(embeddings_data)
    for key, value in embeddings_data.items():
        for key_nested, vector in value.items():
            mean = means[key]
            std = stds[key]
            vector_normalized = (vector - mean) / std
            embeddings_data_normalized[key][key_nested] = vector_normalized
    return embeddings_data_normalized


def save_means_and_stds(means: dict, stds:dict) -> None:
    with open("embedding_means.pkl", "wb") as f:
        pickle.dump(means, f)

    with open("embedding_stds.pkl", "wb") as f:
        pickle.dump(stds, f)


def initialize_all_clip_embeddings(
    clip_model_name: str = "openai/clip-vit-base-patch32",
    cache_file: str = "clip_embeddings_cache.pkl",
    chunk_size: int = 100,
    embedding_dimension: int = 512,
    embedding_mode: str = "default",
    n_components_pca: int = 10
) -> Dict[str, Any]:
    """Initialize conditioning embeddings from a validated cache or CLIP.

    ``n_components_pca`` is deprecated and ignored, but remains accepted for
    compatibility with older callers. PCA conditioning is unsupported.
    """
    if embedding_mode == "pca":
        raise ValueError(
            "embedding_mode='pca' changes the conditioning width and returns "
            "a projection map, so it is not supported by LensCraft dataset or "
            "loss consumers. Use 'default' or 'normal'."
        )
    if embedding_mode not in {"default", "normal"}:
        raise ValueError(f"Unsupported embedding_mode: {embedding_mode}")

    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
    embeddings = None
    embeddings_rebuilt = False
    try:
        with open(cache_file, 'rb') as f:
            print(f"Loading CLIP embeddings from cache: {cache_file}")
            embeddings = pickle.load(f)
        if (
            not _cache_has_current_vocabulary(embeddings, embedding_dimension)
            or not _embedding_cache_metadata_matches(
                cache_file, clip_model_name, embedding_dimension
            )
        ):
            print(
                "CLIP embedding cache metadata or parameter vocabulary is "
                "stale; rebuilding it."
            )
            embeddings = None
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        embeddings = None

    if embeddings is None:
        embeddings_rebuilt = True
        print("Generating new CLIP embeddings...")

        embedder = _create_clip_embedder(clip_model_name, chunk_size)

        all_sentences: List[str] = []
        metadata: List[Tuple[str, str]] = []  

        for param_type, descriptions in enum_descriptions.items():
            for key, sentence in descriptions.items():
                all_sentences.append(sentence)
                metadata.append((param_type, key))

        bool_keys = list(_BOOLEAN_DESCRIPTIONS)
        all_sentences.extend(_BOOLEAN_DESCRIPTIONS[key] for key in bool_keys)
        metadata.extend([("boolean", str(key)) for key in bool_keys])

        all_embeddings = embedder.extract_clip_embeddings(all_sentences).to('cpu')
        actual_dimension = (
            all_embeddings.shape[-1] if all_embeddings.ndim > 0 else 0
        )
        if all_embeddings.ndim != 2 or actual_dimension != embedding_dimension:
            raise ValueError(
                f"CLIP model produced {actual_dimension}-D embeddings; "
                f"configured embedding_dimension is {embedding_dimension}"
            )

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
        _write_embedding_cache_metadata(
            cache_file, clip_model_name, embedding_dimension
        )

        print(f"Saved CLIP embeddings to cache: {cache_file}")

    means = None
    stds = None
    if not embeddings_rebuilt:
        try:
            with open("embedding_means.pkl", "rb") as file:
                means = pickle.load(file)
            with open("embedding_stds.pkl", "rb") as file:
                stds = pickle.load(file)
        except (OSError, pickle.UnpicklingError, EOFError):
            means = None
            stds = None
    if embeddings_rebuilt or not _statistics_match_embeddings(
        embeddings, means, stds, embedding_dimension
    ):
        generate_mean_stds(embeddings, embedding_dimension)

    if embedding_mode == "default":
        return embeddings
    if embedding_mode == "normal":
        return normalize_embeddings(embeddings, embedding_dimension)
    raise AssertionError("embedding_mode validation and dispatch disagree")
