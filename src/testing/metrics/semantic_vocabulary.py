"""Frozen semantic metric vocabulary; never used to condition the generator."""
from typing import Dict, Any
import hashlib
import json
import pickle
import torch
from pathlib import Path
from data.simulation.caption import enum_descriptions
import numpy as np


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


def _create_clip_embedder(clip_model_name: str, chunk_size: int):
    # Keep the large transformers dependency lazy. Validating or consuming an
    # existing cache should not require downloading/importing a CLIP model.
    from testing.metrics.clip_embeddings import CLIPEmbedder

    return CLIPEmbedder(clip_model_name, chunk_size=chunk_size)


def load_metric_clip_embeddings(
    clip_model_name: str = "openai/clip-vit-base-patch32",
    cache_file: str = "clip_embeddings_cache.pkl",
    chunk_size: int = 100,
    embedding_dimension: int = 512,
) -> Dict[str, Any]:
    """Load raw frozen metric vectors, building their cache only if absent/stale.

    The cache fingerprint and sentence ordering retain the historical evaluator
    protocol. No normalization, PCA, generator fill statistics, or trainable
    conditioning table is supported or created here.
    """
    embeddings = None
    try:
        with open(cache_file, 'rb') as stream:
            embeddings = pickle.load(stream)
        if (not _cache_has_current_vocabulary(embeddings, embedding_dimension)
                or not _embedding_cache_metadata_matches(cache_file, clip_model_name, embedding_dimension)):
            embeddings = None
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        pass
    if embeddings is not None:
        return embeddings

    embedder = _create_clip_embedder(clip_model_name, chunk_size)
    sentences, metadata = [], []
    for family, descriptions in enum_descriptions.items():
        for value, sentence in descriptions.items():
            sentences.append(sentence)
            metadata.append((family, value))
    for value, sentence in _BOOLEAN_DESCRIPTIONS.items():
        sentences.append(sentence)
        metadata.append(('boolean', value))
    vectors = embedder.extract_clip_embeddings(sentences).cpu()
    if vectors.shape != (len(sentences), embedding_dimension) or not torch.isfinite(vectors).all():
        raise ValueError('Frozen metric CLIP vectors have an unexpected shape or nonfinite values')
    embeddings = {family: {} for family in enum_descriptions}
    embeddings['boolean'] = {}
    for (family, value), vector in zip(metadata, vectors):
        embeddings[family][value] = vector
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as stream:
        pickle.dump(embeddings, stream)
    _write_embedding_cache_metadata(cache_file, clip_model_name, embedding_dimension)
    return embeddings
