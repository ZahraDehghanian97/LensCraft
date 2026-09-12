"""Fixed decoder-memory scales from the conditioning vocabulary, not samples."""

from collections.abc import Mapping
from numbers import Integral

import torch

from .constants import NumericFeature
from .utils import CLIP_PARAMETERS


def vocabulary_memory_norms(clip_embeddings: Mapping, embedding_dim: int) -> list[float]:
    """Mean categorical vector norms and analytic numeric Fourier norms.

    The returned values follow encoder query/structured-caption slot order.
    Neither caption values nor per-sample field presence enter this calculation.
    """
    if (isinstance(embedding_dim, bool) or not isinstance(embedding_dim, Integral)
            or embedding_dim <= 0 or embedding_dim % 2):
        raise ValueError("Camera memory calibration requires a positive even embedding_dim")
    if not isinstance(clip_embeddings, Mapping):
        raise ValueError("clip_embeddings must be a conditioning vocabulary mapping")

    norms_by_type = {}
    norms = []
    for _, field_type in CLIP_PARAMETERS:
        if isinstance(field_type, NumericFeature):
            norms.append((embedding_dim / 2) ** 0.5)
            continue

        key = "boolean" if field_type is bool else field_type.__name__
        if key not in norms_by_type:
            vocabulary = clip_embeddings.get(key)
            if not isinstance(vocabulary, Mapping) or not vocabulary:
                raise ValueError(f"Missing or empty conditioning vocabulary for {key}")
            vector_norms = []
            for value, vector in vocabulary.items():
                vector = torch.as_tensor(vector, dtype=torch.float32).detach()
                if vector.shape != (embedding_dim,) or not torch.isfinite(vector).all():
                    raise ValueError(f"Invalid {embedding_dim}-D vocabulary vector for {key}.{value}")
                norm = vector.norm()
                if not torch.isfinite(norm) or norm <= 0:
                    raise ValueError(f"Vocabulary vector norm must be finite and positive: {key}.{value}")
                vector_norms.append(norm)
            norms_by_type[key] = torch.stack(vector_norms).mean().item()
        norms.append(norms_by_type[key])
    return norms
