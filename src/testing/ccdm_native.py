"""CCDM's released motion-classifier metrics, independent of CLaTr and poses.

The release uses 300 native XYZ/screen-XY frames, rather than the 60-frame
paper experiment. A compatible 60-frame classifier checkpoint is supported,
but this module never changes sequence length by resampling. A newly trained
classifier must be reported as a reconstruction, not the authors' evaluator.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import linalg
from torch import nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.shape[0], :])


class _TimestepEmbedder(nn.Module):
    """Retained although unused: its parameters are in released state_dicts."""

    def __init__(self, latent_dim: int, sequence_pos_encoder: _PositionalEncoding):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class CCDMNativeClassifier(nn.Module):
    """Exact released classify.py Transformer, including checkpoint key names."""

    def __init__(self, frames: int = 300):
        super().__init__()
        self.frames = _positive_integer(frames, "frames")
        if self.frames > 5000:
            raise ValueError("frames exceeds the released positional-encoding length (5000)")
        self.n_feature, self.n_label = 5, 6
        self.latent_dim, self.num_heads, self.ff_size = 256, 4, 1024
        self.dropout, self.activation, self.num_layers = 0.1, "gelu", 4
        self.input_process = nn.Linear(self.n_feature, self.latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=self.num_heads,
            dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.sequence_pos_encoder = _PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = _TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.output_process = nn.Sequential(nn.Linear(self.latent_dim, 1), nn.ReLU())
        self.pred = nn.Sequential(nn.Linear(self.frames, self.n_label))

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.frames, 5) or len(x) == 0:
            raise ValueError(f"classifier input must have shape [N, {self.frames}, 5], N > 0")
        if not torch.isfinite(x).all():
            raise ValueError("classifier input contains nonfinite values")
        xseq = self.input_process(x.permute(1, 0, 2))
        xseq = self.sequence_pos_encoder(xseq)
        xseq = self.seqTransEncoder(xseq)
        xseq = self.output_process(xseq).permute(1, 0, 2)
        return xseq.reshape(len(x), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.forward_feature(x))


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_array(value: Any, name: str, *, dtype: Any = np.float32) -> np.ndarray:
    try:
        raw = _numpy(value)
        if raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must contain real numbers")
        result = np.asarray(raw, dtype=dtype)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def load_classifier(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    """Load a trusted, plain state_dict; infer frames from its classification head.

    weights_only avoids arbitrary checkpoint pickle execution. It does not
    establish that an externally supplied evaluator has an authentic origin.
    """
    path = Path(checkpoint_path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("classifier checkpoint must be a plain tensor state_dict")
    head = state.get("pred.0.weight")
    if head is None or head.ndim != 2 or head.shape[0] != 6:
        raise ValueError("classifier checkpoint must have a six-class pred.0.weight head")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("classifier checkpoint contains nonfinite values")
    model = CCDMNativeClassifier(frames=int(head.shape[1]))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "path": str(path.resolve()), "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size, "frames": model.frames,
        "feature_dim": model.frames, "motion_classes": 6,
        "architecture": "released classify.py Transformer: latent256/heads4/layers4/ff1024",
        "load": "plain state_dict; weights_only=True; strict=True",
    }
    return model, metadata


def prepare_native_trajectories(
    trajectories: Sequence[Any] | np.ndarray | torch.Tensor,
    mean: Any, std: Any, expected_frames: int = 300,
) -> np.ndarray:
    """Normalize raw native5D, then last-frame-pad; never resample or truncate."""
    expected_frames = _positive_integer(expected_frames, "expected_frames")
    # Retain input/statistics dtypes until the dataset's final float32 cast,
    # exactly as classify.py does (including float32 normalization when used).
    mean = _finite_array(mean, "native mean", dtype=None)
    std = _finite_array(std, "native std", dtype=None)
    if mean.shape != (5,) or std.shape != (5,):
        raise ValueError("native mean/std must each have shape [5]")
    if (std < 0).any():
        raise ValueError("native std must be nonnegative")
    if not isinstance(trajectories, (Sequence, np.ndarray, torch.Tensor)) or len(trajectories) == 0:
        raise ValueError("trajectories must contain at least one native5D sequence")
    prepared = []
    for index, trajectory in enumerate(trajectories):
        raw = _finite_array(trajectory, f"trajectory {index}", dtype=None)
        if raw.ndim != 2 or raw.shape[1] != 5 or raw.shape[0] == 0:
            raise ValueError(f"trajectory {index} must have shape [T, 5], T > 0")
        if len(raw) > expected_frames:
            raise ValueError(
                f"trajectory {index} has {len(raw)} frames, exceeding classifier {expected_frames}; "
                "resampling/truncation requires a separate explicit protocol"
            )
        normalized = (raw - mean[None, :]) / (std[None, :] + 1e-8)
        if len(raw) < expected_frames:
            normalized = np.concatenate((
                normalized, np.repeat(normalized[-1:], expected_frames - len(raw), axis=0),
            ))
        normalized = normalized.astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError(f"normalized trajectory {index} contains nonfinite values")
        prepared.append(normalized)
    return np.stack(prepared)


def extract_native_features(
    model: CCDMNativeClassifier, prepared: Any, batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    batch_size = _positive_integer(batch_size, "batch_size")
    values = _finite_array(prepared, "prepared trajectories")
    if values.ndim != 3 or values.shape[1:] != (model.frames, 5) or len(values) == 0:
        raise ValueError(f"prepared trajectories must have shape [N, {model.frames}, 5], N > 0")
    model.to(device).eval()
    features, logits = [], []
    with torch.inference_mode():
        for offset in range(0, len(values), batch_size):
            x = torch.from_numpy(values[offset:offset + batch_size]).to(device)
            feature = model.forward_feature(x)
            score = model.pred(feature)
            if not torch.isfinite(feature).all() or not torch.isfinite(score).all():
                raise ValueError("classifier produced nonfinite features/logits")
            features.append(feature.cpu().numpy())
            logits.append(score.cpu().numpy())
    scores = np.concatenate(logits)
    return {
        "features": np.concatenate(features), "logits": scores,
        "predictions": scores.argmax(axis=1),
    }


def frechet_distance(mu1: Any, sigma1: Any, mu2: Any, sigma2: Any, eps: float = 1e-6) -> float:
    """The released metric.py Gaussian distance, including singular fallback."""
    mu1 = np.atleast_1d(_finite_array(mu1, "first mean", dtype=None))
    mu2 = np.atleast_1d(_finite_array(mu2, "second mean", dtype=None))
    sigma1 = np.atleast_2d(_finite_array(sigma1, "first covariance", dtype=None))
    sigma2 = np.atleast_2d(_finite_array(sigma2, "second covariance", dtype=None))
    if mu1.ndim != 1 or mu1.shape != mu2.shape or sigma1.shape != sigma2.shape or sigma1.shape != (len(mu1), len(mu1)):
        raise ValueError("FID means/covariances have incompatible shapes")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("FID eps must be positive and finite")
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("FID covariance square root has a nontrivial imaginary component")
        covmean = covmean.real
    result = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    if not np.isfinite(result):
        raise ValueError("FID calculation produced a nonfinite result")
    return float(result)


def _feature_matrix(values: Any, name: str) -> np.ndarray:
    matrix = _finite_array(values, name)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N, D], N >= 2 and D >= 1")
    return matrix


def calculate_diversity(data: Any, first_indices: Any, second_indices: Any) -> float:
    """Released float32 Euclidean accumulation for explicitly supplied pairs."""
    matrix = _feature_matrix(data, "diversity features")
    first, second = _numpy(first_indices), _numpy(second_indices)
    if first.ndim != 1 or first.shape != second.shape or len(first) == 0:
        raise ValueError("diversity pair indices must have the same nonempty vector shape")
    if first.dtype.kind not in "iu" or second.dtype.kind not in "iu":
        raise ValueError("diversity indices must be integers")
    if (first < 0).any() or (second < 0).any() or (first >= len(matrix)).any() or (second >= len(matrix)).any():
        raise ValueError("diversity indices are out of range")
    d = torch.from_numpy(matrix)
    total = torch.zeros((), dtype=torch.float32)
    for first_idx, second_idx in zip(first, second):
        total += torch.dist(d[int(first_idx)], d[int(second_idx)])
    result = float(total / len(first))
    if not np.isfinite(result):
        raise ValueError("diversity calculation produced a nonfinite result")
    return result


def _sampled_diversity(values: np.ndarray, rng: np.random.RandomState, pairs: int) -> float:
    first = rng.randint(0, len(values), pairs)
    second = rng.randint(0, len(values), pairs)
    return calculate_diversity(values, first, second)


def _multimodality(values: np.ndarray, group_ids: Any, seed: int, pairs: int):
    if isinstance(group_ids, (str, bytes)) or len(group_ids) != len(values):
        raise ValueError("prompt_group_ids must provide one identity per generated sequence")
    groups: OrderedDict[Any, list[int]] = OrderedDict()
    for index, group_id in enumerate(group_ids):
        if isinstance(group_id, np.generic):
            group_id = group_id.item()
        if not isinstance(group_id, (str, int)) or isinstance(group_id, bool) or (isinstance(group_id, str) and not group_id.strip()):
            raise ValueError("prompt group identities must be nonempty strings or integer IDs")
        groups.setdefault(group_id, []).append(index)
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("each prompt group needs at least two generated samples")
    rng = np.random.RandomState(seed)
    scores = []
    for indices in groups.values():
        first = rng.randint(0, len(indices), pairs)
        second = rng.randint(0, len(indices) - 1, pairs)
        second += second >= first
        indices = np.asarray(indices)
        scores.append(calculate_diversity(values, indices[first], indices[second]))
    return float(np.mean(scores)), [len(indices) for indices in groups.values()]


def evaluate_native_features(
    train_features: Any, generated_features: Any, *, seed: int = 42,
    diversity_pairs: int = 10000, expected_labels: Any = None,
    generated_predictions: Any = None, prompt_group_ids: Any = None,
    reference_features: Any = None, multimodality_pairs_per_group: int = 1000,
) -> dict[str, Any]:
    """Evaluate released FID/diversity; optional prompt-MM is a declared extension.

    Primary FID is relative to training features, as in metric.py. A supplied
    real test cohort yields separately named diagnostics. All feature sets use
    the same training feature z-score, without epsilon or refitting.
    """
    train = _feature_matrix(train_features, "train features")
    generated = _feature_matrix(generated_features, "generated features")
    if generated.shape[1] != train.shape[1]:
        raise ValueError("train/generated feature dimensions differ")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    seed = int(seed)
    diversity_pairs = _positive_integer(diversity_pairs, "diversity_pairs")
    mm_pairs = _positive_integer(multimodality_pairs_per_group, "multimodality_pairs_per_group")
    mean, std = np.mean(train, axis=0), np.std(train, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("training feature dimensions have zero/nonfinite standard deviation; upstream z-score is undefined")

    def standardized(values):
        result = (values - mean[None, :]) / std[None, :]
        if not np.isfinite(result).all():
            raise ValueError("feature standardization produced nonfinite values")
        return result

    train, generated = standardized(train), standardized(generated)

    def fid(a, b):
        return frechet_distance(np.mean(a, axis=0), np.cov(a, rowvar=False),
                                np.mean(b, axis=0), np.cov(b, rowvar=False))

    result: dict[str, Any] = {
        "protocol": "ccdm-released-classifier-v1",
        "fid_train": fid(train, generated),
        "diversity_generated": _sampled_diversity(generated, np.random.RandomState(seed), diversity_pairs),
        "fid_test_diagnostic": None, "diversity_reference": None, "accuracy": None,
        "multimodality": None,
        "multimodality_reason": "No prompt_group_ids supplied; motion-class diversity is not prompt multimodality.",
        "metadata": {
            "train_count": len(train), "generated_count": len(generated), "reference_count": 0,
            "feature_dim": train.shape[1], "seed": seed, "diversity_pairs": diversity_pairs,
            "feature_mean": mean.tolist(), "feature_std": std.tolist(),
            "feature_normalization": "training feature population mean/std; no epsilon",
            "fid_reference": "train", "diversity_sampling": "seeded RandomState iid pairs with replacement; self pairs allowed; float32 accumulation",
            "multimodality_groups": 0, "multimodality_group_sizes": [],
            "multimodality_pairs_per_group": mm_pairs,
            "multimodality_protocol": "optional extension: same-prompt distinct samples; pairs sampled with replacement; equal group weight; not upstream shuffled-class diversity",
            "accuracy_count": 0,
        },
    }
    if reference_features is not None:
        reference = _feature_matrix(reference_features, "reference features")
        if reference.shape[1] != train.shape[1]:
            raise ValueError("train/reference feature dimensions differ")
        reference = standardized(reference)
        result["fid_test_diagnostic"] = fid(reference, generated)
        result["diversity_reference"] = _sampled_diversity(reference, np.random.RandomState(seed), diversity_pairs)
        result["metadata"]["reference_count"] = len(reference)
    if expected_labels is not None or generated_predictions is not None:
        if expected_labels is None or generated_predictions is None:
            raise ValueError("accuracy requires both expected_labels and generated_predictions")
        expected, predicted = _numpy(expected_labels), _numpy(generated_predictions)
        for values in (expected, predicted):
            if values.shape != (len(generated),) or values.dtype.kind not in "iu" or (values < 0).any() or (values >= 6).any():
                raise ValueError("accuracy labels/predictions must be one integer in [0, 6) per generated sample")
        result["accuracy"] = float(np.mean(expected == predicted))
        result["metadata"]["accuracy_count"] = len(expected)
        result["metadata"]["accuracy_class_counts"] = np.bincount(expected, minlength=6).tolist()
    if prompt_group_ids is not None:
        score, sizes = _multimodality(generated, prompt_group_ids, seed, mm_pairs)
        result["multimodality"], result["multimodality_reason"] = score, None
        result["metadata"]["multimodality_groups"] = len(sizes)
        result["metadata"]["multimodality_group_sizes"] = sizes
    return result
