"""Load and identify the fixed semantic evaluator independently of a generator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from testing.metrics.semantic_evaluator import (
    FROZEN_ENCODER_SOURCE_SHA256,
    FROZEN_POSITION_SOURCE_SHA256,
    FrozenSemanticEvaluator,
)
from testing.metrics.semantic_vocabulary import load_metric_clip_embeddings


def checkpoint_identity(path: str | Path) -> dict:
    """Content identity remains useful when artifacts move between machines."""
    resolved = Path(hydra.utils.to_absolute_path(str(path))).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluator artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def _config_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _clip_name(value: str) -> str:
    return value if value.startswith("openai/") else f"openai/{value}"


def _evaluator_model_config(ref_cfg: DictConfig) -> tuple[DictConfig, dict, dict | None]:
    config_path = ref_cfg.inference.get("config")
    module = ref_cfg.module
    clip = ref_cfg.clip
    config_identity = None
    if config_path not in (None, "", "None"):
        config_identity = checkpoint_identity(config_path)
        saved = OmegaConf.load(config_identity["path"])
        # A saved training configuration can also contain ref_model. The
        # checkpoint contains training.model weights, not that other reference.
        trained_module = OmegaConf.select(saved, "training.model.module")
        if trained_module is not None:
            module = trained_module
            clip = saved.get("clip", clip)
        elif "_target_" in saved:
            module = saved
        elif "module" in saved:
            module = saved.module
            clip = saved.get("clip", clip)
        elif OmegaConf.select(saved, "ref_model.module") is not None:
            module = saved.ref_model.module
            clip = saved.ref_model.get("clip", saved.get("clip", clip))
        else:
            raise ValueError(
                "Semantic evaluator config must contain training.model.module, "
                "module, ref_model.module, or a standalone model _target_."
            )

    # Resolve inside the SAVED config before detaching it. In particular,
    # ${training.use_merged_memory} must never resolve against the ablation.
    resolved_module = OmegaConf.create(OmegaConf.to_container(module, resolve=True))
    resolved_clip = OmegaConf.to_container(clip, resolve=True)
    return resolved_module, resolved_clip, config_identity


def load_semantic_evaluator(
    ref_cfg: DictConfig, device, *, input_clip: DictConfig
):
    """Load only the historical trajectory encoder, never a generator.

    LensCraft checkpoints do not save their architecture hyperparameters. The
    saved config supplies encoder dimensions; its historical generator target
    and decoder settings are provenance only and are never instantiated.
    """
    checkpoint_path = ref_cfg.inference.get("checkpoint_path")
    if checkpoint_path in (None, "", "None"):
        raise ValueError(
            "A fixed semantic evaluator checkpoint is required. Set "
            "SEMANTIC_EVALUATOR_CHECKPOINT_PATH (or "
            "ref_model.inference.checkpoint_path) and optionally "
            "SEMANTIC_EVALUATOR_CONFIG_PATH to its saved training config. "
            "TEST_CHECKPOINT_PATH selects only the generator."
        )

    checkpoint = checkpoint_identity(checkpoint_path)
    module, clip, saved_config = _evaluator_model_config(ref_cfg)
    model_definition = OmegaConf.to_container(module, resolve=True)
    embedding_dim = int(module.get("latent_dim", 512))
    clip_model_name = _clip_name(str(clip["model_name"]))
    if (
        embedding_dim != int(clip["latent_dim"])
        or embedding_dim != int(input_clip.latent_dim)
        or clip_model_name != _clip_name(str(input_clip.model_name))
    ):
        raise ValueError(
            "Semantic evaluator and metric targets must use the "
            "same CLIP model and embedding dimension; evaluator expects "
            f"{clip_model_name} ({embedding_dim}), metric config uses "
            f"{input_clip.model_name} ({input_clip.latent_dim})."
        )

    encoder_keys = (
        "input_dim", "subject_dim", "nhead", "num_encoder_layers",
        "dim_feedforward", "dropout_rate", "seq_length", "latent_dim",
        "num_query_tokens",
    )
    encoder_definition = {key: module[key] for key in encoder_keys if key in module}
    evaluator = FrozenSemanticEvaluator(**encoder_definition)
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    evaluator.load_metric_checkpoint(payload).to(device)
    raw_clip_name = str(clip["model_name"])
    cache_file = clip.get("cache_file") or os.path.join(
        os.environ.get("CLIP_EMBEDDINGS_CACHE_DIR", "."),
        f"clip_embeddings_{raw_clip_name}_cache.pkl",
    )
    evaluator.clip_embeddings = load_metric_clip_embeddings(
        clip_model_name=raw_clip_name,
        cache_file=cache_file,
        chunk_size=int(clip.get("chunk_size", 100)),
        embedding_dimension=embedding_dim,
    )
    evaluator.eval()
    evaluator.requires_grad_(False)
    definition_hash = _config_hash(model_definition)
    evaluator.evaluation_provenance = {
        "protocol_version": 1,
        "checkpoint": checkpoint,
        "config": saved_config,
        "model_definition": model_definition,
        "model_definition_sha256": definition_hash,
        "clip_model_name": clip_model_name,
        "embedding_dim": embedding_dim,
        "frozen": True,
        "implementation": {
            "target": "testing.metrics.semantic_evaluator.FrozenSemanticEvaluator",
            "encoder_definition": encoder_definition,
            "historical_encoder_source_sha256": FROZEN_ENCODER_SOURCE_SHA256,
            "historical_position_source_sha256": FROZEN_POSITION_SOURCE_SHA256,
            "checkpoint_load": evaluator.checkpoint_load_info,
            "generation_supported": False,
        },
        "fingerprint": _config_hash({
            "checkpoint_sha256": checkpoint["sha256"],
            "model_definition_sha256": definition_hash,
            "clip_model_name": clip_model_name,
        }),
    }
    return evaluator


def identify_clatr_evaluator(extractor, checkpoint_path, backend: str) -> dict:
    """Record the actual CLaTr weights used, including default checkpoint paths."""
    checkpoint = checkpoint_identity(checkpoint_path)
    identity = {
        "backend": backend,
        "checkpoint": checkpoint,
        "feature_dim": int(extractor.feature_dim),
    }
    identity["fingerprint"] = _config_hash({
        "backend": backend,
        "checkpoint_sha256": checkpoint["sha256"],
        "feature_dim": identity["feature_dim"],
    })
    extractor.evaluation_provenance = identity
    return identity
