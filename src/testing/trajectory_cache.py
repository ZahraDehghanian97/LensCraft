from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


TRAJECTORY_CACHE_VERSION = 5
# Old E.T. evaluations restarted the same latent seeds in every batch.
# Only E.T.'s cache key changes when its generation protocol changes.
ET_GENERATION_SEED_VERSION = 2

_PATH_INPUT_KEYS = frozenset(
    {
        "cache_file",
        "checkpoint_path",
        "clip_embeddings_cache",
        "config",
        "data_dir",
        "data_path",
        "dataset_dir",
        "et_cin_lang_path",
        "project_config_dir",
    }
)
_SIMULATION_DATA_GLOBS = (
    "manifest.json",
    "parameter_dictionary.msgpack",
    "simulation_*.msgpack",
)


def _stat_record(path: Path, relative_path: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative_path,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mode": stat.st_mode,
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "ctimeNs": stat.st_ctime_ns,
    }


def stat_path_provenance(
    path: str | Path,
    *,
    include_globs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fingerprint a file or a directory from filesystem metadata only.

    Directory traversal is restricted to the explicitly configured input root.
    ``include_globs`` can narrow a large directory to the files that actually
    define an input contract (as is done for simulator exports).  Size, inode,
    and nanosecond modification/change times make ordinary in-place and atomic
    replacements visible without reading large checkpoints or dataset payloads.
    """

    configured_path = Path(path).expanduser()
    try:
        resolved_path = configured_path.resolve(strict=False)
    except OSError:
        resolved_path = configured_path.absolute()

    result: dict[str, Any] = {
        "configuredPath": str(configured_path),
        "resolvedPath": str(resolved_path),
    }
    if not resolved_path.exists():
        result["kind"] = "missing"
        return result

    if resolved_path.is_file():
        result.update(
            {
                "kind": "file",
                "entryCount": 1,
                "metadataHash": _metadata_hash(
                    [_stat_record(resolved_path, ".")]
                ),
            }
        )
        return result

    if not resolved_path.is_dir():
        result["kind"] = "other"
        result["metadataHash"] = _metadata_hash(
            [_stat_record(resolved_path, ".")]
        )
        return result

    patterns = tuple(include_globs or ())
    digest = hashlib.sha256()
    entry_count = 0

    def add_record(record: Mapping[str, Any]) -> None:
        nonlocal entry_count
        digest.update(
            json.dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
        entry_count += 1

    # For a filtered directory, the selected file inventory is the complete
    # input contract.  Hashing the directory itself would make unrelated
    # derived/output files change its mtime and spuriously invalidate the key.
    if not patterns:
        add_record(_stat_record(resolved_path, "."))
    for directory, directory_names, file_names in os.walk(
        resolved_path, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        for file_name in file_names:
            file_path = directory_path / file_name
            relative_path = file_path.relative_to(resolved_path).as_posix()
            if patterns and not any(
                fnmatch.fnmatch(file_name, pattern) for pattern in patterns
            ):
                continue
            try:
                add_record(_stat_record(file_path, relative_path))
            except OSError as error:
                add_record(
                    {
                        "path": relative_path,
                        "statError": error.errno,
                    }
                )

    result.update(
        {
            "kind": "directory",
            "entryCount": entry_count,
            "metadataHash": digest.hexdigest(),
            "includeGlobs": patterns,
        }
    )
    return result


def _metadata_hash(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iter_configured_input_paths(
    value: Any,
    prefix: str,
    *,
    simulation_context: bool = False,
):
    if isinstance(value, Mapping):
        target = str(value.get("_target_", ""))
        current_simulation_context = (
            simulation_context or "SimulationDataset" in target
        )
        for key in sorted(value, key=str):
            item = value[key]
            item_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key in _PATH_INPUT_KEYS and isinstance(item, (str, Path)):
                if str(item) not in ("", "None", "none", "null"):
                    yield (
                        item_prefix,
                        item,
                        _SIMULATION_DATA_GLOBS
                        if current_simulation_context and key == "data_path"
                        else None,
                    )
            if isinstance(item, (Mapping, list, tuple)):
                yield from _iter_configured_input_paths(
                    item,
                    item_prefix,
                    simulation_context=current_simulation_context,
                )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_configured_input_paths(
                item,
                f"{prefix}[{index}]",
                simulation_context=simulation_context,
            )


def configured_input_provenance(
    config: Mapping[str, Any],
    *,
    resolve_path: Callable[[str], str | Path] = Path,
) -> dict[str, Any]:
    """Return provenance for inputs that can change generated trajectories."""

    sections: list[tuple[str, Any]] = []
    data = config.get("data")
    if isinstance(data, Mapping):
        sections.append(("data", data))

    training = config.get("training")
    model_type = "simulation"
    if isinstance(training, Mapping):
        model = training.get("model")
        if isinstance(model, Mapping):
            data_format = model.get("data_format")
            if isinstance(data_format, Mapping):
                model_type = str(data_format.get("type", model_type))
            inference = model.get("inference")
            if isinstance(inference, Mapping):
                sections.append(("training.model.inference", inference))

    # Baseline trajectories can use the LensCraft reference checkpoint for
    # initial-position alignment.  For LensCraft evaluation it is not loaded.
    if model_type != "simulation":
        reference = config.get("ref_model")
        if isinstance(reference, Mapping):
            inference = reference.get("inference")
            if isinstance(inference, Mapping):
                sections.append(("ref_model.inference", inference))

    provenance = {}
    for section_prefix, section in sections:
        for label, raw_path, include_globs in _iter_configured_input_paths(
            section, section_prefix
        ):
            try:
                resolved = resolve_path(str(raw_path))
                provenance[label] = stat_path_provenance(
                    resolved, include_globs=include_globs
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                provenance[label] = {
                    "configuredPath": str(raw_path),
                    "resolutionError": type(error).__name__,
                }
    return provenance


def build_trajectory_cache_key(
    config: Mapping[str, Any],
    *,
    resolve_path: Callable[[str], str | Path] = Path,
) -> str:
    """Hash resolved config plus the current identity of its input files."""

    cache_config = dict(config)
    cache_config.pop("trajectory_cache", None)
    cache_config.pop("trajectory_cache_dir", None)
    source = {
        "config": cache_config,
        "inputProvenance": configured_input_provenance(
            cache_config, resolve_path=resolve_path
        ),
    }
    model_type = (
        cache_config.get("training", {}).get("model", {}).get("data_format", {}).get("type")
    )
    if model_type == "et":
        source["etGenerationSeedVersion"] = ET_GENERATION_SEED_VERSION
    canonical = json.dumps(
        source, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_trajectory(value, label: str, expected_sequence_length: int) -> int:
    if (
        not torch.is_tensor(value)
        or value.ndim != 3
        or tuple(value.shape[-2:]) != (expected_sequence_length, 6)
        or value.shape[0] < 1
        or not torch.isfinite(value).all()
    ):
        raise ValueError(
            f"{label} must be a finite [batch, {expected_sequence_length}, 6] tensor"
        )
    return value.shape[0]


def validate_cached_metric_batch(
    batch,
    required_metric_items: Sequence[str],
    *,
    model_type: str,
    require_encoder_features: bool,
    expected_token_count: int,
    expected_embedding_dim: int,
    expected_sequence_length: int = 30,
) -> None:
    """Reject stale/corrupt cached trajectories before metric evaluation."""

    if not isinstance(batch, Mapping):
        raise ValueError("cached batch must be a mapping")

    required = set(required_metric_items)
    if model_type != "lens_craft":
        trajectories = batch.get("trajectories")
        if not isinstance(trajectories, Mapping) or not required.issubset(
            trajectories
        ):
            raise ValueError("cache does not contain every requested metric mode")
        for metric_item in required:
            _validate_trajectory(
                trajectories[metric_item], f"{metric_item} trajectory",
                expected_sequence_length,
            )
        padding_masks = batch.get("padding_masks")
        if padding_masks is not None:
            if not isinstance(padding_masks, Mapping) or not required.issubset(padding_masks):
                raise ValueError("cache does not contain every requested padding mask")
            for metric_item in required:
                mask = padding_masks[metric_item]
                if (
                    not torch.is_tensor(mask)
                    or mask.dtype != torch.bool
                    or mask.shape != trajectories[metric_item].shape[:2]
                ):
                    raise ValueError(f"{metric_item} padding mask must match its trajectory")
        return

    items = batch.get("items")
    if not isinstance(items, Mapping) or not required.issubset(items):
        raise ValueError("cache does not contain every requested metric mode")

    for metric_item in required:
        item = items[metric_item]
        if not isinstance(item, Mapping):
            raise ValueError(f"{metric_item} cache entry must be a mapping")
        batch_size = _validate_trajectory(
            item.get("trajectory"), f"{metric_item} trajectory",
            expected_sequence_length,
        )
        features = item.get("encoder_features")
        if features is None:
            if require_encoder_features:
                raise ValueError(
                    f"{metric_item} cache entry has no encoder features"
                )
            continue
        if (
            not torch.is_tensor(features)
            or features.ndim != 3
            or tuple(features.shape)
            != (batch_size, expected_token_count, expected_embedding_dim)
            or not torch.isfinite(features).all()
        ):
            raise ValueError(
                f"{metric_item} encoder features must have shape "
                f"[batch, {expected_token_count}, {expected_embedding_dim}]"
            )
