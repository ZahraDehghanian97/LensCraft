"""Lightweight metadata helpers for trusted simulator exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


DEFAULT_FIXED_POINT_SCALE = 1000.0
SIMULATION_FRAME_COUNT = 30
CACHE_METADATA_VERSION = 1


def resolve_dataset_root(data_path: Path) -> Path:
    """Return the directory that owns one simulator export.

    New archives are allowed to contain a dataset-scoped top-level directory so
    extracting two archives cannot mix their dictionaries and sample files.  A
    caller may therefore point either at that directory or at its extraction
    parent.  Multiple dictionaries are deliberately rejected: choosing one
    implicitly could decode samples with the wrong archive-local dictionary.
    """

    data_path = Path(data_path)
    candidates = sorted({
        path.parent.resolve()
        for path in data_path.rglob("parameter_dictionary.msgpack")
        if path.is_file()
    })
    if not candidates:
        return data_path
    if len(candidates) > 1:
        roots = ", ".join(str(path) for path in candidates)
        raise ValueError(
            "Multiple simulation datasets were found below "
            f"{data_path}: {roots}. Point data_path at one dataset directory."
        )
    return candidates[0]


def load_dataset_manifest(data_path: Path) -> Optional[Dict[str, Any]]:
    data_path = resolve_dataset_root(data_path)
    manifest_path = Path(data_path) / "manifest.json"
    if not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_fixed_point_scale(manifest: Optional[Dict[str, Any]]) -> float:
    if manifest is None:
        return DEFAULT_FIXED_POINT_SCALE
    return float(
        manifest.get("encoding", {}).get("fixedPointScale", DEFAULT_FIXED_POINT_SCALE)
    )


def compute_dataset_fingerprint(
    data_path: Path,
    manifest: Optional[Dict[str, Any]],
    simulation_files: Sequence[Path],
) -> str:
    """Build a cheap cache key without reading simulation payloads."""

    data_path = Path(data_path)
    dictionary_path = data_path / "parameter_dictionary.msgpack"
    dictionary_stat = dictionary_path.stat()
    # A manifest identifies a generated archive, but it does not prove that
    # extracted payloads were left untouched. Include the actual dictionary
    # and sample inventory for schema-v2 and legacy datasets alike so replacing
    # a file in place cannot silently reuse stale normalization/movement caches.
    source = {
        "manifest": manifest,
        "parameterDictionary": {
            "size": dictionary_stat.st_size,
            "mtimeNs": dictionary_stat.st_mtime_ns,
        },
        "simulationFiles": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtimeNs": path.stat().st_mtime_ns,
            }
            for path in sorted(
                (Path(path) for path in simulation_files),
                key=lambda path: path.name,
            )
        ],
    }

    encoded = json.dumps(
        source, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_metadata_matches(metadata_path: Path, dataset_fingerprint: str) -> bool:
    try:
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    return (
        metadata.get("version") == CACHE_METADATA_VERSION
        and metadata.get("datasetFingerprint") == dataset_fingerprint
    )


def write_cache_metadata(metadata_path: Path, dataset_fingerprint: str) -> None:
    metadata = {
        "version": CACHE_METADATA_VERSION,
        "datasetFingerprint": dataset_fingerprint,
    }
    with Path(metadata_path).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")
