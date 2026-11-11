#!/usr/bin/env python3
"""Prepare the fixed ET pure-test intersection for fresh Gateway translations.

Only native split lists, captions, and modality filenames are read. Source data
is linked into OUTPUT/et-data without modifying it. Gateway batch output can be
finalized with --annotations and --translation-metadata into annotations.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


MODEL = "meta/muse-ember"
REASONING_EFFORT = "low"
# config.yaml -> traj+caption+char -> caption/caption.yaml. ETDataset uses
# caption_raw.caption, whose raw_caption_dir is ${dataset.dataset_dir}/caption.
CAPTION_DIRECTORY = "caption"
REQUIRED_FILES = (
    ("traj", ".txt"), ("intrinsics", ".npy"),
    ("char", ".npy"), ("char_raw", ".npy"),
    ("caption", ".txt"), ("cam_segments", ".npy"),
    ("caption_clip/seq", ".npy"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def unique_object(pairs):
    """Reject duplicate JSON IDs rather than silently accepting the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def read_split(path: Path):
    raw = path.read_bytes()
    ids = raw.decode("utf-8").splitlines()
    if not ids or any(not item or item.strip() != item or "/" in item or "\\" in item
                      or item in (".", "..") for item in ids):
        raise ValueError(f"Empty or invalid sample ID in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return ids, {"path": str(path), "sha256": sha256(raw), "count": len(ids)}


def check_write_once(path: Path, content: bytes):
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != content)):
        raise ValueError(f"Refusing to replace different existing output: {path}")


def write_once(path: Path, content: bytes):
    """Permit identical reruns while protecting an already prepared cohort."""
    check_write_once(path, content)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.chmod(0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(dataset_root: Path, output_dir: Path, limit: int | None = None):
    """Export a deterministic cohort and the exact text consumed by ETDataset."""
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {dataset_root}")
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise ValueError("--output-dir must be outside the original dataset")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")

    pure_ids, pure_info = read_split(dataset_root / "pure_test_split.txt")
    mixed_ids, mixed_info = read_split(dataset_root / "mixed_train_split.txt")
    mixed_set = set(mixed_ids)
    eligible = sorted(set(pure_ids) & mixed_set)
    excluded = sorted(set(pure_ids) - mixed_set)
    selected = eligible if limit is None else eligible[:limit]
    if not selected:
        raise ValueError("pure_test and mixed_train have no shared sample IDs")
    if limit is not None and limit > len(eligible):
        raise ValueError(f"--limit {limit} exceeds the {len(eligible)} eligible samples")

    prompts, caption_hashes = [], {}
    missing = []
    for item_id in selected:
        for directory, suffix in REQUIRED_FILES:
            path = dataset_root / directory / f"{item_id}{suffix}"
            if not path.is_file():
                missing.append(str(path))
        caption_path = dataset_root / CAPTION_DIRECTORY / f"{item_id}.txt"
        if not caption_path.is_file():
            continue
        raw = caption_path.read_bytes()
        # read_text uses universal newline conversion, matching DIRECTOR.load_txt.
        prompt = caption_path.read_text(encoding="utf-8")
        if not prompt.strip():
            raise ValueError(f"Empty native caption: {caption_path}")
        prompts.append({"id": f"et-{item_id}", "prompt": prompt})
        caption_hashes[item_id] = {
            "file_sha256": sha256(raw),
            "prompt_sha256": sha256(prompt.encode("utf-8")),
            "translation_prompt_sha256": sha256(prompt.strip().encode("utf-8")),
        }
    if missing:
        raise ValueError(f"Missing {len(missing)} required modality files; no samples dropped: "
                         + "; ".join(missing[:10]))

    view = output_dir / "et-data"
    if view.is_symlink() or (view.exists() and not view.is_dir()):
        raise ValueError(f"Dataset view must be an ordinary directory: {view}")
    directories = sorted(path for path in dataset_root.iterdir()
                         if path.is_dir() and not path.name.startswith("."))
    for source in directories:
        destination = view / source.name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"Existing dataset link points elsewhere: {destination}")
        elif destination.exists():
            raise ValueError(f"Refusing to replace existing dataset directory: {destination}")

    # DIRECTOR reads split files using split('\n'), so never append a newline.
    split_bytes = "\n".join(selected).encode("utf-8")
    prompt_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts).encode("utf-8")
    manifest = {
        "version": 1,
        "dataset_root": str(dataset_root),
        "dataset_view": str(view),
        "selection_rule": "pure_test_split.txt intersect mixed_train_split.txt",
        "selection_order": "lexicographic sample ID",
        "caption_source": "caption/{id}.txt (DIRECTOR default raw_caption_dir)",
        "translation_model": MODEL,
        "legacy_annotations_used": False,
        "original_pure_test_count": len(pure_ids),
        "mixed_train_count": len(mixed_ids),
        "intersection_count": len(eligible),
        "selected_count": len(selected),
        "excluded_pure_test_count": len(excluded),
        "limited_out_count": len(eligible) - len(selected),
        "limit": limit,
        "selected_ids": selected,
        "eligible_ids": eligible,
        "excluded_ids": excluded,
        "overlap": {"mixed_train_count": len(selected),
                    "note": "Every selected pure-test ID also belongs to mixed train; this is not a mixed-test cohort."},
        "source_splits": {"pure_test": pure_info, "mixed_train": mixed_info},
        "selected_split_sha256": sha256(split_bytes),
        "prompts_jsonl_sha256": sha256(prompt_bytes),
        "caption_hashes": caption_hashes,
        "source_access": "Modality directories are symlinked; this helper never writes source data.",
    }
    outputs = {view / "pure_test_split.txt": split_bytes,
               output_dir / "prompts.jsonl": prompt_bytes,
               output_dir / "manifest.json": json_bytes(manifest)}
    for path, content in outputs.items():
        check_write_once(path, content)
    view.mkdir(parents=True, exist_ok=True)
    for source in directories:
        destination = view / source.name
        if not destination.is_symlink():
            destination.symlink_to(source, target_is_directory=True)
    for path, content in outputs.items():
        write_once(path, content)
    return manifest, prompts


def finalize(output_dir: Path, annotations_path: Path, metadata_path: Path, prompts: list[dict]):
    """Validate current Gateway translations and provenance before ET conversion.

    Preparation stays dependency-free. Finalization lazily uses the translator's
    current schema validator so obsolete or malformed conditioning cannot pass.
    """
    source_path = str(Path(__file__).resolve().parents[1] / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from annotator.gateway import (
        GATEWAY_BASE_URL, SCHEMA_HASH, SYSTEM_PROMPT, _digest, validate_translation,
    )

    annotations = read_json(annotations_path)
    metadata = read_json(metadata_path)
    expected = {row["id"] for row in prompts}
    for label, value in (("annotations", annotations), ("translation metadata", metadata)):
        if not isinstance(value, dict):
            raise ValueError(f"Gateway {label} must be an ID-keyed object")
        if set(value) != expected:
            raise ValueError(f"Gateway {label} coverage mismatch: "
                             f"{len(expected - set(value))} missing, {len(set(value) - expected)} extra IDs")
    converted, settings = [], set()
    system_prompt_hash = _digest(SYSTEM_PROMPT)
    for row in prompts:
        item_id = row["id"]
        try:
            annotation = validate_translation(annotations[item_id])
        except ValueError as error:
            raise ValueError(f"{item_id}: {error}") from error
        record = metadata[item_id]
        if not isinstance(record, dict) or record.get("requested_model") != MODEL:
            raise ValueError(f"{item_id}: fresh translations must request {MODEL}")
        if record.get("response_model") != MODEL:
            raise ValueError(f"{item_id}: Gateway response model must equal {MODEL}")
        if record.get("source_prompt") != row["prompt"].strip():
            raise ValueError(f"{item_id}: translation source differs from the native caption")
        if record.get("schema_hash") != SCHEMA_HASH:
            raise ValueError(f"{item_id}: translation must use the current conditioning schema")
        if record.get("system_prompt_hash") != system_prompt_hash:
            raise ValueError(f"{item_id}: translation must use the current system prompt")
        response_format, max_tokens = record.get("response_format"), record.get("max_tokens")
        if response_format not in ("json_schema", "text"):
            raise ValueError(f"{item_id}: invalid Gateway response_format")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError(f"{item_id}: invalid Gateway max_tokens")
        reasoning_effort = record.get("reasoning_effort")
        if reasoning_effort != REASONING_EFFORT:
            raise ValueError(f"{item_id}: this evaluation requires reasoning_effort={REASONING_EFFORT}")
        expected_cache_key = _digest({
            "prompt": row["prompt"].strip(), "model": MODEL, "schema": SCHEMA_HASH,
            "system": SYSTEM_PROMPT, "response_format": response_format,
            "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
            "base_url": GATEWAY_BASE_URL,
        })
        if record.get("cache_key") != expected_cache_key:
            raise ValueError(f"{item_id}: translation settings fingerprint does not match")
        settings.add((response_format, max_tokens, reasoning_effort))
        converted.append({"custom_id": item_id, **annotation})
    if len(settings) != 1:
        raise ValueError("Gateway batch mixes different translation settings")
    response_format, max_tokens, reasoning_effort = settings.pop()
    output_dir = output_dir.expanduser().resolve()
    annotation_bytes = json_bytes(converted)
    provenance = {
        "annotation_count": len(converted), "requested_model": MODEL,
        "response_model": MODEL,
        "annotations_source": str(annotations_path.resolve()),
        "annotations_source_sha256": sha256(annotations_path.read_bytes()),
        "translation_metadata_source": str(metadata_path.resolve()),
        "translation_metadata_sha256": sha256(metadata_path.read_bytes()),
        "schema_hash": SCHEMA_HASH, "system_prompt_hash": system_prompt_hash,
        "response_format": response_format, "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "annotations_sha256": sha256(annotation_bytes),
        "prepared_manifest_sha256": sha256((output_dir / "manifest.json").read_bytes()),
    }
    outputs = {output_dir / "annotations.json": annotation_bytes,
               output_dir / "annotations.provenance.json": json_bytes(provenance)}
    for path, content in outputs.items():
        check_write_once(path, content)
    for path, content in outputs.items():
        write_once(path, content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Use a fixed prefix for a separately labeled smoke cohort")
    parser.add_argument("--annotations", type=Path, help="Fresh ID-keyed output from translate_prompt.py")
    parser.add_argument("--translation-metadata", type=Path, help="Matching Gateway --metadata-output JSON")
    args = parser.parse_args(argv)
    if bool(args.annotations) != bool(args.translation_metadata):
        parser.error("--annotations and --translation-metadata must be supplied together")
    try:
        manifest, prompts = prepare(args.dataset_root, args.output_dir, args.limit)
        if args.annotations:
            finalize(args.output_dir, args.annotations, args.translation_metadata, prompts)
        print(json.dumps({"selected_count": manifest["selected_count"],
                          "intersection_count": manifest["intersection_count"],
                          "original_pure_test_count": manifest["original_pure_test_count"],
                          "dataset_view": manifest["dataset_view"],
                          "annotations_finalized": bool(args.annotations)}))
        return 0
    except (OSError, ValueError, ImportError) as error:
        print(f"ET evaluation preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
