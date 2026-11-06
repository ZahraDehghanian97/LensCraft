#!/usr/bin/env python3
"""Prepare a reproducible CCDM cohort and finalize fresh Gateway annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_once(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Refusing to replace a different cohort file: {path}")
    path.write_text(content)


def prepare(data_path, output, sample_count=None, seed=42):
    data_path = Path(data_path).resolve()
    raw = np.load(data_path, allow_pickle=True).item()
    count = len(raw["cam"])
    if count != len(raw["info"]):
        raise ValueError("Camera and caption counts differ")
    if sample_count is None:
        sample_count = count
    if not 1 <= sample_count <= count:
        raise ValueError("sample_count must be between 1 and dataset size")
    indices = sorted(np.random.default_rng(seed).choice(count, sample_count, replace=False).tolist())
    rows, lengths = [], []
    for index in indices:
        camera = np.asarray(raw["cam"][index])
        if camera.ndim != 2 or camera.shape[1] != 5 or len(camera) == 0 or not np.isfinite(camera).all():
            raise ValueError(f"Invalid native trajectory: {index}")
        sentences = raw["info"][index]
        if not isinstance(sentences, (list, tuple)) or not all(isinstance(s, str) for s in sentences):
            raise ValueError(f"Invalid native caption: {index}")
        prompt = " ".join(sentences)
        if not prompt.strip():
            raise ValueError(f"Empty native caption: {index}")
        rows.append({"id": f"ccdm-{index:06d}", "prompt": prompt})
        lengths.append(len(camera))
    manifest = {
        "dataset": str(data_path), "dataset_sha256": digest(data_path),
        "normalization_sha256": digest(data_path.parent / "Mean_Std.npy"),
        "total_samples": count, "selected_count": sample_count,
        "seed": seed, "selection": "sorted numpy.default_rng(seed).choice without replacement",
        "selected_indices": indices, "native_lengths": lengths,
        "unique_captions": len({r["prompt"] for r in rows}),
        "protocol": "Simulation-trained LensCraft cross-dataset evaluation; no CCDM training or checkpoint selection",
        "split_note": "Released data has no official split IDs; this is a newly seeded evaluation cohort",
        "translation_model": "deepseek/deepseek-v4.1-flash-1017", "reasoning_effort": "low",
        "native_caption_sha256": {r["id"]: hashlib.sha256(r["prompt"].encode()).hexdigest() for r in rows},
        "pose_assumptions": {"vertical_fov_deg": 45, "aspect": 16 / 9,
                             "stored_screen_y": "positive down",
                             "subject_origin": [0, 0, 0], "subject_dimensions": [0.5, 1.7, 0.3],
                             "subject_heading": "180-degree Y rotation; native front=+Z, simulation zero-heading front=-Z",
                             "roll": "inferred upright", "target_frames": 30},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_once(output / "indices.json", indices)
    write_once(output / "prompts.json", rows)
    write_once(output / "manifest.json", manifest)
    return manifest


def finalize(output, annotations, metadata):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from annotator.gateway import SCHEMA_HASH, validate_translation

    rows = json.loads((output / "prompts.json").read_text())
    translated = json.loads(annotations.read_text())
    provenance = json.loads(metadata.read_text())
    expected = {r["id"] for r in rows}
    if set(translated) != expected or set(provenance) != expected:
        raise ValueError("Translation and metadata must cover exactly the selected cohort")
    result = []
    for row in rows:
        item_id, text = row["id"], row["prompt"]
        info = provenance[item_id]
        if (info["source_prompt"] != text.strip() or info["schema_hash"] != SCHEMA_HASH
                or info["requested_model"] != "deepseek/deepseek-v4.1-flash-1017"
                or info["reasoning_effort"] != "low"):
            raise ValueError(f"Translation provenance mismatch: {item_id}")
        validate_translation(translated[item_id])
        result.append({"custom_id": item_id, "source_prompt": text, **translated[item_id]})
    write_once(output / "annotations.json", result)
    write_once(output / "annotation_manifest.json", {
        "count": len(result), "annotations_sha256": digest(output / "annotations.json"),
        "translation_sha256": digest(annotations), "metadata_sha256": digest(metadata),
        "schema_hash": SCHEMA_HASH,
        "response_models": sorted({str(x.get("response_model")) for x in provenance.values()}),
    })
    return len(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--translation-metadata", type=Path)
    args = parser.parse_args()
    if bool(args.annotations) != bool(args.translation_metadata):
        parser.error("Provide annotations and translation-metadata together")
    manifest = prepare(args.data_path, args.output_dir, args.sample_count, args.seed)
    if args.annotations:
        finalize(args.output_dir, args.annotations, args.translation_metadata)
    print(json.dumps({key: manifest[key] for key in ("total_samples", "selected_count", "unique_captions")}))


if __name__ == "__main__":
    main()
