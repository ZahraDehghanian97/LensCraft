#!/usr/bin/env python3
"""Freeze shared generation IDs from the reconstructed classifier's test split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(classifier_dir, output, count=12000, seed=42, unique=False):
    root, output = Path(classifier_dir), Path(output)
    if count < 1 or seed < 0:
        raise ValueError("count must be positive and seed nonnegative")
    if output.exists():
        raise FileExistsError(output)
    manifest_path, data_path = root / "data_manifest.json", root / "data.npy"
    manifest = json.loads(manifest_path.read_text())
    if digest(data_path) != manifest["artifacts"]["data.npy"]:
        raise ValueError("classifier split data differs from its preparation manifest")
    split = np.load(data_path, allow_pickle=True).item()
    indices, labels = np.asarray(split["test_source_indices"]), np.asarray(split["test_label"])
    train = np.asarray(split["train_source_indices"])
    if indices.ndim != 1 or not len(indices) or labels.shape != indices.shape:
        raise ValueError("test source indices and labels must have equal nonzero counts")
    if indices.dtype.kind not in "iu" or labels.dtype.kind not in "iu" or not ((labels >= 0) & (labels < 6)).all():
        raise ValueError("native source indices and six-class labels must be integers")
    if len(set(indices.tolist())) != len(indices) or set(indices.tolist()) & set(train.tolist()):
        raise ValueError("classifier source IDs overlap or contain duplicates")
    if unique and count > len(indices):
        raise ValueError("unique cohort cannot exceed classifier test split")
    selected = np.random.RandomState(seed).choice(len(indices), size=count, replace=not unique)
    result = {
        "protocol": "ccdm-native-shared-cohort-v1",
        "source_indices": indices[selected].tolist(), "labels": labels[selected].tolist(),
        "source_data_sha256": manifest["source_sha256"],
        "classifier_data_sha256": digest(data_path), "data_manifest_sha256": digest(manifest_path),
        "samples": count, "classifier_test_sources": len(indices), "seed": seed,
        "sampling": "without replacement" if unique else "with replacement",
        "classifier_heldout": True, "generator_heldout_verified": False,
        "note": "Same frozen IDs/labels for all generators; original generator training overlap is unknown.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unique", action="store_true", help="For a small set of prompts repeated for MM.")
    args = parser.parse_args()
    result = prepare(args.classifier_dir, args.output, args.count, args.seed, args.unique)
    print(json.dumps({key: result[key] for key in ("samples", "classifier_test_sources", "sampling")}))


if __name__ == "__main__":
    main()
