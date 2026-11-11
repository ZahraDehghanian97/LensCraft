#!/usr/bin/env python3
"""Score native CCDM 5D trajectories with its released motion-classifier method.

Requires a motion-classifier checkpoint and its labeled train/test dataset.
The diffusion generator's weight/latest.pth is not a classifier checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-data", type=Path, required=True,
                        help="Trusted upstream .npy: train_cam/test_cam/train_label/test_label.")
    parser.add_argument("--generated", type=Path, required=True,
                        help="Trusted .npy dict (result/label), or .npz (trajectories/labels). Raw native5D.")
    parser.add_argument("--classifier-origin", choices=("original", "retrained", "unknown"), default="unknown")
    parser.add_argument("--training-manifest", type=Path,
                        help="Bind a reconstructed classifier to its training split and normalization.")
    parser.add_argument("--cohort", type=Path,
                        help="Frozen matched source IDs/labels; requires --generation-manifest.")
    parser.add_argument("--generation-manifest", type=Path,
                        help="Exporter complete.json binding generated data to the matched cohort.")
    parser.add_argument("--output", type=Path, required=True, help="New result directory.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diversity-pairs", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.diversity_pairs < 1 or args.seed < 0:
        parser.error("batch-size/diversity-pairs must be positive and seed nonnegative")
    if args.classifier_origin == "retrained" and args.training_manifest is None:
        parser.error("--training-manifest is required for a retrained classifier")
    if (args.cohort is None) != (args.generation_manifest is None):
        parser.error("--cohort and --generation-manifest must be supplied together")
    if args.cohort is not None and args.training_manifest is None:
        parser.error("--training-manifest is required to verify a matched cohort")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_payload(path):
    import numpy as np

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    # Original CCDM artifacts are pickled numpy dictionaries; use trusted files.
    value = np.load(path, allow_pickle=True)
    if not isinstance(value, np.ndarray) or value.shape != () or not isinstance(value.item(), dict):
        raise ValueError(f"{path}: expected an upstream numpy dictionary or numeric .npz")
    return value.item()


def native_rows(value, name):
    import numpy as np

    rows = [np.asarray(row) for row in value]
    if not rows:
        raise ValueError(f"{name}: empty trajectory collection")
    for row in rows:
        if row.ndim != 2 or row.shape[1] != 5 or not len(row) or not np.isfinite(row).all():
            raise ValueError(f"{name}: each trajectory must be finite, nonempty raw native [T,5]")
    return rows


def labels(value, count, name):
    import numpy as np

    result = np.asarray(value)
    if result.shape != (count,) or result.dtype.kind not in "iu" or not ((result >= 0) & (result < 6)).all():
        raise ValueError(f"{name}: expected {count} integer labels in [0,5]")
    return result


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_matched_cohort(args, data, payload, generated, expected, identities, training_manifest, frames):
    """Bind ordered outputs to classifier-held-out IDs, without claiming paper reproduction."""
    import numpy as np

    cohort_identity = file_identity(args.cohort)
    generation_identity = file_identity(args.generation_manifest)
    cohort = json.loads(args.cohort.read_text())
    generation = json.loads(args.generation_manifest.read_text())
    if (cohort.get("protocol") != "ccdm-native-shared-cohort-v1"
            or generation.get("protocol") != "ccdm-native-matched-generation-v1"):
        raise ValueError("Matched cohort/generation protocol is not recognized")

    def same_hash(actual, expected_hash, name):
        if not isinstance(actual, str) or not actual or actual != expected_hash:
            raise ValueError("Matched cohort binding differs: " + name)

    same_hash(generation.get("generated", {}).get("sha256"), identities["generated"]["sha256"], "generated artifact")
    same_hash(generation.get("cohort", {}).get("sha256"), cohort_identity["sha256"], "cohort artifact")
    same_hash(cohort.get("classifier_data_sha256"), identities["classifier_data"]["sha256"], "classifier data")
    same_hash(cohort.get("data_manifest_sha256"), training_manifest.get("training", {}).get("data_manifest_sha256"), "training data manifest")
    same_hash(cohort.get("source_data_sha256"), generation.get("source_data", {}).get("sha256"), "source corpus")
    count, repeats = len(generated), generation.get("repeats")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Matched generation repeats must be a positive integer")
    if (generation.get("state") != "complete"
            or any(type(generation.get(key)) is not int or generation[key] != count
                   for key in ("output_count", "processed_samples"))
            or type(generation.get("frames")) is not int or generation["frames"] != frames
            or any(len(row) != frames for row in generated)
            or type(generation.get("target_camera_used")) is not int or generation["target_camera_used"] != 0):
        raise ValueError("Matched generation must be complete with matching count/frames and no target camera")

    def source_ids(value, count, name):
        result = np.asarray(value)
        if result.shape != (count,) or result.dtype.kind not in "iu" or (result < 0).any():
            raise ValueError(name + " must contain one nonnegative integer source ID per row")
        return result

    cohort_labels = np.asarray(cohort.get("labels"))
    if cohort_labels.ndim != 1 or not len(cohort_labels):
        raise ValueError("Matched cohort needs nonempty labels and source IDs")
    cohort_labels = labels(cohort_labels, len(cohort_labels), "cohort labels")
    cohort_ids = source_ids(cohort.get("source_indices"), len(cohort_labels), "cohort source IDs")
    if (count != len(cohort_ids) * repeats
            or type(generation.get("source_draws")) is not int or generation["source_draws"] != len(cohort_ids)):
        raise ValueError("Matched generation count does not match cohort draws and repeats")
    actual_ids = source_ids(payload.get("source_indices"), count, "generated source IDs")
    if (not np.array_equal(actual_ids, np.repeat(cohort_ids, repeats))
            or expected is None or not np.array_equal(expected, np.repeat(cohort_labels, repeats))):
        raise ValueError("Generated source IDs/labels do not match the ordered repeated cohort")
    test_ids = source_ids(data.get("test_source_indices"), len(data["test_label"]), "classifier test source IDs")
    train_ids = source_ids(data.get("train_source_indices"), len(data["train_label"]), "classifier train source IDs")
    if (len(set(test_ids.tolist())) != len(test_ids) or len(set(train_ids.tolist())) != len(train_ids)
            or set(test_ids.tolist()) & set(train_ids.tolist())):
        raise ValueError("Classifier source IDs are duplicated or overlap train/test")
    test_label_map = dict(zip(test_ids.tolist(), np.asarray(data["test_label"]).tolist()))
    if any(test_label_map.get(int(index)) != int(label) for index, label in zip(cohort_ids, cohort_labels)):
        raise ValueError("Matched cohort IDs/labels differ from the classifier test split")
    identities.update(cohort=cohort_identity, generation_manifest=generation_identity)
    return {"source_draws": len(cohort_ids), "repeats": repeats, "generated_count": count,
            "classifier_heldout_verified": True, "generator_heldout_verified": False}


def evaluate(args):
    import numpy as np
    import torch
    from testing.ccdm_native import (
        evaluate_native_features, extract_native_features, load_classifier,
        prepare_native_trajectories,
    )

    model, classifier_metadata = load_classifier(args.classifier_checkpoint, device=args.device)
    data = load_payload(args.classifier_data)
    required = ("train_cam", "test_cam", "train_label", "test_label")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("Classifier split is missing " + ", ".join(missing) +
                         "; the released cam/info generator corpus is not this labeled split.")
    train = native_rows(data["train_cam"], "train_cam")
    test = native_rows(data["test_cam"], "test_cam")
    train_labels = labels(data["train_label"], len(train), "train_label")
    test_labels = labels(data["test_label"], len(test), "test_label")
    if set(train_labels.tolist()) != set(range(6)) or set(test_labels.tolist()) != set(range(6)):
        raise ValueError("Classifier train and test splits must each cover all six native motion labels")
    # Match classify.py: statistics over all unpadded train + test frames.
    frames = np.concatenate(train + test, axis=0)
    mean, std = frames.mean(axis=0), frames.std(axis=0)
    del frames
    identities = {"classifier_checkpoint": file_identity(args.classifier_checkpoint),
                  "classifier_data": file_identity(args.classifier_data),
                  "generated": file_identity(args.generated)}
    training_manifest = None
    if args.training_manifest is not None:
        training_manifest = json.loads(args.training_manifest.read_text())
        for name in ("classifier_checkpoint", "classifier_data"):
            if training_manifest.get(name, {}).get("sha256") != identities[name]["sha256"]:
                raise ValueError("Training manifest does not match " + name)
        normalization = training_manifest.get("normalization", {})
        if not (np.array_equal(mean, normalization.get("mean")) and np.array_equal(std, normalization.get("std"))):
            raise ValueError("Training manifest normalization does not match classifier data")
        if training_manifest.get("frames") != model.frames:
            raise ValueError("Training manifest frame count does not match classifier checkpoint")
        identities["training_manifest"] = file_identity(args.training_manifest)

    payload = load_payload(args.generated)
    generated = native_rows(payload.get("result", payload.get("trajectories", [])), "generated")
    expected = payload.get("label", payload.get("labels"))
    if expected is not None:
        expected = labels(expected, len(generated), "generated labels")
    matched_cohort = None
    if args.cohort is not None:
        matched_cohort = verify_matched_cohort(
            args, data, payload, generated, expected, identities, training_manifest, model.frames,
        )
    groups = payload.get("prompt_group_ids")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(generated),):
            raise ValueError("prompt_group_ids must have one ID per generated trajectory")
        groups = groups.tolist()

    def extract(rows):
        prepared = prepare_native_trajectories(rows, mean, std, model.frames)
        return extract_native_features(model, prepared, batch_size=args.batch_size, device=args.device)

    train_result, test_result, generated_result = extract(train), extract(test), extract(generated)
    report = evaluate_native_features(
        train_result["features"], generated_result["features"], seed=args.seed,
        diversity_pairs=args.diversity_pairs, expected_labels=expected,
        generated_predictions=generated_result["predictions"] if expected is not None else None,
        prompt_group_ids=groups,
        reference_features=test_result["features"],
    )
    report.update({
        "status": "complete",
        "classifier": classifier_metadata,
        "classifier_origin_user_declared": args.classifier_origin,
        "matched_cohort_verified": matched_cohort is not None,
        "matched_cohort": matched_cohort,
        "paper_table_reproduction_verified": False,
        "classifier_real_test_accuracy": float(np.mean(test_result["predictions"] == test_labels)),
        "native_input_normalization": {"mean": mean.tolist(), "std": std.tolist(),
            "basis": "all unpadded classifier train+test frames, matching released classify.py"},
        "temporal_policy": {"classifier_frames": model.frames,
            "generated_original_lengths": sorted({len(row) for row in generated}),
            "padding": "repeat final frame after normalization; no resampling or truncation"},
        "inputs": identities,
        "sources": {"classify.py": file_identity(REPO_ROOT / "third_parties/Camera-control/[2024][EG]Text+keyframe/classify.py"),
                    "metric.py": file_identity(REPO_ROOT / "third_parties/Camera-control/[2024][EG]Text+keyframe/metric.py"),
                    "runner": file_identity(Path(__file__)),
                    "native_evaluator": file_identity(REPO_ROOT / "src/testing/ccdm_native.py")},
        "limits": ["Released classifier code defaults to 300 frames; the paper evaluates 60-frame generation.",
                   "Original paper split/checkpoint identities and 12,000-sample sampling must be verified separately.",
                   "R Prec here is six-motion classification accuracy, not text-retrieval recall.",
                   "Prompt multimodality needs repeated generation under identical text conditions; motion class is not a prompt group."],
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": args.device},
    })
    np.savez_compressed(args.output / "classifier_features.npz",
                        train=train_result["features"], test=test_result["features"],
                        generated=generated_result["features"], predictions=generated_result["predictions"])
    return report


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    required = [args.classifier_checkpoint, args.classifier_data, args.generated]
    if args.training_manifest is not None:
        required.append(args.training_manifest)
    if args.cohort is not None:
        required.extend((args.cohort, args.generation_manifest))
    missing = [str(path) for path in required
               if not path.is_file()]
    if missing:
        write_json(args.output / "status.json", {"status": "missing_assets", "missing": missing,
            "required_classifier_data_keys": ["train_cam", "test_cam", "train_label", "test_label"],
            "note": "The diffusion weight/latest.pth and cam/info corpus do not replace classifier assets."})
        print("Native evaluation requires missing assets; see " + str(args.output / "status.json"), file=sys.stderr)
        return 2
    try:
        report = evaluate(args)
    except Exception as error:
        write_json(args.output / "status.json", {"status": "failed", "error_type": type(error).__name__,
                                                "reason": str(error)})
        raise
    write_json(args.output / "results.json", report)
    write_json(args.output / "status.json", {"status": "complete", "results": "results.json"})
    print(json.dumps({key: report[key] for key in
          ("fid_train", "accuracy", "diversity_generated", "multimodality")}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
