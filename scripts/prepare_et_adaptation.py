#!/usr/bin/env python3
"""Prepare explicit ET training/validation inventories from existing annotations.

Selection is pure_train intersect annotated IDs, excluding every official test
split and any --excluded-cohort inputs. By default entire held-out source videos
are excluded and train/validation are grouped by source video. The source dataset
is never modified. Output/et-data links modality directories and contains separate
pure_train_split.txt and pure_val_split.txt files (without trailing newlines).
Consume these splits explicitly; do not randomly split the combined annotation
inventory with the generic CameraTrajectoryDataModule.

This helper uses only the Python standard library. --self-test runs fixture tests
without ET data. Annotation content is preserved; no translation or pose fitting
is performed here, and shape validation is not a claim of semantic correctness.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile


TEST_SPLITS = ("pure_test", "mixed_test", "full_test", "clatr_test")
REQUIRED_FILES = (("traj", ".txt"), ("intrinsics", ".npy"), ("char", ".npy"),
                  ("char_raw", ".npy"), ("caption", ".txt"),
                  ("cam_segments", ".npy"), ("caption_clip/seq", ".npy"))
VIDEO_PATTERN = re.compile(r"^(?P<video>.+)_[0-9]{5}_[0-9]{5}$")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def native_id(value):
    if not isinstance(value, str):
        raise ValueError("Sample IDs must be strings")
    value = value[3:] if value.startswith("et-") else value
    if (not value or value.strip() != value or value in (".", "..")
            or any(char in value for char in ("/", "\\", "\n", "\r", "\x00"))):
        raise ValueError(f"Unsafe or empty ET sample ID: {value!r}")
    return value


def read_split(path):
    raw = Path(path).read_bytes()
    values = raw.decode("utf-8").splitlines()
    ids = [native_id(value) for value in values]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate IDs in {path}")
    return ids, {"path": str(path), "sha256": digest(raw), "count": len(ids)}


def video_id(item_id):
    match = VIDEO_PATTERN.fullmatch(native_id(item_id))
    if match is None:
        raise ValueError(f"Cannot infer source video from {item_id!r}; expected VIDEO_00000_00000")
    # ET repeats media IDs under different year prefixes. The year describes
    # an inventory alias, not a different source video. Preserve underscores
    # inside the media ID while making these aliases one isolation group.
    return re.sub(r"^[0-9]{4}_", "", match.group("video"), count=1)


def load_annotations(path):
    payload = read_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Annotations must be a nonempty list of custom_id records")
    lookup, invalid, multientry = {}, {}, {}
    seen = set()
    for record in payload:
        if not isinstance(record, dict) or not isinstance(record.get("custom_id"), str):
            raise ValueError("Every annotation needs a string custom_id")
        if not record["custom_id"].startswith("et-"):
            raise ValueError(f"Expected et- custom_id prefix: {record['custom_id']!r}")
        item_id = native_id(record["custom_id"])
        if item_id in seen:
            raise ValueError(f"Duplicate annotated sample ID: {item_id}")
        seen.add(item_id)
        reasons, lengths = [], {}
        for name in ("cinematographyPrompts", "simulationInstructions"):
            value = record.get(name)
            if not isinstance(value, list) or not value or not isinstance(value[0], dict):
                reasons.append(f"{name} must be a nonempty list whose first item is an object")
            elif len(value) > 1:
                lengths[name] = len(value)
        if reasons:
            invalid[item_id] = reasons
            continue
        if lengths:
            multientry[item_id] = lengths
        lookup[item_id] = record
    return lookup, invalid, multientry


def excluded_ids(path):
    """Accept a preparation manifest, annotation list, or ID-keyed Gateway output."""
    payload = read_json(path)
    if isinstance(payload, dict):
        if "selected_ids" in payload:
            payload = payload["selected_ids"]
        elif "item_ids" in payload:
            payload = payload["item_ids"]
        elif payload and all(isinstance(key, str) and key.startswith("et-") for key in payload):
            payload = list(payload)
        else:
            raise ValueError(f"Excluded manifest needs selected_ids/item_ids or et- keyed annotations: {path}")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Excluded cohort must contain a nonempty ID list: {path}")
    result = []
    for row in payload:
        value = row.get("custom_id", row.get("item_id")) if isinstance(row, dict) else row
        result.append(native_id(value))
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate IDs in excluded cohort: {path}")
    return set(result)


def split_groups(ids, fraction, seed, group_by):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("val-fraction must lie strictly between zero and one")
    grouped = defaultdict(list)
    for item_id in sorted(ids):
        grouped[video_id(item_id) if group_by == "video" else item_id].append(item_id)
    if len(grouped) < 2:
        raise ValueError(f"Need at least two {group_by} groups to form train and validation; found {len(grouped)}")
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    target = fraction * len(ids)
    running = 0
    options = []
    for count, key in enumerate(keys[:-1], 1):
        running += len(grouped[key])
        options.append((abs(running - target), count))
    # A deterministic prefix of seeded groups, closest to the requested clip
    # count. At least one group remains on each side; no group is subdivided.
    prefix = min(options)[1]
    val_groups = set(keys[:prefix])
    train = sorted(item for key, values in grouped.items() if key not in val_groups for item in values)
    val = sorted(item for key, values in grouped.items() if key in val_groups for item in values)
    return train, val


def check_write_once(path, content):
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != content)):
        raise ValueError(f"Refusing to replace different existing output: {path}")


def write_once(path, content):
    check_write_once(path, content)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.chmod(0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(dataset_root, annotations, output, *, excluded_cohorts=(), seed=42,
            val_fraction=.1, group_by="video"):
    dataset_root = Path(dataset_root).expanduser().resolve(strict=True)
    annotations = Path(annotations).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError("dataset-root must be a directory")
    if output == dataset_root or dataset_root in output.parents:
        raise ValueError("Output must be outside the original ET dataset")
    if group_by not in ("video", "clip"):
        raise ValueError("group-by must be video or clip")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    annotation_lookup, invalid_annotations, multientry_annotations = load_annotations(annotations)
    pure_train, train_info = read_split(dataset_root / "pure_train_split.txt")
    official_exclusions = set()
    split_info = {"pure_train": train_info}
    exclusions_by_source = {}
    for name in TEST_SPLITS:
        values, info = read_split(dataset_root / f"{name}_split.txt")
        split_info[name] = info
        exclusions_by_source[name] = set(values)
        official_exclusions.update(values)
    excluded_inputs = []
    explicit_exclusions = set()
    for path in sorted({Path(value).expanduser().resolve(strict=True) for value in excluded_cohorts}):
        ids = excluded_ids(path)
        explicit_exclusions.update(ids)
        excluded_inputs.append({"path": str(path), "sha256": digest(path.read_bytes()), "count": len(ids)})
    held_out = official_exclusions | explicit_exclusions
    intersection = set(pure_train) & set(annotation_lookup)
    eligible = intersection - held_out
    same_video = set()
    held_out_videos = set()
    if group_by == "video":
        held_out_videos = {video_id(item_id) for item_id in held_out}
        same_video = {item_id for item_id in eligible if video_id(item_id) in held_out_videos}
        eligible -= same_video
    counts = {"annotated": len(annotation_lookup) + len(invalid_annotations),
              "structurally_valid_annotations": len(annotation_lookup),
              "invalid_annotations": len(invalid_annotations),
              "invalid_annotations_in_pure_train": len(set(invalid_annotations) & set(pure_train)),
              "multientry_annotations": len(multientry_annotations),
              "multientry_selected_annotations": len(set(multientry_annotations) & eligible),
              "official_pure_train": len(pure_train),
              "annotated_pure_train": len(intersection), "excluded_official_test_ids": len(intersection & official_exclusions),
              "excluded_explicit_ids": len(intersection & explicit_exclusions),
              "excluded_held_out_video_siblings": len(same_video), "eligible": len(eligible),
              "eligible_groups": len({video_id(item_id) for item_id in eligible}) if group_by == "video" else len(eligible)}
    if counts["eligible_groups"] < 2:
        raise ValueError(f"Insufficient eligible ET adaptation samples: {counts}")
    train_ids, val_ids = split_groups(eligible, val_fraction, seed, group_by)
    if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) != eligible or eligible & held_out:
        raise AssertionError("Adaptation inventory collision")
    train_videos = {video_id(item_id) for item_id in train_ids} if group_by == "video" else set()
    val_videos = {video_id(item_id) for item_id in val_ids} if group_by == "video" else set()
    if group_by == "video" and (train_videos & val_videos or (train_videos | val_videos) & held_out_videos):
        raise AssertionError("Adaptation source-video collision")

    caption_hashes = {}
    missing = []
    for item_id in sorted(eligible):
        for directory, suffix in REQUIRED_FILES:
            path = dataset_root / directory / f"{item_id}{suffix}"
            if not path.is_file():
                missing.append(str(path))
        path = dataset_root / "caption" / f"{item_id}.txt"
        if path.is_file():
            raw = path.read_bytes()
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Empty native caption: {path}")
            caption_hashes[item_id] = {"file_sha256": digest(raw), "normalized_text_sha256": digest(text.strip().encode("utf-8"))}
    if missing:
        raise ValueError(f"Missing {len(missing)} required modality files; no samples dropped: {missing[:10]}")

    view = output / "et-data"
    if view.is_symlink() or (view.exists() and not view.is_dir()):
        raise ValueError(f"Dataset view must be an ordinary directory: {view}")
    directories = sorted(path for path in dataset_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    for source in directories:
        destination = view / source.name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"Existing dataset link points elsewhere: {destination}")
        elif destination.exists():
            raise ValueError(f"Refusing to replace existing dataset directory: {destination}")

    train_annotations = [annotation_lookup[item_id] for item_id in train_ids]
    val_annotations = [annotation_lookup[item_id] for item_id in val_ids]
    outputs = {
        view / "pure_train_split.txt": "\n".join(train_ids).encode("utf-8"),
        view / "pure_val_split.txt": "\n".join(val_ids).encode("utf-8"),
        output / "train_ids.json": json_bytes(train_ids), output / "val_ids.json": json_bytes(val_ids),
        output / "train_annotations.json": json_bytes(train_annotations),
        output / "val_annotations.json": json_bytes(val_annotations),
        output / "annotations.json": json_bytes([annotation_lookup[item_id] for item_id in sorted(eligible)]),
    }
    train_captions = {caption_hashes[item_id]["normalized_text_sha256"] for item_id in train_ids}
    val_captions = {caption_hashes[item_id]["normalized_text_sha256"] for item_id in val_ids}
    manifest = {
        "version": 2, "dataset_root": str(dataset_root), "dataset_view": str(view),
        "selection_rule": "pure_train intersect annotated IDs minus all official test IDs and explicit exclusions",
        "group_by": group_by,
        "source_video_rule": "Remove final two _00000 clip/segment suffixes and leading YYYY_ inventory alias; preserve embedded media-ID underscores",
        "held_out_video_siblings_excluded": group_by == "video",
        "seed": seed, "requested_validation_fraction": val_fraction,
        "actual_validation_fraction": len(val_ids) / len(eligible),
        "split_algorithm": "Sorted groups, Python Random(seed) shuffle, closest nonempty validation prefix retaining at least one training group",
        "counts": {**counts, "train": len(train_ids), "validation": len(val_ids)},
        "group_counts": {"train": len(train_videos) if group_by == "video" else len(train_ids),
                         "validation": len(val_videos) if group_by == "video" else len(val_ids)},
        "train_ids": train_ids, "validation_ids": val_ids,
        "held_out_ids": sorted(held_out),
        "held_out_video_ids": sorted(held_out_videos) if group_by == "video" else None,
        "excluded_test_overlap_ids": sorted(intersection & held_out),
        "excluded_held_out_video_sibling_ids": sorted(same_video),
        "collision_checks": {"train_validation_ids": 0, "adaptation_official_test_ids": 0,
                             "adaptation_explicit_excluded_ids": 0,
                             "train_validation_videos": 0 if group_by == "video" else None,
                             "adaptation_held_out_videos": 0 if group_by == "video" else None,
                             "shared_native_caption_texts_train_validation": len(train_captions & val_captions)},
        "official_test_overlap_before_exclusion": {name: len(intersection & ids) for name, ids in exclusions_by_source.items()},
        "source_splits": split_info,
        "source_annotations": {"path": str(annotations), "sha256": digest(annotations.read_bytes()),
                               "translation_content_modified": False},
        "annotation_audit": {
            "entry_selection": "ETDataset consumes index 0 of each nonempty prompt/instruction list; extra entries are retained unchanged",
            "invalid_records_excluded_before_split": invalid_annotations,
            "invalid_pure_train_ids": sorted(set(invalid_annotations) & set(pure_train)),
            "multientry_records": multientry_annotations,
            "multientry_selected_ids": sorted(set(multientry_annotations) & eligible),
        },
        "explicit_excluded_cohorts": excluded_inputs, "caption_hashes": caption_hashes,
        "output_sha256": {str(path.relative_to(output)): digest(content) for path, content in outputs.items()},
        "script_sha256": digest(Path(__file__).read_bytes()),
        "source_access": "Read-only preparation; native modality directories are linked, never copied or modified.",
        "loader_contract": "Use ETDataset(set_name='pure', split='train') and a separate ETDataset(split='val'); consume the explicit inventories without re-splitting.",
        "annotation_scope": "Existing training translations retained verbatim; structural shape checked, semantic correctness and schema freshness not established.",
    }
    outputs[output / "manifest.json"] = json_bytes(manifest)
    # Validate every destination before creating links or writing the first file.
    for path, content in outputs.items():
        check_write_once(path, content)
    view.mkdir(parents=True, exist_ok=True)
    for source in directories:
        destination = view / source.name
        if not destination.is_symlink():
            destination.symlink_to(source, target_is_directory=True)
    for path, content in outputs.items():
        write_once(path, content)
    return manifest


def self_test():
    import unittest

    class PreparationTests(unittest.TestCase):
        def setUp(self):
            self.temporary = tempfile.TemporaryDirectory()
            self.addCleanup(self.temporary.cleanup)
            self.base = Path(self.temporary.name)
            self.source = self.base / "source"
            self.source.mkdir()
            self.ids = [f"2011_video_{i}_00000_00000" for i in range(10)]
            self.ids += ["2011_video_0_00001_00000", "2011_video_9_00001_00000"]
            for name, values in {"pure_train": self.ids, **{name: [self.ids[9]] for name in TEST_SPLITS}}.items():
                (self.source / f"{name}_split.txt").write_text("\n".join(values))
            records = [{"custom_id": "et-" + item_id, "cinematographyPrompts": [{}], "simulationInstructions": [{}]} for item_id in self.ids]
            self.annotations = self.base / "annotations.json"
            self.annotations.write_bytes(json_bytes(records))
            for item_id in self.ids:
                for directory, suffix in REQUIRED_FILES:
                    path = self.source / directory / f"{item_id}{suffix}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("native fixture")

        def test_video_grouping_exclusion_and_idempotence(self):
            output = self.base / "output"
            result = prepare(self.source, self.annotations, output, seed=7, val_fraction=.25)
            self.assertEqual(result["counts"]["eligible"], 10)
            self.assertEqual(result["counts"]["excluded_held_out_video_siblings"], 1)
            self.assertNotIn(self.ids[9], result["train_ids"] + result["validation_ids"])
            groups = lambda values: {video_id(value) for value in values}
            self.assertFalse(groups(result["train_ids"]) & groups(result["validation_ids"]))
            self.assertTrue((output / "et-data/traj").is_symlink())
            self.assertFalse((output / "et-data/pure_train_split.txt").read_bytes().endswith(b"\n"))
            self.assertEqual(result, prepare(self.source, self.annotations, output, seed=7, val_fraction=.25))
            with self.assertRaisesRegex(ValueError, "replace different existing output"):
                prepare(self.source, self.annotations, output, seed=8, val_fraction=.25)

        def test_explicit_reviewed_cohort_excludes_sibling_video(self):
            cohort = self.base / "review.json"
            cohort.write_bytes(json_bytes({"selected_ids": [self.ids[0]]}))
            result = prepare(self.source, self.annotations, self.base / "out", excluded_cohorts=[cohort])
            selected = result["train_ids"] + result["validation_ids"]
            self.assertNotIn(self.ids[0], selected)
            self.assertNotIn("2011_video_0_00001_00000", selected)
            self.assertEqual(result["counts"]["eligible"], 8)

        def test_duplicate_annotations_and_missing_splits_rejected(self):
            records = read_json(self.annotations)
            self.annotations.write_bytes(json_bytes(records + records[:1]))
            with self.assertRaisesRegex(ValueError, "Duplicate annotated"):
                prepare(self.source, self.annotations, self.base / "out")
            self.annotations.write_bytes(json_bytes(records))
            (self.source / "clatr_test_split.txt").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(self.source, self.annotations, self.base / "out")

        def test_missing_modality_fails_without_dropping_sample(self):
            (self.source / "traj" / f"{self.ids[0]}.txt").unlink()
            with self.assertRaisesRegex(ValueError, "no samples dropped"):
                prepare(self.source, self.annotations, self.base / "out")
            self.assertFalse((self.base / "out").exists())

        def test_video_parser_preserves_embedded_underscores(self):
            self.assertEqual(video_id("2011_-ftyIj2_b8Y_00041_00000"), "-ftyIj2_b8Y")
            self.assertEqual(video_id("et-2018_-ftyIj2_b8Y_00042_00001"), "-ftyIj2_b8Y")
            with self.assertRaises(ValueError):
                video_id("unrecognized")
            for value in ("../escape", "et-", "has\nnewline"):
                with self.assertRaises(ValueError):
                    native_id(value)

        def test_official_test_year_alias_excludes_source_video(self):
            # No exact IDs match this official test clip, but both year aliases
            # of its source media must still be excluded from train and val.
            for name in TEST_SPLITS:
                (self.source / f"{name}_split.txt").write_text("2018_video_9_99999_00000")
            result = prepare(self.source, self.annotations, self.base / "out")
            selected = result["train_ids"] + result["validation_ids"]
            self.assertNotIn(self.ids[9], selected)
            self.assertNotIn("2011_video_9_00001_00000", selected)
            self.assertEqual(result["counts"]["excluded_official_test_ids"], 0)
            self.assertEqual(result["counts"]["excluded_held_out_video_siblings"], 2)

        def test_explicit_diagnostic_year_alias_excludes_source_video(self):
            cohort = self.base / "diagnostic.json"
            cohort.write_bytes(json_bytes({"selected_ids": ["2018_video_0_99999_00000"]}))
            result = prepare(self.source, self.annotations, self.base / "out", excluded_cohorts=[cohort])
            selected = result["train_ids"] + result["validation_ids"]
            self.assertFalse(any(video_id(item) == "video_0" for item in selected))
            self.assertEqual(result["counts"]["excluded_explicit_ids"], 0)

        def test_year_aliases_cannot_cross_train_validation(self):
            values = ["2011_-ftyIj2_b8Y_00041_00000", "2018_-ftyIj2_b8Y_00042_00001",
                      "2011_other-video_00000_00000", "2011_third-video_00000_00000"]
            for seed in range(10):
                train, val = split_groups(values, .5, seed, "video")
                self.assertFalse({video_id(item) for item in train} & {video_id(item) for item in val})

        def test_training_verifier_rejects_legacy_year_alias_leakage(self):
            from train_et_adaptation import verify_prepared_data
            output = self.base / "out"
            manifest = prepare(self.source, self.annotations, output)
            verify_prepared_data(output)
            selected = manifest["train_ids"][0]
            manifest["held_out_ids"].append("2018_" + selected.split("_", 1)[1])
            (output / "manifest.json").chmod(0o644)
            (output / "manifest.json").write_bytes(json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "canonical source-video"):
                verify_prepared_data(output)

        def test_training_verifier_rechecks_explicit_diagnostic_provenance(self):
            from train_et_adaptation import verify_prepared_data
            cohort = self.base / "diagnostic.json"
            cohort.write_bytes(json_bytes({"selected_ids": [self.ids[0]]}))
            output = self.base / "out"
            prepare(self.source, self.annotations, output, excluded_cohorts=[cohort])
            verify_prepared_data(output)
            cohort.write_bytes(json_bytes({"selected_ids": [self.ids[1]]}))
            with self.assertRaisesRegex(ValueError, "exclusion cohort provenance"):
                verify_prepared_data(output)

        def test_training_verifier_rejects_clip_only_isolation(self):
            from train_et_adaptation import verify_prepared_data
            output = self.base / "out"
            prepare(self.source, self.annotations, output, group_by="clip")
            with self.assertRaisesRegex(ValueError, "group-by video"):
                verify_prepared_data(output)

        def test_clip_mode_explicitly_reports_weaker_video_isolation(self):
            result = prepare(self.source, self.annotations, self.base / "out", group_by="clip")
            self.assertEqual(result["counts"]["eligible"], 11)
            self.assertEqual(result["held_out_ids"], [self.ids[9]])
            self.assertIsNone(result["collision_checks"]["adaptation_held_out_videos"])
            self.assertFalse(result["held_out_video_siblings_excluded"])
            self.assertIn("2011_video_9_00001_00000", result["train_ids"] + result["validation_ids"])

        def test_multientry_retained_and_invalid_records_explicitly_excluded(self):
            records = read_json(self.annotations)
            records[0]["cinematographyPrompts"] = [{"movement": {"type": "static"}}, {"alternate": True}]
            records[1]["simulationInstructions"] = []
            records.append({"custom_id": "et-2011_outside_00000_00000", "cinematographyPrompts": None,
                            "simulationInstructions": [None]})
            self.annotations.write_bytes(json_bytes(records))
            output = self.base / "out"
            result = prepare(self.source, self.annotations, output)
            self.assertEqual(result["counts"]["multientry_annotations"], 1)
            self.assertEqual(result["counts"]["invalid_annotations"], 2)
            self.assertEqual(result["counts"]["invalid_annotations_in_pure_train"], 1)
            self.assertEqual(result["annotation_audit"]["invalid_pure_train_ids"], [self.ids[1]])
            selected = read_json(output / "annotations.json")
            self.assertEqual(next(row for row in selected if row["custom_id"] == records[0]["custom_id"]), records[0])
            self.assertNotIn(self.ids[1], result["train_ids"] + result["validation_ids"])
            self.annotations.write_bytes(json_bytes(records + records[1:2]))
            with self.assertRaisesRegex(ValueError, "Duplicate annotated"):
                prepare(self.source, self.annotations, self.base / "other")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "annotations", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--excluded-cohort", action="append", type=Path, default=[], help="Repeatable preparation manifest or annotation JSON to exclude.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--group-by", choices=("video", "clip"), default="video", help="video excludes held-out video siblings and groups train/val; clip enforces ID isolation only.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    for name in ("dataset_root", "annotations", "output"):
        if getattr(args, name) is None:
            parser.error("--" + name.replace("_", "-") + " is required")
    result = prepare(args.dataset_root, args.annotations, args.output, excluded_cohorts=args.excluded_cohort,
                     seed=args.seed, val_fraction=args.val_fraction, group_by=args.group_by)
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"],
                      "group_counts": result["group_counts"], "collision_checks": result["collision_checks"]}, indent=2))


if __name__ == "__main__":
    main()
