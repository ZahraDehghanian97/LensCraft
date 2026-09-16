#!/usr/bin/env python3
"""Extract diverse, actual paper candidates from a completed evaluation on CPU.

Run this script with --project-dir pointing to the frozen experiment source.
No model is instantiated and no inference is performed. Rendering is a separate
step so the frozen data readers can coexist with the latest figure renderer.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys


HELPER_SHA256 = "c3d0e77a3f42a8393914a1f9327cc2af3246f739812c3bfe2fa9dce392e06ec3"
SOURCE_FILES = (
    "src/visualization/data.py", "src/models/camera_trajectory_model.py",
    "src/utils/load_lens_craft.py", "src/data/simulation/dataset.py",
    "src/data/simulation/convertor.py", "src/data/simulation/loader.py",
    "src/data/simulation/metadata.py", "src/data/simulation/caption.py",
    "src/data/simulation/utils.py", "src/data/convertor/utils.py",
    "src/testing/process.py", "src/testing/keyframes.py",
    "config/inference.yaml", "config/training/model/default.yaml",
)
MODELS = ("ccdm", "et", "gendop", "lenscraft")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def frozen_helpers(project_dir):
    candidates = (project_dir / "scripts/export_paper_qualitative.py",
                  Path(__file__).with_name("export_paper_qualitative.py"))
    path = next((path for path in candidates if path.is_file()), None)
    if path is None or digest(path) != HELPER_SHA256:
        raise ValueError("The original, fingerprinted export_paper_qualitative.py helper is required")
    spec = importlib.util.spec_from_file_location("frozen_paper_qualitative", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def verify_frozen_sources(project_dir, manifest):
    result = {}
    for name in SOURCE_FILES:
        expected = manifest.get("source_sha256", {}).get(name)
        if expected is None or digest(project_dir / name) != expected:
            raise ValueError(f"Candidate source differs from frozen evaluation: {name}")
        result[name] = expected
    return result


def camera_motion(sample):
    raw = sample.metadata.get("raw_prompt", {})
    motion = raw.get("movement", {}).get("type") if isinstance(raw, dict) else None
    if not isinstance(motion, str) or not motion:
        raise ValueError(f"Missing actual camera movement annotation for {sample.sample_id}")
    return motion


def diverse_selection(candidates, count, seed):
    """Round-robin annotated camera movements; never inspect model outputs."""
    if count < 1 or count > len(candidates):
        raise ValueError("Requested count must fit the candidate pool")
    if len({row["cohort_offset"] for row in candidates}) != len(candidates):
        raise ValueError("Repeated candidate cohort offsets")
    groups = {}
    for row in candidates:
        groups.setdefault(row["camera_motion"], []).append(row)
    motions = sorted(groups)
    random.Random(seed).shuffle(motions)
    selected = []
    while len(selected) < count:
        for motion in motions:
            if groups[motion]:
                selected.append(groups[motion].pop(0))
                if len(selected) == count:
                    break
    return selected


def keyframe_indices(source_mask, padding_mask, count):
    if not source_mask or len(source_mask) != len(padding_mask):
        raise ValueError("Source and padding masks must have matching nonempty timelines")
    if any(type(value) is not bool for value in source_mask + padding_mask):
        raise ValueError("Cached masks must be boolean")
    if any(padding and not source for source, padding in zip(source_mask, padding_mask)):
        raise ValueError("Cached keyframes include padding")
    valid = [index for index, padding in enumerate(padding_mask) if not padding]
    known = [index for index, hidden in enumerate(source_mask) if not hidden]
    if len(known) != min(count, len(valid)) or not valid:
        raise ValueError("Cached keyframe count differs from recorded K")
    remapping = {old: new for new, old in enumerate(valid)}
    return valid, known, [remapping[index] for index in known]


def cached_keyframes(path, offsets, expected_count, count):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("config_hash") != path.stem or not payload.get("batches"):
        raise ValueError("Invalid cached keyframe payload fingerprint or empty batches")
    modes = (f"key_framing_k{count}", f"key_framing+prompt_k{count}")
    selected = {mode: {} for mode in modes}
    start = 0
    for batch in payload["batches"]:
        batch_length = None
        for mode in modes:
            item = batch.get("items", {}).get(mode)
            if not isinstance(item, dict):
                raise ValueError(f"Completed cache has no {mode}; select a measured K")
            trajectory = item["trajectory"]
            source, padding = item["source_mask"], item["padding_mask"]
            if trajectory.ndim != 3 or trajectory.shape[2] != 6 or not torch.isfinite(trajectory).all():
                raise ValueError("Invalid cached keyframe trajectory")
            if (source.dtype != torch.bool or padding.dtype != torch.bool or
                    source.shape != trajectory.shape[:2] or padding.shape != source.shape):
                raise ValueError("Cached keyframe mask shape/type mismatch")
            if batch_length is not None and batch_length != len(trajectory):
                raise ValueError("Cached modes have different batch lengths")
            batch_length = len(trajectory)
            for offset in offsets:
                if start <= offset < start + batch_length:
                    local = offset - start
                    valid, known, visible = keyframe_indices(source[local].tolist(), padding[local].tolist(), count)
                    selected[mode][offset] = {"trajectory": trajectory[local][valid].clone(),
                        "valid_indices": valid, "model_indices": known, "visible_indices": visible,
                        "sequence_length": trajectory.shape[1], "source_mask": source[local].tolist(),
                        "padding_mask": padding[local].tolist()}
        start += batch_length
    if start != expected_count or any(set(rows) != set(offsets) for rows in selected.values()):
        raise ValueError("Cached keyframe cohort count or selected offsets differ")
    return selected


def supplied_camera_poses(sample, frame_count):
    """Recreate the exact camera input preprocessing, without text embeddings."""
    from data.simulation.dataset import SimulationDataset
    from data.simulation.loader import (parse_simulation_file_to_dict, extract_camera_trajectory,
        extract_subject_components, resample_paired_euler_trajectories)
    from visualization.data import _sim_standard

    _, handle, native_index = sample._context
    native = handle["native"]
    raw = parse_simulation_file_to_dict(native["files"][native_index], native["dictionary"], native["scale"])
    camera = extract_camera_trajectory(raw["cameraFrames"])
    subject, _ = extract_subject_components(raw["subjectsInfo"])
    camera, _ = resample_paired_euler_trajectories(camera, subject, frame_count)
    normalized = SimulationDataset.normalize_item(camera, None, None, True)[0]
    return _sim_standard(normalized, normalized=True)[0]


def run(args):
    helper, helper_path = frozen_helpers(args.project_dir)
    manifest_path = args.base_run_dir / "evaluation/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_hashes = verify_frozen_sources(args.project_dir, manifest)
    for key in ("lenscraft_checkpoint", "lenscraft_config"):
        entry = manifest["inputs"][key]
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen model input changed: {key}")
    if (args.output_dir / "candidate_manifest.json").exists() and not args.overwrite:
        raise ValueError("Candidate manifest already exists; use a new directory or --overwrite")
    sys.path.insert(0, str(args.project_dir / "src"))
    os.chdir(args.project_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from dotenv import load_dotenv
    load_dotenv(args.project_dir / ".env", override=False)
    os.environ.update(SIMULATION_DATA_PATH=manifest["dataset"]["path"],
        TEST_CHECKPOINT_PATH=manifest["inputs"]["lenscraft_checkpoint"]["path"],
        TEST_CONFIG_PATH=manifest["inputs"]["lenscraft_config"]["path"])
    import torch
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    from data.simulation.loader import load_movement_types
    from visualization.data import VisualizationRepository, _sim_standard, save_result

    dataset_path = Path(manifest["dataset"]["path"])
    if digest(dataset_path / "parameter_dictionary.msgpack") != manifest["dataset"]["parameter_dictionary_sha256"]:
        raise ValueError("Dataset parameter dictionary changed")
    split_seed = manifest["dataset"].get("split_seed", 42)
    repository = VisualizationRepository(config_dir=args.project_dir / "config", device="cpu", seed=split_seed,
        overrides=["data.val_size=0.2", "data.test_size=0.2", "+data.dataset.config.allowed_movement_types=[]",
            "training.model.inference.checkpoint_path=" + json.dumps(os.environ["TEST_CHECKPOINT_PATH"]),
            "training.model.inference.config=" + json.dumps(os.environ["TEST_CONFIG_PATH"])])
    handle = repository._dataset("simulation", "test", dataset_path)
    if len(handle["ids"]) != manifest["dataset"]["sample_count"]:
        raise ValueError("Dataset inventory count differs from frozen evaluation")
    holdout = helper.select_measured_holdout(handle["ids"], handle["indices"], manifest["test_fraction"],
        manifest["dataset"]["test_indices_sha256"])
    measurements = {cohort: json.loads((args.base_run_dir / "evaluation/results" /
        f"metrics_lens_craft_{cohort}.json").read_text()) for cohort in ("static", "dynamic")}
    cohorts = helper.partition_cohorts(holdout, load_movement_types(dataset_path), measurements)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"schema": "lenscraft.paper-candidates.v1", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "generator_sha256": digest(__file__),
        "helper_path": str(helper_path), "helper_sha256": HELPER_SHA256,
        "base_manifest_sha256": digest(manifest_path), "frozen_source_sha256": source_hashes,
        "checkpoint": manifest["inputs"]["lenscraft_checkpoint"], "seed": args.seed,
        "split_seed": split_seed, "test_fraction": manifest["test_fraction"],
        "test_indices_sha256": manifest["dataset"]["test_indices_sha256"],
        "movement_types_sha256": digest(dataset_path / "movement_types.txt"),
        "per_cohort": args.per_cohort, "keyframe_count": None if args.prompt_only else args.keyframe_count,
        "selection": "Seeded candidate permutation, round-robin actual annotated camera movements; no prediction-quality filtering",
        "generation": "Existing completed-evaluation caches only; no inference or model loading",
        "candidates": [], "selection_pools": {}, "cache_sources": []}
    for cohort_number, cohort in enumerate(("static", "dynamic")):
        if args.per_cohort > len(cohorts[cohort]):
            raise ValueError(f"Requested count exceeds measured {cohort} cohort")
        permutation = list(range(len(cohorts[cohort])))
        random.Random(args.seed + cohort_number).shuffle(permutation)
        pool_offsets = permutation[:max(args.per_cohort, min(200, len(permutation))) ]
        pool, loaded = [], {}
        for offset in pool_offsets:
            item = cohorts[cohort][offset]
            sample = repository.load_sample("simulation", "test", sample_id=item["sample_id"], data_path=dataset_path)
            loaded[offset] = sample
            pool.append({**item, "cohort_offset": offset, "camera_motion": camera_motion(sample)})
        selection = diverse_selection(pool, args.per_cohort, args.seed + cohort_number)
        provenance["selection_pools"][cohort] = pool
        samples = {item["cohort_offset"]: loaded[item["cohort_offset"]] for item in selection}
        print(f"Selected {len(samples)} {cohort} scenes across {len({row['camera_motion'] for row in selection})} camera motions", flush=True)
        lenscraft_path, lenscraft_record = None, None
        for model in MODELS:
            path, log = helper.cache_path(args.base_run_dir, model, cohort)
            cached, record = helper.cached_samples(path, model, list(samples), len(cohorts[cohort]))
            provenance["cache_sources"].append({**record, "model": model, "cohort": cohort,
                "log": str(log), "log_sha256": digest(log)})
            name = ("lens_craft" if model == "lenscraft" else model) + ":prompt_generation"
            for offset, sample in samples.items():
                sample.trajectories[name] = _sim_standard(cached[offset], normalized=True)[0]
                sample.metadata.setdefault("runs", {})[name] = {"mode": "prompt_generation", "keyframes": [],
                    "source": "completed-evaluation normalized trajectory cache", "cache_path": str(path),
                    "cache_sha256": record["sha256"], "cohort_offset": offset, "lenscraft_initialization": False}
            if model == "lenscraft":
                lenscraft_path, lenscraft_record = path, record
            print(f"Read {cohort} {model} cached predictions", flush=True)
        keyframes = None if args.prompt_only else cached_keyframes(lenscraft_path, list(samples), len(cohorts[cohort]), args.keyframe_count)
        for rank, selected in enumerate(selection, 1):
            sample = samples[selected["cohort_offset"]]
            sample.metadata.update(cohort=cohort, subject_mode=cohort, camera_motion=selected["camera_motion"],
                cohort_offset=selected["cohort_offset"], selection=provenance["selection"], selection_seed=args.seed,
                paper_task="prompt", cached_actual_predictions=True, checkpoint_sha256=provenance["checkpoint"]["sha256"])
            stem = f"{cohort}-{rank:02d}-{sample.sample_id.rsplit('_', 1)[-1]}"

            def save_candidate(item, task, suffix):
                path = save_result(item, output / "bundles" / f"{stem}-{suffix}.json")
                provenance["candidates"].append({"sample_id": item.sample_id, "cohort": cohort,
                    "camera_motion": selected["camera_motion"], "paper_task": task, "bundle": str(path.relative_to(output)),
                    "bundle_sha256": digest(path), "methods": list(item.trajectories), "cohort_offset": selected["cohort_offset"]})

            save_candidate(sample, "prompt", "prompt")
            if keyframes is not None:
                item = replace(sample, trajectories={name: sample.trajectories[name] for name in ("GT", "lens_craft:prompt_generation")},
                    metadata=copy.deepcopy(sample.metadata), _context=None)
                item.metadata["paper_task"] = "keyframe"
                item.metadata["keyframe_count"] = args.keyframe_count
                common = None
                for mode, cached in keyframes.items():
                    record = cached[selected["cohort_offset"]]
                    reference = supplied_camera_poses(sample, record["sequence_length"])
                    if common is not None and common != record["model_indices"]:
                        raise ValueError("Keyframe-only and keyframe+prompt caches used different supplied poses")
                    common = record["model_indices"]
                    item.trajectories["GT"] = reference[record["valid_indices"]]
                    item.keyframes = record["visible_indices"]
                    method = "lens_craft:" + mode
                    item.trajectories[method] = _sim_standard(record["trajectory"], normalized=True)[0]
                    item.metadata.setdefault("runs", {})[method] = {"mode": mode.split("_k")[0], "keyframes": item.keyframes,
                        "keyframe_model_indices": record["model_indices"], "keyframe_poses": reference[record["model_indices"]].tolist(),
                        "keyframe_sequence_length": record["sequence_length"], "keyframe_count": len(item.keyframes),
                        "source_mask": record["source_mask"], "padding_mask": record["padding_mask"],
                        "source": "completed-evaluation normalized trajectory cache", "cache_path": str(lenscraft_path),
                        "cache_sha256": lenscraft_record["sha256"], "cohort_offset": selected["cohort_offset"]}
                item.metadata["keyframe_source"] = "GT"
                save_candidate(item, "keyframe", f"keyframe-k{args.keyframe_count}")
        helper.write_json(output / "candidate_manifest.json", provenance)
    provenance.update(status="complete", distinct_scene_count=len({item["sample_id"] for item in provenance["candidates"]}),
        bundle_count=len(provenance["candidates"]))
    helper.write_json(output / "candidate_manifest.json", provenance)
    return provenance


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-cohort", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keyframe-count", type=int, default=4)
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.per_cohort < 1 or args.keyframe_count < 1 or not 0 <= args.seed < 2 ** 32:
        parser.error("Counts must be positive and seed must be in [0, 2**32)")
    for name in ("project_dir", "base_run_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in ("status", "distinct_scene_count", "bundle_count")}, indent=2))
