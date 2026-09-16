#!/usr/bin/env python3
"""Export paper Figures 4-6 from cached predictions and the fixed trained model.

Run inside the frozen base-run source environment. Only this new file is added;
existing visualization and inference code is reused without modification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys

MOTION_WORDS = re.compile(r"\b(doll(?:y|ies|ying)|zoom(?:s|ing)?|truck(?:s|ing)?|pedestal|arc(?:s|ing)?|orbit(?:s|ing)?|crane|sweep(?:s|ing)?|pan(?:s|ning)?|tilt(?:s|ing)?)\b", re.I)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_measured_holdout(identifiers, indices, fraction, expected_hash):
    if hashlib.sha256(json.dumps(indices).encode()).hexdigest() != expected_hash:
        raise ValueError("Visualization split differs from the measured heldout split")
    if not isinstance(fraction, (float, int)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Invalid measured holdout fraction")
    if not indices or len(set(indices)) != len(indices) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(identifiers) for i in indices):
        raise ValueError("Invalid or repeated heldout sample indices")
    count = max(1, math.ceil(len(indices) * fraction))
    result = [{"sample_id": identifiers[index], "dataset_index": index} for index in indices[:count]]
    if len({row["sample_id"] for row in result}) != len(result):
        raise ValueError("Repeated measured sample identifiers")
    return result


def partition_cohorts(holdout, movement_types, measurements):
    cohorts = {}
    for cohort in ("static", "dynamic"):
        measured = measurements[cohort]
        movements = measured["test_movement_types"]
        cohorts[cohort] = [item for item in holdout if movement_types[item["sample_id"] + ".msgpack"] in movements]
        counts = measured["test_sample_counts"]
        if counts["cohort"] != len(cohorts[cohort]) or counts["fractional_holdout"] != len(holdout):
            raise ValueError(f"Visualization {cohort} cohort count differs from measured data")
        if len(cohorts[cohort]) < 2:
            raise ValueError(f"At least two measured {cohort} samples are required")
    memberships = [{item["sample_id"] for item in cohorts[cohort]} for cohort in ("static", "dynamic")]
    if memberships[0] & memberships[1] or memberships[0] | memberships[1] != {item["sample_id"] for item in holdout}:
        raise ValueError("Measured cohorts do not form a disjoint partition of selected holdout")
    return cohorts


def directional_conflict_candidate(prompt, start_position, end_position):
    if len(start_position) != 3 or len(end_position) != 3 or not all(math.isfinite(float(v)) for v in (*start_position, *end_position)):
        raise ValueError("Conflict selection requires finite 3D endpoint positions")
    displacement = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(start_position, end_position)))
    return bool(MOTION_WORDS.search(prompt)) and displacement > 1e-3


def reversed_valid_indices(padding):
    if not padding or any(type(value) is not bool for value in padding):
        raise ValueError("Padding must be a nonempty boolean sequence")
    valid = [index for index, hidden in enumerate(padding) if not hidden]
    if len(valid) < 2:
        raise ValueError("A source conflict requires at least two valid camera frames")
    return valid, list(reversed(valid))


def blinded_study_selection(holdout, seed=42):
    if len(holdout) < 50:
        raise ValueError("Paper human-study preparation requires at least 50 heldout scenes")
    rng = random.Random(seed)
    result = []
    for item in rng.sample(holdout, 50):
        methods = ["ccdm", "et_ca", "gendop", "lens_craft"]
        rng.shuffle(methods)
        result.append({**item, "blind_labels": dict(zip(("A", "B", "C", "D"), methods))})
    return result


def cache_path(base_run, model, cohort):
    log = base_run / "evaluation/logs" / f"evaluation_{model}_{cohort}.log"
    content = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(?:Saved generated trajectories to|Trajectory cache hit:)\s+([^\n\r]+\.pt)", content)
    if not matches:
        raise ValueError(f"No recorded trajectory-cache path in {log}")
    path = Path(matches[-1].strip())
    if not path.is_file():
        raise FileNotFoundError(f"Recorded trajectory cache is missing: {path}")
    return path, log


def cached_samples(path, model, selected_offsets, expected_count):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("config_hash") != path.stem:
        raise ValueError(f"Cache config hash differs from filename: {path}")
    if not isinstance(payload.get("batches"), list) or not payload["batches"]:
        raise ValueError(f"Trajectory cache contains no batches: {path}")
    if not selected_offsets or len(set(selected_offsets)) != len(selected_offsets) or any(type(i) is not int or not 0 <= i < expected_count for i in selected_offsets):
        raise ValueError("Cache selections must be unique valid cohort offsets")
    selected, offset = {}, 0
    for batch in payload["batches"]:
        if model == "lenscraft":
            item = batch["items"]["prompt_generation"]
            trajectories = item["trajectory"]
            padding = item.get("padding_mask")
        else:
            trajectories = batch["trajectories"]["prompt_generation"]
            padding = batch.get("padding_masks", {}).get("prompt_generation")
        if trajectories.ndim != 3 or not torch.isfinite(trajectories).all():
            raise ValueError(f"Invalid cached simulation trajectories: {path}")
        for index in selected_offsets:
            if offset <= index < offset + len(trajectories):
                local = index - offset
                values = trajectories[local]
                if padding is not None:
                    if len(padding[local]) != len(values):
                        raise ValueError("Cached trajectory and padding lengths differ")
                    values = values[~padding[local]]
                if len(values) == 0:
                    raise ValueError("Selected cached trajectory has no valid frames")
                selected[index] = values.clone()
        offset += len(trajectories)
    if offset != expected_count or set(selected) != set(selected_offsets):
        raise ValueError(f"Cached cohort/sample mapping inconsistent: {path}, count={offset}, expected={expected_count}")
    return selected, {"path": str(path), "sha256": digest(path), "config_hash": payload["config_hash"], "cohort_samples": offset}


def pose_errors(predicted, reference, indices=None):
    import numpy as np
    if indices is not None:
        predicted, reference = predicted[indices], reference[indices]
    if predicted.shape != reference.shape:
        raise ValueError("Pose-error trajectories must have equal dimensions")
    positions = np.linalg.norm(predicted[:, :3, 3] - reference[:, :3, 3], axis=-1)
    relative = np.swapaxes(predicted[:, :3, :3], -1, -2) @ reference[:, :3, :3]
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1., 1.)))
    return {"position_error_world_units": float(positions.mean()), "rotation_error_deg": float(angles.mean()), "frame_count": len(positions)}


def require_generation(sample, method):
    if sample.metadata.get("last_run", {}).get("errors") or method not in sample.trajectories:
        raise RuntimeError(f"Generation failed for {sample.sample_id}: {sample.metadata.get('errors')}")


def export(samples, identifier, output, methods, caption, protocol, include_input):
    from visualization.data import save_result
    from visualization.export import export_figure
    bundle = save_result(samples, output / "bundles" / f"{identifier}.json")
    written = export_figure(samples, output / "figures" / identifier, methods=methods,
                            show_keyframes=include_input, include_input=include_input,
                            camera_count=6, dpi=180, title=caption)
    return {"id": identifier, "path": str(written["svg"].relative_to(output)), "caption": caption,
            "protocol": {**protocol, "sample_ids": [sample.sample_id for sample in samples]},
            "bundle": str(bundle.relative_to(output)), "bundle_sha256": digest(bundle),
            "assets": {key: {"path": str(path.relative_to(output)), "sha256": digest(path)} for key, path in written.items()}}


def run(args):
    sys.path.insert(0, str(args.project_dir / "src"))
    os.chdir(args.project_dir)
    from dotenv import load_dotenv
    import numpy as np
    import torch
    from data.sim_format import to_simulation_format
    from data.simulation.loader import load_movement_types
    from data.simulation.utils import structured_conditioning_from_batch
    from utils.device import move_batch_to_device
    from visualization.data import VisualizationRepository, _sim_standard, _generation_seed

    load_dotenv(args.project_dir / ".env", override=False)
    manifest_path = args.base_run_dir / "evaluation/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_files = ("src/visualization/data.py", "src/visualization/export.py", "src/models/camera_trajectory_model.py",
                    "src/utils/load_lens_craft.py", "src/data/simulation/dataset.py", "src/data/simulation/convertor.py")
    for relative in source_files:
        expected = manifest.get("source_sha256", {}).get(relative)
        if expected is None or digest(args.project_dir / relative) != expected:
            raise ValueError(f"Qualitative generation source differs from the frozen base run: {relative}")
    for name in ("lenscraft_checkpoint", "lenscraft_config"):
        entry = manifest["inputs"][name]
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Fixed model input changed: {name}")
    os.environ.update({key: value for key, value in manifest.get("baseline_environment", {}).items() if value})
    os.environ.update(SIMULATION_DATA_PATH=manifest["dataset"]["path"],
                      TEST_CHECKPOINT_PATH=manifest["inputs"]["lenscraft_checkpoint"]["path"],
                      TEST_CONFIG_PATH=manifest["inputs"]["lenscraft_config"]["path"])
    dataset_path = Path(manifest["dataset"]["path"])
    if digest(dataset_path / "parameter_dictionary.msgpack") != manifest["dataset"]["parameter_dictionary_sha256"]:
        raise ValueError("Dataset parameter dictionary changed")
    overrides = ["data.val_size=0.2", "data.test_size=0.2", "+data.dataset.config.allowed_movement_types=[]",
                 "training.model.inference.checkpoint_path=" + json.dumps(os.environ["TEST_CHECKPOINT_PATH"]),
                 "training.model.inference.config=" + json.dumps(os.environ["TEST_CONFIG_PATH"])]
    repository = VisualizationRepository(config_dir=args.project_dir / "config", overrides=overrides, device=args.device, seed=42)
    handle = repository._dataset("simulation", "test", dataset_path)
    holdout_indices = handle["indices"]
    holdout = select_measured_holdout(handle["ids"], holdout_indices, manifest["test_fraction"], manifest["dataset"]["test_indices_sha256"])
    movement_types = load_movement_types(dataset_path)
    measurements = {}
    for cohort in ("static", "dynamic"):
        metric_path = args.base_run_dir / "evaluation/results" / f"metrics_lens_craft_{cohort}.json"
        measurements[cohort] = json.loads(metric_path.read_text())
    cohorts = partition_cohorts(holdout, movement_types, measurements)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "qualitative_manifest.json").exists() and not args.overwrite:
        raise ValueError("Qualitative output already exists; use --overwrite for deterministic regeneration")
    checkpoint = manifest["inputs"]["lenscraft_checkpoint"]
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "generator_sha256": digest(__file__),
                  "base_manifest_sha256": digest(manifest_path), "checkpoint": checkpoint,
                  "frozen_source_sha256": {relative: manifest["source_sha256"][relative] for relative in source_files},
                  "split_seed": 42, "test_fraction": manifest["test_fraction"], "test_indices_sha256": manifest["dataset"]["test_indices_sha256"],
                  "movement_types_sha256": digest(dataset_path / "movement_types.txt"), "figures": [], "cache_sources": []}

    # Figure 4: exact predictions from the completed evaluation, with no reruns.
    samples = []
    for cohort in ("static", "dynamic"):
        for offset in (0, 1):
            item = cohorts[cohort][offset]
            sample = repository.load_sample("simulation", "test", sample_id=item["sample_id"], data_path=dataset_path)
            sample.metadata.update(cohort=cohort, cohort_offset=offset, selection="first two samples in each measured cohort; no quality filtering", checkpoint_sha256=checkpoint["sha256"])
            samples.append(sample)
        for model in ("ccdm", "et", "gendop", "lenscraft"):
            path, log = cache_path(args.base_run_dir, model, cohort)
            cached, cache_record = cached_samples(path, model, (0, 1), len(cohorts[cohort]))
            provenance["cache_sources"].append({**cache_record, "model": model, "cohort": cohort, "log": str(log), "log_sha256": digest(log)})
            for sample in samples:
                if sample.metadata["cohort"] != cohort:
                    continue
                name = ("lens_craft" if model == "lenscraft" else model) + ":prompt_generation"
                sample.trajectories[name] = _sim_standard(cached[sample.metadata["cohort_offset"]], normalized=True)[0]
                sample.metadata.setdefault("runs", {})[name] = {"mode": "prompt_generation", "keyframes": [],
                    "source": "base-run normalized trajectory cache", "cache_path": str(path), "cache_sha256": cache_record["sha256"],
                    "cohort_offset": sample.metadata["cohort_offset"], "lenscraft_initialization": False}
    provenance["figures"].append(export(samples, "figure4", output,
        ["ccdm:prompt_generation", "et:prompt_generation", "gendop:prompt_generation", "lens_craft:prompt_generation", "GT"],
        "Figure 4. Prompt-only comparison from the completed evaluation",
        {"selection": "first two static and first two dynamic measured holdout samples", "baseline_mode": "normalized without LensCraft initialization",
         "cached_actual_predictions": True, "checkpoint_sha256": checkpoint["sha256"], "camera_convention": "OpenCV camera-to-world; world Y-up"}, False))
    write_json(output / "qualitative_manifest.json", provenance)
    print("Figure 4 exported", flush=True)

    # Figure 5: supplied reference poses are preserved as input constraints.
    keyframe_samples = []
    for sample in samples:
        source = repository.load_sample("simulation", "test", sample_id=sample.sample_id, data_path=dataset_path)
        generated = repository.generate(source, models=("lens_craft",), mode="prompt_generation", seed=42)
        require_generation(generated, "lens_craft:prompt_generation")
        generated = repository.generate(generated, models=("lens_craft",), mode="key_framing+prompt", keyframes=[0, 14, 29], seed=42)
        require_generation(generated, "lens_craft:key_framing+prompt")
        run = generated.metadata["runs"]["lens_craft:key_framing+prompt"]
        known = run["keyframe_model_indices"]
        reference = np.asarray(run["keyframe_poses"])
        generated.metadata["known_pose_errors"] = pose_errors(generated.trajectories["lens_craft:key_framing+prompt"][known], reference)
        generated.metadata["checkpoint_sha256"] = checkpoint["sha256"]
        generated.metadata["cohort"] = sample.metadata["cohort"]
        keyframe_samples.append(generated)
    provenance["figures"].append(export(keyframe_samples, "figure5", output,
        ["lens_craft:key_framing+prompt", "lens_craft:prompt_generation", "GT"],
        "Figure 5. Text and actual keyframe constraints at frames 0, 14 and 29",
        {"selection": "same four deterministic samples as Figure 4", "keyframe_indices": [0, 14, 29],
         "poses_drawn_from": "actual supplied ground-truth inputs", "checkpoint_sha256": checkpoint["sha256"],
         "error_units": "world position units / SO(3) geodesic degrees; individual errors in portable bundle"}, True))
    write_json(output / "qualitative_manifest.json", provenance)
    print("Figure 5 exported", flush=True)

    # Figure 6 is a controlled directional conflict, not an arbitrary-scene edit.
    conflicts = []
    for item in holdout:
        source = repository.load_sample("simulation", "test", sample_id=item["sample_id"], data_path=dataset_path)
        gt = source.trajectories["GT"]
        if not directional_conflict_candidate(source.prompt, gt[0, :3, 3].tolist(), gt[-1, :3, 3].tolist()):
            continue
        generated = repository.generate(source, models=("lens_craft",), mode="prompt_generation", seed=42)
        require_generation(generated, "lens_craft:prompt_generation")
        _, sample_handle, native_index = source._context
        device = torch.device(args.device)
        batch = move_batch_to_device(repository._batch(sample_handle, native_index), device)
        camera, subject, volume, padding = to_simulation_format(batch, "simulation", target_len=30)
        if padding is None:
            padding = torch.zeros(camera.shape[:2], dtype=torch.bool, device=device)
        valid_indices, source_indices = reversed_valid_indices(padding[0].cpu().tolist())
        known = torch.tensor(valid_indices, dtype=torch.long, device=device)
        reversed_camera = camera.clone()
        reversed_camera[:, known] = camera[:, source_indices]
        caption = structured_conditioning_from_batch(batch)
        model = repository._models[("lens_craft", str(device))]
        with _generation_seed(42, device), torch.inference_mode():
            reconstructed = model.generate_camera_trajectory(subject_trajectory=subject, subject_volume=volume,
                camera_trajectory=reversed_camera, src_key_mask=padding, padding_mask=padding,
                memory_teacher_forcing_ratio=.5, caption_embedding=caption)["reconstructed"]
        generated.trajectories["Source (time-reversed GT)"] = _sim_standard(reversed_camera, normalized=True)[0][0][known.cpu().numpy()]
        generated.trajectories["lens_craft:conflicting_source+prompt"] = _sim_standard(reconstructed, normalized=True)[0][0][known.cpu().numpy()]
        generated.metadata.update(conflicting_prompt=True, source_transformation="reverse valid camera frames in time; original subject and original target prompt retained",
            checkpoint_sha256=checkpoint["sha256"], scope="Controlled time-reversed-source test; not evidence for arbitrary reference-scene transfer",
            selection="first two heldout samples with camera endpoint displacement > 0.001 world units and directional-motion words; no generated-quality filtering")
        prediction = generated.trajectories["lens_craft:conflicting_source+prompt"]
        generated.metadata["conflict_errors"] = {"output_to_target": pose_errors(prediction, generated.trajectories["GT"]),
            "source_to_target": pose_errors(generated.trajectories["Source (time-reversed GT)"], generated.trajectories["GT"])}
        generated.metadata.setdefault("runs", {})["lens_craft:conflicting_source+prompt"] = {"mode": "source_trajectory+conflicting_prompt", "keyframes": [], "memory_teacher_forcing_ratio": .5}
        conflicts.append(generated)
        if len(conflicts) == 2:
            break
    if len(conflicts) != 2:
        raise ValueError("Fewer than two suitable measured-holdout directional-motion samples for Figure 6")
    provenance["figures"].append(export(conflicts, "figure6", output,
        ["lens_craft:conflicting_source+prompt", "Source (time-reversed GT)", "GT", "lens_craft:prompt_generation"],
        "Figure 6. Controlled time-reversed source with the original directional prompt",
        {"conflicting_prompt": True, "source_transformation": "time reversal of reference camera trajectory; same subject and target prompt",
         "checkpoint_sha256": checkpoint["sha256"], "selection": conflicts[0].metadata["selection"],
         "scope": "controlled opposite-direction source conflict; no claim of arbitrary reference transfer"}, False))
    print("Figure 6 exported", flush=True)

    study = blinded_study_selection(holdout)
    write_json(output / "human_study_preparation_private.json", {"status": "prepared only; no human ratings collected", "seed": 42,
        "sampling": "50 samples without replacement from the measured 10% holdout; deterministic per-scene method order",
        "keep_private": "Method-key mapping must not be exposed to evaluators", "samples": study,
        "remaining": "Render method outputs and present blinded stimuli to 30 independent evaluators; collect scores and ranks"})
    provenance["human_study_preparation"] = "human_study_preparation_private.json"
    provenance["status"] = "complete"
    write_json(output / "qualitative_manifest.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("project_dir", "base_run_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    result = run(args)
    print(json.dumps({"output_dir": str(args.output_dir), "status": result["status"], "figures": [f["id"] for f in result["figures"]]}, indent=2))


if __name__ == "__main__":
    main()
