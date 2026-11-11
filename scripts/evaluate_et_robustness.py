#!/usr/bin/env python3
"""Evaluate one checkpoint on the frozen ET cohort and an optional SIM holdout.

Run twice with different --checkpoint/--train-config values and the same data
configs. --output is an artifact directory. No pose alignment is applied during
generation. First-frame-relative orientation is a reference-assisted diagnostic,
not an unconditioned generation score. Existing ET configs may reference
ET_DATA_DIR/ET_CIN_LANG_PATH; the usual prepared cohort beside evaluation/hydra
is detected, otherwise supply those variables in the environment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def angle_deg(rotation):
    import numpy as np
    sine = np.linalg.norm(np.stack((
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ), axis=-1), axis=-1) / 2
    cosine = (np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2
    return np.rad2deg(np.arctan2(sine, np.clip(cosine, -1, 1)))


def xyz_matrix(angles):
    """Intrinsic XYZ camera-to-world Euler angles, matching simulation poses."""
    import numpy as np
    x, y, z = np.moveaxis(angles, -1, 0)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    return np.stack((
        cy * cz, -cy * sz, sy,
        cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy,
        sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy,
    ), axis=-1).reshape(angles.shape[:-1] + (3, 3))


def pose_totals(generated, reference, subject, valid):
    """Per-clip metric sums/counts; angular paths count clips, errors count frames."""
    import numpy as np
    generated, reference = np.asarray(generated), np.asarray(reference)
    valid = np.asarray(valid, dtype=bool)
    if generated.shape != reference.shape or generated.ndim != 2 or generated.shape[-1] != 6:
        raise ValueError("Expected matching (T,6) camera poses")
    if valid.shape != generated.shape[:1]:
        raise ValueError("Expected (T,) valid mask")
    totals = {}

    def add(name, values, mask):
        selected = np.asarray(values)[mask]
        totals[name] = (float(selected.sum()), int(selected.size))

    gen_finite = np.isfinite(generated).all(-1)
    ref_finite = np.isfinite(reference).all(-1)
    good = valid & gen_finite & ref_finite
    add("invalid_generated_pose_rate", ~gen_finite, valid)
    gp = np.where(np.isfinite(generated), generated, 0)
    rp = np.where(np.isfinite(reference), reference, 0)
    gr, rr = xyz_matrix(gp[:, 3:]), xyz_matrix(rp[:, 3:])
    transpose = lambda matrix: np.swapaxes(matrix, -1, -2)
    add("position_error", np.linalg.norm(gp[:, :3] - rp[:, :3], axis=-1), good)
    add("rotation_error_deg", angle_deg(transpose(rr) @ gr), good)

    # Anchor to the first real frame; never silently substitute a later valid
    # prediction when the first generated pose is invalid.
    first_frames = np.flatnonzero(valid)
    relative_good = np.zeros_like(good)
    relative_error = np.zeros_like(valid, dtype=float)
    if first_frames.size and good[first_frames[0]]:
        first = int(first_frames[0])
        g_relative, r_relative = gr[first].T @ gr, rr[first].T @ rr
        relative_error = angle_deg(transpose(r_relative) @ g_relative)
        relative_good = good
    add("first_frame_relative_rotation_error_deg", relative_error, relative_good)
    pairs = good[:-1] & good[1:]
    gd, rd = transpose(gr[:-1]) @ gr[1:], transpose(rr[:-1]) @ rr[1:]
    add("relative_rotation_increment_error_deg", angle_deg(transpose(rd) @ gd), pairs)
    add("position_increment_error", np.linalg.norm(np.diff(gp[:, :3] - rp[:, :3], axis=0), axis=-1), pairs)
    for name, positions, rotations, delta in (("generated", gp[:, :3], gr, gd),
                                              ("reference", rp[:, :3], rr, rd)):
        totals[name + "_angular_path_length_deg"] = (float(angle_deg(delta)[pairs].sum()), int(pairs.any()))
        totals[name + "_path_length"] = (float(np.linalg.norm(np.diff(positions, axis=0), axis=-1)[pairs].sum()), int(pairs.any()))
        if subject is not None:
            direction = np.asarray(subject)[..., :3] - positions
            distance = np.linalg.norm(direction, axis=-1)
            eligible = good & np.isfinite(direction).all(-1) & (distance > 1e-8)
            # Simulation camera local -Z points forward. No intrinsics needed.
            cosine = np.sum(-rotations[..., :, 2] * direction, axis=-1) / np.maximum(distance, 1e-8)
            add(name + "_subject_look_angle_deg", np.rad2deg(np.arccos(np.clip(cosine, -1, 1))), eligible)
    return totals


def summarize(totals):
    return {name: {"mean": total / count if count else None, "count": count}
            for name, (total, count) in totals.items()}


def diagnostic_baselines(reference, generated, valid):
    """Build controls in denormalized SIM coordinates, never normalization space.

    SIM Euler zero is standard/OpenCV diag(1,-1,-1), not CV identity.
    Only the explicitly named oracle baseline consumes reference pose values.
    """
    import numpy as np
    reference, generated = np.asarray(reference), np.asarray(generated)
    valid = np.asarray(valid, dtype=bool)
    if reference.shape != generated.shape or reference.ndim != 3 or reference.shape[-1] != 6:
        raise ValueError('Expected matching [batch, frames, 6] baseline inputs')
    if valid.shape != reference.shape[:2] or not valid.any(axis=1).all():
        raise ValueError('Baseline inputs require at least one valid frame per clip')
    first = np.argmax(valid, axis=1)
    batch = np.arange(len(reference))
    fixed_rotation = generated.copy()
    fixed_rotation[..., 3:] = 0
    return {
        'baseline_fixed_sim_zero_pose': np.zeros_like(reference),
        'baseline_generated_position_fixed_sim_zero_rotation': fixed_rotation,
        'baseline_freeze_generated_first_pose': np.broadcast_to(generated[batch, first, None], generated.shape).copy(),
        'oracle_freeze_reference_first_pose': np.broadcast_to(reference[batch, first, None], reference.shape).copy(),
    }


def self_test():
    """Small numerical contracts requiring NumPy, but no checkpoint or GPU."""
    import numpy as np
    reference = np.zeros((4, 6))
    reference[:, 4] = np.arange(4) * np.pi / 12
    generated = reference.copy()
    generated[:, 4] += np.pi
    subject = np.zeros((4, 6))
    subject[:, 2] = -10
    valid = np.array([True, True, True, False])
    generated[-1] = np.nan
    values = summarize(pose_totals(generated, reference, subject, valid))
    assert np.isclose(values["rotation_error_deg"]["mean"], 180)
    assert values["first_frame_relative_rotation_error_deg"]["mean"] < 1e-10
    assert np.isclose(values["generated_angular_path_length_deg"]["mean"], 30)
    assert values["invalid_generated_pose_rate"]["mean"] == 0
    assert values["rotation_error_deg"]["count"] == 3
    values = summarize(pose_totals(np.zeros((1, 6)), np.zeros((1, 6)), subject[:1], np.ones(1, bool)))
    assert values["generated_subject_look_angle_deg"]["mean"] == 0
    assert values["generated_angular_path_length_deg"]["mean"] is None
    invalid = generated.copy()
    invalid[0] = np.nan
    assert pose_totals(invalid, reference, subject, valid)["first_frame_relative_rotation_error_deg"][1] == 0
    print("Pose diagnostic math checks passed.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("train-config", "checkpoint", "et-config", "simulation-config", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Prefix of the frozen test order; zero evaluates all.")
    parser.add_argument("--batch-size", type=int, help="Defaults to each evaluation config's batch size.")
    parser.add_argument("--keyframes", nargs="*", type=int, default=[], choices=(1, 4))
    parser.add_argument("--save-trajectories", action="store_true", help="Save world-space trajectories for qualitative review.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        for name in ("train_config", "checkpoint", "et_config", "output"):
            if getattr(args, name) is None:
                parser.error("--" + name.replace("_", "-") + " is required")
    if args.limit < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("limit must be nonnegative and batch size positive")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    args.keyframes = sorted(set(args.keyframes))
    return args


def bind_environment(args):
    from dotenv import load_dotenv
    load_dotenv(args.repo_root / ".env", override=False)
    os.environ.setdefault("DIRECTOR_PROJECT_DIR", str(args.repo_root / "third_parties/DIRECTOR"))
    # Bind a prepared frozen cohort explicitly rather than accidentally using
    # the historical ET annotation path inherited from a login shell.
    for parent in args.et_config.parents:
        if (parent / "annotations.json").is_file() and (parent / "et-data").is_dir():
            os.environ["ET_DATA_DIR"] = str(parent / "et-data")
            os.environ["ET_CIN_LANG_PATH"] = str(parent / "annotations.json")
            break
    # Resolve environment-owned relative paths before changing the workdir;
    # generated CLIP statistics must be isolated in this output directory.
    for name in ("DIRECTOR_PROJECT_DIR", "ET_DATA_DIR", "ET_CIN_LANG_PATH",
                 "SIMULATION_DATA_PATH", "SIMULATION_NORMALIZATION_DATA_PATH", "CLIP_EMBEDDINGS_CACHE_DIR"):
        if os.environ.get(name):
            path = Path(os.environ[name]).expanduser()
            os.environ[name] = str(path if path.is_absolute() else (args.repo_root / path).resolve())
    os.environ.setdefault("CLIP_EMBEDDINGS_CACHE_DIR", str(args.repo_root))
    sys.path.insert(0, str(args.repo_root / "src"))


def evaluate_cohort(args, cfg, model, dataset_type, output, normalization):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import Subset
    from data.datamodule import CameraTrajectoryDataModule
    from data.sim_format import build_keyframing_mask, geometry_context_from_batch, to_simulation_format
    from data.simulation.dataset import SimulationDataset
    from data.simulation.utils import structured_conditioning_from_batch
    from generation import generate_lenscraft_trajectory
    from utils.device import move_batch_to_device
    from utils.normalization import prepare_evaluation_normalization

    if not cfg.data.dataset.config.normalize:
        raise ValueError("This runner requires normalized production model inputs")
    expected = "ETDataset" if dataset_type == "et" else "SimulationDataset"
    if expected not in cfg.data.dataset.config._target_:
        raise ValueError(f"Expected {expected} in {dataset_type} evaluation config")
    output.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    dm = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=args.batch_size or int(cfg.data.batch_size), num_workers=0,
        val_size=cfg.data.val_size, test_size=cfg.data.test_size,
        split_seed=int(cfg.data.get("split_seed", 42)),
        test_movement_types=cfg.get("test_movement_types"), test_fraction=cfg.get("test_fraction", 1.0),
    )
    # Both native ET conversion and simulation samples must use the transform
    # selected once from this checkpoint (or its explicit legacy training source).
    if getattr(model, "normalization_parameters", None) is not None:
        prepare_evaluation_normalization(dm, model, "lens_craft")
    else:
        dm.load_state_dict({"normalization": normalization})
    dm.setup()
    full_count = len(dm.test_dataset)
    indices = list(dm.test_dataset.indices)
    if args.limit:
        indices = indices[:args.limit]
        dm.test_dataset = Subset(dm.test_dataset.dataset, indices)
    dataset = dm.test_dataset.dataset
    if not len(dm.test_dataset):
        raise ValueError("The selected cohort is empty")
    if dataset_type == "et":
        filenames = dataset.original_dataset.root_filenames
        identifiers = ["et-" + str(filenames[i]) for i in indices]
    else:
        identifiers = [dataset.simulation_files[i].stem for i in indices]
        if any(not torch.equal(value.cpu(), dataset.normalization_parameters[key][field].cpu())
               for key, fields in normalization.items() for field, value in fields.items()):
            raise ValueError("Simulation evaluation normalization differs from the checkpoint training data")
    modes = ["prompt_original", "prompt_shuffled", "reconstruction"] + [f"keyframe_prompt_k{k}" for k in args.keyframes]
    baseline_modes = ['baseline_fixed_sim_zero_pose', 'baseline_generated_position_fixed_sim_zero_rotation',
                      'baseline_freeze_generated_first_pose', 'oracle_freeze_reference_first_pose']
    modes += baseline_modes
    manifest = {
        "dataset_type": dataset_type, "full_test_count": full_count, "selected_count": len(indices),
        "indices": indices, "item_ids_sha256": hashlib.sha256(json.dumps(identifiers).encode()).hexdigest(),
        "seed": seed, "batch_size": dm.batch_size, "modes": modes,
        "resolved_data_config": OmegaConf.to_container(cfg.data, resolve=True),
        "shuffle": "One-place cyclic shift within each batch; singleton unchanged; recorded per sample.",
        "keyframes": "Deterministic per-ID visible frames; explicit pose constraints condition generation independently of caption weights.",
        "reference_assisted_modes": ["reconstruction", "oracle_freeze_reference_first_pose"] + [m for m in modes if m.startswith("keyframe_")],
        "diagnostic_baselines": {
            'modes': baseline_modes,
            'constant_rotation': 'SIM/OpenGL intrinsic XYZ Euler (0,0,0); standard/OpenCV rotation diag(1,-1,-1).',
            'scope': 'Diagnostics, not usable camera systems. Oracle first-reference-pose baseline is reference-assisted.',
        },
        "metric_units": "Dataset world units and degrees; paths per clip, errors per valid frame; no intrinsics assumed.",
        "relative_rotation_diagnostic": "Compares R_g(0).T R_g(t) to R_r(0).T R_r(t); reference used only for scoring.",
    }
    if dataset_type == "et":
        annotations = Path(cfg.data.dataset.config.et_cin_lang_path)
        manifest["annotations"] = {"path": str(annotations), "sha256": sha256(annotations)}
        native_split = Path(cfg.data.dataset.config.dataset_dir) / f"{cfg.data.dataset.config.set_name}_{cfg.data.dataset.config.split}_split.txt"
        manifest["native_split"] = {"path": str(native_split), "sha256": sha256(native_split)}
    write_json(output / "manifest.json", manifest)
    aggregates = {mode: defaultdict(lambda: [0., 0]) for mode in modes}
    records = []
    trajectories = defaultdict(list)
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(dm.test_dataloader()):
            batch = move_batch_to_device(raw_batch, args.device)
            camera, subject, volume, padding = to_simulation_format(batch, dataset_type, target_len=model.decoder.seq_length)
            geometry_context = geometry_context_from_batch(batch, dataset_type, target_len=camera.shape[1])
            if padding is None:
                padding = torch.zeros(camera.shape[:2], dtype=torch.bool, device=camera.device)
            view = camera, subject, volume, padding
            caption = structured_conditioning_from_batch(batch)
            if caption is None:
                raise ValueError("Structured conditioning is required")
            item_ids = identifiers[len(records):len(records) + camera.shape[0]]
            permutation = torch.roll(torch.arange(camera.shape[0], device=camera.device), 1)
            presence = batch.get('prompt_none_mask')
            shuffled_presence = None if presence is None else presence[permutation]
            generated = {
                "prompt_original": generate_lenscraft_trajectory(model, view, padding, caption, 1., caption_presence_mask=presence, geometry_context=geometry_context),
                "prompt_shuffled": generate_lenscraft_trajectory(model, view, padding, caption[:, permutation], 1., caption_presence_mask=shuffled_presence, geometry_context=geometry_context),
                "reconstruction": generate_lenscraft_trajectory(model, view, padding, caption, 0., caption_presence_mask=presence, geometry_context=geometry_context),
            }
            for count in args.keyframes:
                sample_seeds = [int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16], 16) for name in item_ids]
                source_mask = build_keyframing_mask(camera.shape[0], camera.device, camera.shape[1],
                                                   num_keyframes=count, padding_mask=padding, sample_seeds=sample_seeds)
                generated[f"keyframe_prompt_k{count}"] = generate_lenscraft_trajectory(
                    model, view, source_mask, caption, .5,
                    caption_presence_mask=presence, known_camera_mask=~source_mask,
                    geometry_context=geometry_context,
                )
            if batch_index == 0:
                controls = {}
                for name, replacement in (("zero", torch.zeros_like(camera)), ("shuffled", camera[permutation]), ("none", None)):
                    control = generate_lenscraft_trajectory(model, (replacement, subject, volume, padding), padding, caption, 1., caption_presence_mask=presence, geometry_context=geometry_context)
                    difference = (control - generated["prompt_original"]).abs()
                    controls[name + "_max_abs_difference"] = float(difference.max()) if torch.isfinite(difference).all() else None
                controls["passed_at_1e-5"] = all(value is not None and value <= 1e-5 for value in controls.values())
                controls["sample_count"] = camera.shape[0]
                manifest["source_camera_invariance"] = controls
                write_json(output / "manifest.json", manifest)
                if not controls["passed_at_1e-5"]:
                    raise RuntimeError(f"Caption-only source-camera invariance failed: {controls}")
            reference, denorm_subject, _ = SimulationDataset.normalize_item(camera, subject, volume, False)
            reference = reference.cpu().numpy().astype(np.float64)
            denorm_subject = None if denorm_subject is None else denorm_subject.cpu().numpy().astype(np.float64)
            valid = (~padding).cpu().numpy()
            outputs = {mode: SimulationDataset.normalize_item(value, None, None, False)[0].cpu().numpy().astype(np.float64)
                       for mode, value in generated.items()}
            outputs.update(diagnostic_baselines(reference, outputs['prompt_original'], valid))
            if args.save_trajectories:
                trajectories["reference"].append(reference)
                trajectories["valid"].append(valid)
                if denorm_subject is not None:
                    trajectories["subject"].append(denorm_subject)
                for mode, value in outputs.items():
                    trajectories[mode].append(value)
            for i, identifier in enumerate(item_ids):
                record = {"item_id": identifier, "caption": batch["text_prompts"][i],
                          "shuffled_caption_source": item_ids[int(permutation[i])],
                          "shuffled_caption_identical": bool(torch.equal(caption[:, i], caption[:, permutation[i]])),
                          "present_field_count": int(batch["prompt_none_mask"][i].sum()), "metrics": {}}
                for mode in modes:
                    totals = pose_totals(outputs[mode][i], reference[i], None if denorm_subject is None else denorm_subject[i], valid[i])
                    record["metrics"][mode] = summarize(totals)
                    for key, (value, count) in totals.items():
                        aggregates[mode][key][0] += value
                        aggregates[mode][key][1] += count
                records.append(record)
            if batch_index % 5 == 0 or len(records) == len(identifiers):
                progress = {"dataset_type": dataset_type, "completed_samples": len(records),
                            "expected_samples": len(identifiers), "elapsed_seconds": time.monotonic() - started}
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    if len(records) != len(identifiers):
        raise RuntimeError("Incomplete cohort")
    result = {"sample_count": len(records), "metrics": {mode: summarize(totals) for mode, totals in aggregates.items()},
              "source_camera_invariance": manifest["source_camera_invariance"],
              "shuffle_identical_prompt_count": sum(r["shuffled_caption_identical"] for r in records),
              "elapsed_seconds": time.monotonic() - started}
    write_json(output / "per_clip.json", records)
    write_json(output / "summary.json", result)
    if args.save_trajectories:
        np.savez_compressed(output / "trajectories.npz", item_ids=np.asarray(identifiers),
                            **{key: np.concatenate(value) for key, value in trajectories.items()})
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    bind_environment(args)
    import torch
    from omegaconf import OmegaConf, open_dict
    from data.datamodule import CameraTrajectoryDataModule
    from models.factory import load_model
    from utils.normalization import prepare_evaluation_normalization
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    train_cfg = OmegaConf.load(args.train_config)
    et_cfg = OmegaConf.load(args.et_config)
    model_cfg = deepcopy(train_cfg)
    with open_dict(model_cfg.training.model.inference):
        model_cfg.training.model.inference.checkpoint_path = str(args.checkpoint)
        model_cfg.training.model.inference.config = str(args.train_config)
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output)
    # ET conversion must use the checkpoint's simulation normalization, not
    # statistics selected by another datamodule left in this process.
    simulation_data = (OmegaConf.select(train_cfg, "data.dataset.config")
                       or OmegaConf.select(train_cfg, "data.dataset.simulation_config"))
    if simulation_data is None or "SimulationDataset" not in simulation_data._target_:
        raise ValueError("--train-config must identify the checkpoint's simulation normalization dataset")
    simulation_data = deepcopy(simulation_data)
    if simulation_data.get("normalization_frame_count") is None:
        with open_dict(simulation_data):
            simulation_data.normalization_frame_count = int(
                OmegaConf.select(train_cfg, "data.dataset.info.seq_length", default=30))
    model = load_model(model_cfg, "lens_craft", torch.device(args.device))
    normalization_module = CameraTrajectoryDataModule(
        dataset_config=simulation_data, batch_size=1, num_workers=0,
    )
    normalization_provenance = prepare_evaluation_normalization(normalization_module, model, "lens_craft")
    normalization = normalization_module.state_dict()["normalization"]
    normalization_json = {key: {field: value.cpu().tolist() if torch.is_tensor(value) else value
                               for field, value in fields.items()} for key, fields in normalization.items()}
    write_json(args.output / "normalization.json", normalization_json)
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "device": args.device,
                "torch_version": torch.__version__, "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
                "train_config": {"path": str(args.train_config), "sha256": sha256(args.train_config)},
                "et_config": {"path": str(args.et_config), "sha256": sha256(args.et_config)},
                "normalization_sha256": sha256(args.output / "normalization.json"),
                "normalization": normalization_provenance,
                "model_inference": OmegaConf.to_container(model_cfg.training.model.inference, resolve=True),
                "keyframe_pose_conditioning": bool(model.keyframe_pose_conditioning),
                "semantic_fields": 10, "semantic_values": 93,
                "script_sha256": sha256(__file__),
                "source_sha256": {str(path.relative_to(args.repo_root)): sha256(path)
                                  for path in sorted((args.repo_root / "src").rglob("*.py"))}}
    if args.simulation_config:
        manifest["simulation_config"] = {"path": str(args.simulation_config), "sha256": sha256(args.simulation_config)}
    write_json(args.output / "manifest.json", manifest)
    results = {"et": evaluate_cohort(args, et_cfg, model, "et", args.output / "et", normalization)}
    if args.simulation_config:
        sim_cfg = OmegaConf.load(args.simulation_config)
        results["simulation"] = evaluate_cohort(args, sim_cfg, model, "simulation", args.output / "simulation", normalization)
    write_json(args.output / "summary.json", results)
    write_json(args.output / "status.json", {"exit_code": 0, "completed_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(results, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
