#!/usr/bin/env python3
"""Evaluate a small common held-out sample across LensCraft and its baselines.

Example (the environment also supplies baseline checkpoint/data paths)::

    python scripts/run_pilot_evaluation.py --dataset-path /data/simulation \
        --lenscraft-checkpoint /runs/train/best.ckpt \
        --lenscraft-config /runs/train/.hydra/config.yaml \
        --clatr-checkpoint /runs/clatr/best.ckpt --output-dir /runs/pilot \
        --split-manifest /runs/metadata/split_indices.json --samples 128

Add --dry-run to print commands without importing model dependencies or writing
files. Native baseline lengths are retained; the evaluator handles conversion
to the reference and metric coordinate/temporal formats.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


MODELS = ("lenscraft", "ccdm", "et", "gendop")
NATIVE_LENGTHS = {"lenscraft": 30, "ccdm": 300, "et": 300, "gendop": 30}
METRICS = ("clatr_score", "fcd", "precision", "recall", "density", "coverage", "clip_score")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--lenscraft-checkpoint", type=Path, required=True)
    parser.add_argument("--lenscraft-config", type=Path, required=True)
    parser.add_argument("--clatr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--samples", type=int, default=128, help="Requested samples, at most 128; must divide into full batches.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0", help="Visible CUDA device, e.g. cuda:0 after CUDA_VISIBLE_DEVICES=1; or cpu/auto.")
    parser.add_argument("--label", default="pilot", help="Experiment description recorded in the report.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 32 <= args.samples <= 128:
        parser.error("--samples must be between 32 and 128")
    if args.batch_size < 1 or args.samples % args.batch_size:
        parser.error("--batch-size must be positive and divide --samples exactly")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    # DIRECTOR PRDC uses five chunks and a 4th-nearest-neighbour lookup.
    if min(_chunk_sizes(args.samples)) < 4:
        parser.error("This sample count leaves a PRDC chunk smaller than four; use 32, 64, 96, or 128")
    args.models = list(dict.fromkeys(args.models))
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    for field in ("project_dir", "dataset_path", "lenscraft_checkpoint", "lenscraft_config", "clatr_checkpoint", "output_dir", "split_manifest"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    return args


def _chunk_sizes(count):
    size = math.ceil(count / 5)
    return [min(size, count - start) for start in range(0, count, size)] if count else []


def _override(key, value):
    # subprocess receives argv directly; the quotes are Hydra value syntax,
    # necessary for checkpoint names containing '=' and paths with spaces.
    return f"{key}={json.dumps(str(value))}"


def build_command(args, model):
    run_dir = args.output_dir / model
    command = [sys.executable, str(args.project_dir / "src/test.py")]
    if model != "lenscraft":
        command.append(f"training/model={model}")
    command += [
        "data/dataset=default",
        _override("data.dataset.config.data_path", args.dataset_path),
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        "data.val_size=0.2", "data.test_size=0.2", "+seed=42",
        "+data.dataset.config.allowed_movement_types=[]",
        f"+limit_test_batches={args.samples // args.batch_size}", "+n_boot=0",
        "+eval_set=pilot", "caption_top1_metric=false", "tsne=false",
        "baseline_norm_ablation=true", "clatr_backend=native",
        _override("clatr_native_checkpoint_path", args.clatr_checkpoint),
        _override("ref_model.inference.checkpoint_path", args.lenscraft_checkpoint),
        _override("ref_model.inference.config", args.lenscraft_config),
        _override("device", args.device), _override("output_dir", run_dir),
        _override("trajectory_cache_dir", args.output_dir / "trajectory_cache"),
        _override("hydra.run.dir", run_dir / "hydra"), "hydra.job.chdir=true",
    ]
    if model == "lenscraft":
        command += [
            _override("training.model.inference.checkpoint_path", args.lenscraft_checkpoint),
            _override("training.model.inference.config", args.lenscraft_config),
        ]
    return command


def _file_record(path):
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def preflight(args):
    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    import torch

    load_dotenv(args.project_dir / ".env", override=False)
    sys.path.insert(0, str(args.project_dir / "src"))
    from data.simulation.metadata import resolve_dataset_root

    for path in (args.lenscraft_checkpoint, args.lenscraft_config, args.clatr_checkpoint, args.project_dir / "src/test.py"):
        if not path.is_file():
            raise ValueError(f"Required file does not exist: {path}")
    training_config = OmegaConf.load(args.lenscraft_config)
    for field in ("val_size", "test_size"):
        if float(training_config.data[field]) != 0.2:
            raise ValueError(f"The supplied training config data.{field} is not 0.2; preserve its original split before using this runner")
    configured_filter = training_config.data.dataset.config.get("allowed_movement_types")
    if configured_filter:
        raise ValueError("The supplied training config filters movement types; its split is not the unfiltered pilot split")

    args.dataset_path = resolve_dataset_root(args.dataset_path)
    if not (args.dataset_path / "parameter_dictionary.msgpack").is_file():
        raise ValueError("Simulation parameter_dictionary.msgpack is missing")
    files = sorted(args.dataset_path.glob("simulation_*.msgpack"))
    if not files:
        raise ValueError(f"No simulation samples found in {args.dataset_path}")
    # Match CameraTrajectoryDataModule exactly, including float rounding.
    train_count = int((1 - 0.2 - 0.2) * len(files))
    validation_count = int(0.2 * len(files))
    permutation = torch.randperm(len(files), generator=torch.Generator().manual_seed(42)).tolist()
    splits = {
        "train": permutation[:train_count],
        "validation": permutation[train_count:train_count + validation_count],
        "test": permutation[train_count + validation_count:],
    }
    if args.split_manifest:
        saved = json.loads(args.split_manifest.read_text())
        for name, indices in splits.items():
            if saved.get(name) != indices:
                raise ValueError(f"Saved {name} split differs from this dataset's seed-42 split; refusing a misleading held-out comparison")
    selected = splits["test"][:args.samples]
    if len(selected) < args.samples:
        raise ValueError(f"Test split has {len(selected)} samples, fewer than requested {args.samples}; select a smaller --samples value")
    inventory_hash = hashlib.sha256("\n".join(path.name for path in files).encode()).hexdigest()
    manifest_path = args.dataset_path / "manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    return {
        "dataset_path": str(args.dataset_path), "dataset_manifest": dataset_manifest,
        "dataset_size": len(files), "sorted_filename_inventory_sha256": inventory_hash,
        "split_seed": 42, "split_counts": {name: len(indices) for name, indices in splits.items()},
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "split_manifest_verified": bool(args.split_manifest),
        "sample_count": len(selected), "selected_indices": selected,
        "selected_files": [_file_record(files[index]) for index in selected],
        "parameter_dictionary": _file_record(args.dataset_path / "parameter_dictionary.msgpack"),
        "torch_version": torch.__version__,
    }


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_summary(args, selection, results):
    caveats = [
        f"Exploratory comparison on {args.samples} identical held-out samples with seed 42; bootstrap is disabled.",
        "FCD uses 256-dimensional features: this small sample has singular covariance and is insufficient for a stable model ranking.",
        "Native input/output frame counts differ by model; the evaluator converts trajectories for reference-model and CLaTr metrics.",
        "The shared prompt_generation row uses each baseline's normalized mode. Raw and LensCraft-initialized variants remain separate in the full metrics JSON.",
        "clip_score uses LensCraft's encoder and is not an independent assessment; native CLaTr metrics use the supplied evaluation checkpoint.",
        "Any GenDoP fallback trajectories are counted from warnings and remain included in its metrics.",
    ]
    payload = {"label": args.label, "sample_count": args.samples, "selection": selection,
               "clatr_checkpoint": str(args.clatr_checkpoint), "results": results, "caveats": caveats}
    write_json(args.output_dir / "summary.json", payload)
    lines = [f"Pilot evaluation: {args.label}", "", f"Samples: {args.samples}; native CLaTr: `{args.clatr_checkpoint}`.", "",
             "| Model | Status | CLaTr score ↑ | FCD ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ | Clip score ↑ |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model in args.models:
        result = results.get(model, {"status": "pending"})
        values = result.get("prompt_generation", {})
        cells = [f"{values[name]:.4f}" if name in values else "—" for name in METRICS]
        lines.append("| " + " | ".join([model, result["status"], *cells]) + " |")
    lines += ["", *[f"- {caveat}" for caveat in caveats], ""]
    for model, result in results.items():
        lines.append(f"- {model}: [log]({model}/evaluation.log); {result.get('fallback_warnings', 0)} fallback warnings.")
        if result.get("error"):
            lines.append(f"  Failure: {result['error']}")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        for model in args.models:
            print(f"{model} ({NATIVE_LENGTHS[model]} native frames):")
            print(shlex.join(build_command(args, model)))
        print(f"\nRead-only preview; {args.samples} common test samples, seed 42, native CLaTr. Input files and split are verified on execution.")
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty to avoid mixing experiments: {args.output_dir}")
    selection = preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"SIMULATION_DATA_PATH": str(args.dataset_path),
                        "TEST_CHECKPOINT_PATH": str(args.lenscraft_checkpoint),
                        "TEST_CONFIG_PATH": str(args.lenscraft_config),
                        "CLATR_NATIVE_CHECKPOINT_PATH": str(args.clatr_checkpoint),
                        "PYTHONUNBUFFERED": "1", "HYDRA_FULL_ERROR": "1"})
    # Keep inherited baseline paths and cache paths absolute when Hydra changes
    # into the model's output directory.
    for key in ("CLIP_EMBEDDINGS_CACHE_DIR", "CCDM_CHECKPOINT_PATH", "CCDM_DATA_DIR", "DIRECTOR_PROJECT_DIR", "ET_DATA_DIR", "ET_CIN_LANG_PATH", "GENDOP_CHECKPOINT_PATH"):
        value = environment.get(key)
        if value:
            path = Path(value).expanduser()
            environment[key] = str((args.project_dir / path).resolve() if not path.is_absolute() else path)
    environment.setdefault("CLIP_EMBEDDINGS_CACHE_DIR", str(args.project_dir))
    commands = {model: build_command(args, model) for model in args.models}
    metadata = {
        "label": args.label, "created_at": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(args.project_dir), "python": sys.executable,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"), "device": args.device,
        "checkpoints": {"lenscraft": _file_record(args.lenscraft_checkpoint), "native_clatr": _file_record(args.clatr_checkpoint)},
        "lenscraft_config": _file_record(args.lenscraft_config), "selection": selection,
        "baseline_inputs": {key: environment.get(key) for key in (
            "CCDM_CHECKPOINT_PATH", "CCDM_DATA_DIR", "DIRECTOR_PROJECT_DIR",
            "ET_DATA_DIR", "ET_CIN_LANG_PATH", "GENDOP_CHECKPOINT_PATH")},
        "lenscraft_config_sha256": hashlib.sha256(args.lenscraft_config.read_bytes()).hexdigest(),
        "native_frame_counts": {model: NATIVE_LENGTHS[model] for model in args.models},
        "batch_size": args.batch_size, "bootstrap_replicates": 0, "commands": commands,
    }
    write_json(args.output_dir / "run_manifest.json", metadata)
    results = {}
    save_summary(args, selection, results)
    for model, command in commands.items():
        run_dir = args.output_dir / model
        run_dir.mkdir()
        print(f"Starting {model}: {run_dir / 'evaluation.log'}", flush=True)
        start = time.monotonic()
        result = {"status": "failed", "command": command}
        results[model] = result
        try:
            child_environment = dict(environment, TEST_OUTPUT_DIR=str(run_dir))
            with (run_dir / "evaluation.log").open("w") as log:
                completed = subprocess.run(command, cwd=args.project_dir, env=child_environment,
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            result["returncode"] = completed.returncode
            log_text = (run_dir / "evaluation.log").read_text(errors="replace")
            result["fallback_warnings"] = log_text.lower().count("falling back to a default trajectory")
            if completed.returncode:
                raise RuntimeError(f"Evaluator exited with code {completed.returncode}; see evaluation.log")
            files = list(run_dir.glob("metrics_*.json"))
            if len(files) != 1:
                raise RuntimeError(f"Expected one metrics JSON, found {len(files)}")
            metrics_payload = json.loads(files[0].read_text())
            values = metrics_payload["metrics"]["prompt_generation"]
            shared = {name: float(values[f"prompt_generation/{name}"]) for name in METRICS if f"prompt_generation/{name}" in values}
            if not all(name in shared and math.isfinite(shared[name]) for name in METRICS[:-1]):
                raise RuntimeError("Missing or non-finite native CLaTr comparison metrics")
            if not all(math.isfinite(value) for value in shared.values()):
                raise RuntimeError("Non-finite supplementary comparison metric")
            result.update(status="complete", prompt_generation=shared, metrics_path=str(files[0]))
        except Exception as error:
            result["error"] = str(error)
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        save_summary(args, selection, results)
        print(f"{model}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    print(f"Report: {args.output_dir / 'summary.md'}", flush=True)
    return int(any(result["status"] != "complete" for result in results.values()))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"Pilot preflight failed: {error}", file=sys.stderr)
        sys.exit(2)
