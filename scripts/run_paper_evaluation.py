#!/usr/bin/env python3
"""Run the full static/dynamic paper evaluations with resumable status and logs."""
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
LENGTHS = dict(zip(MODELS, (30, 300, 300, 30)))
LEARNED_METRICS = ("clatr_score", "fcd", "precision", "recall", "density", "coverage", "clip_score")
PATH_ENV = ("CCDM_CHECKPOINT_PATH", "CCDM_DATA_DIR", "DIRECTOR_PROJECT_DIR", "ET_DATA_DIR", "ET_CIN_LANG_PATH", "GENDOP_CHECKPOINT_PATH", "CLIP_EMBEDDINGS_CACHE_DIR")
COHORTS = {"static": ["static"], "dynamic": ["circular", "zigzag", "linear", "spiral", "figureEight", "wave", "pendulum", "orbital", "bounce"]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    for field in ("dataset-path", "lenscraft-checkpoint", "lenscraft-config", "clatr-checkpoint", "output-dir"):
        p.add_argument("--" + field, type=Path, required=True)
    for field in ("semantic-evaluator-checkpoint", "semantic-evaluator-config", "split-manifest"):
        p.add_argument("--" + field, type=Path)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--test-fraction", type=float, default=1.0,
                   help="Use the first ceil(N_test * fraction) held-out indices before cohort filtering")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.n_boot < 0 or args.n_boot == 1 or args.batch_size < 1 or args.num_workers < 0:
        p.error("n-boot must be zero or at least two; workers nonnegative; batch-size positive")
    if not 0 < args.test_fraction <= 1:
        p.error("--test-fraction must lie in (0, 1]")
    if args.semantic_evaluator_checkpoint is not None and args.semantic_evaluator_config is None:
        p.error("A custom --semantic-evaluator-checkpoint requires --semantic-evaluator-config")
    # Explicit fixed full-model default; the evaluator is loaded independently.
    args.semantic_evaluator_checkpoint = args.semantic_evaluator_checkpoint or args.lenscraft_checkpoint
    args.semantic_evaluator_config = args.semantic_evaluator_config or args.lenscraft_config
    for field, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, field, value.expanduser().resolve())
    return args


def override(key, value):
    return f"{key}={json.dumps(str(value))}"


def build_stages(args):
    stages = []
    for kind in ("evaluation", "efficiency"):
        for model in MODELS:
            for subset in (("static", "dynamic") if kind == "evaluation" else ("all",)):
                name = f"{kind}_{model}_{subset}"
                command = [sys.executable, str(args.project_dir / "src" / ("test.py" if kind == "evaluation" else "efficiency.py"))]
                if model != "lenscraft":
                    command.append(f"training/model={model}")
                if model == "et":
                    command.append("training.model.inference.et_type=ca")
                command += [
                    "data/dataset=default", override("data.dataset.config.data_path", args.dataset_path),
                    "data.val_size=0.2", "data.test_size=0.2", "+seed=42",
                    "+data.dataset.config.allowed_movement_types=[]",
                    f"data.batch_size={args.batch_size if kind == 'evaluation' else 16}",
                    f"data.num_workers={args.num_workers}", override("device", args.device),
                    override("output_dir", args.output_dir / "results"),
                    override("hydra.run.dir", args.output_dir / "hydra" / name), "hydra.job.chdir=true",
                    override("ref_model.inference.checkpoint_path", args.semantic_evaluator_checkpoint),
                    override("ref_model.inference.config", args.semantic_evaluator_config),
                    "clatr_backend=native", override("clatr_native_checkpoint_path", args.clatr_checkpoint),
                ]
                if model == "lenscraft":
                    command += [override("training.model.inference.checkpoint_path", args.lenscraft_checkpoint),
                                override("training.model.inference.config", args.lenscraft_config)]
                if kind == "evaluation":
                    command += [f"+eval_set={subset}", f"+n_boot={args.n_boot}", f"test_fraction={args.test_fraction}",
                                "test_movement_types=[" + ",".join(COHORTS[subset]) + "]", "caption_top1_metric=false", "tsne=false",
                                "baseline_norm_ablation=true", "keyframes.counts=[1,2,4,8,26]", "keyframes.seed=42",
                                "geometry_metrics.enabled=true", override("trajectory_cache_dir", args.output_dir / "trajectory_cache")]
                stages.append({"name": name, "kind": kind, "model": model, "set": subset, "command": command})
    stages.append({"name": "aggregate", "kind": "aggregate", "command": [sys.executable,
                   str(args.project_dir / "src/aggregate_results.py"), "--results-dir", str(args.output_dir / "results"),
                   "--out-dir", str(args.output_dir / "tables")]})
    return stages


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(args, stages):
    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    import torch
    load_dotenv(args.project_dir / ".env", override=False)
    sys.path.insert(0, str(args.project_dir / "src"))
    from data.simulation.metadata import resolve_dataset_root
    args.dataset_path = resolve_dataset_root(args.dataset_path)
    files = sorted(args.dataset_path.glob("simulation_*.msgpack"))
    if not files or not (args.dataset_path / "parameter_dictionary.msgpack").is_file():
        raise ValueError("Simulation samples or parameter_dictionary.msgpack missing")
    config = OmegaConf.load(args.lenscraft_config)
    if any(float(config.data[field]) != 0.2 for field in ("val_size", "test_size")):
        raise ValueError("Generator split must be 60/20/20 for this runner")
    if config.data.dataset.config.get("allowed_movement_types"):
        raise ValueError("Generator was trained on a filtered dataset; cannot assume this holdout")
    indices = torch.randperm(len(files), generator=torch.Generator().manual_seed(42)).tolist()
    train_size, val_size = int((1 - .2 - .2) * len(files)), int(.2 * len(files))
    splits = {"train": indices[:train_size], "validation": indices[train_size:train_size + val_size],
              "test": indices[train_size + val_size:]}
    args.expected_test_counts = {"original_holdout": len(splits["test"]),
                                 "fractional_holdout": max(1, math.ceil(len(splits["test"]) * args.test_fraction))}
    if args.split_manifest:
        saved = json.loads(args.split_manifest.read_text())
        if any(saved.get(key) != value for key, value in splits.items()):
            raise ValueError("Saved training split does not match this dataset/seed")
    environment = os.environ.copy()
    for name in PATH_ENV:
        if environment.get(name):
            path = Path(environment[name]).expanduser()
            environment[name] = str((args.project_dir / path).resolve() if not path.is_absolute() else path.resolve())
    environment.setdefault("CLIP_EMBEDDINGS_CACHE_DIR", str(args.project_dir))
    environment.update(SIMULATION_DATA_PATH=str(args.dataset_path), TEST_CHECKPOINT_PATH=str(args.lenscraft_checkpoint),
                       TEST_CONFIG_PATH=str(args.lenscraft_config), SEMANTIC_EVALUATOR_CHECKPOINT_PATH=str(args.semantic_evaluator_checkpoint),
                       SEMANTIC_EVALUATOR_CONFIG_PATH=str(args.semantic_evaluator_config), CLATR_NATIVE_CHECKPOINT_PATH=str(args.clatr_checkpoint),
                       TEST_OUTPUT_DIR=str(args.output_dir / "results"), PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
    inputs = {field: getattr(args, field) for field in ("lenscraft_checkpoint", "lenscraft_config", "semantic_evaluator_checkpoint", "semantic_evaluator_config", "clatr_checkpoint")}
    for name in ("CCDM_CHECKPOINT_PATH", "GENDOP_CHECKPOINT_PATH"):
        if not environment.get(name):
            raise ValueError(f"Missing baseline input environment variable: {name}")
        inputs[name] = Path(environment[name])
    if environment.get("ET_CIN_LANG_PATH"):
        inputs["ET_CIN_LANG_PATH"] = Path(environment["ET_CIN_LANG_PATH"])
    if not environment.get("DIRECTOR_PROJECT_DIR"):
        raise ValueError("Missing baseline input environment variable: DIRECTOR_PROJECT_DIR")
    inputs["ET_CHECKPOINT_PATH"] = Path(environment["DIRECTOR_PROJECT_DIR"]) / "checkpoints/director/ca-mixed-e449.ckpt"
    fingerprints = {}
    cache = {}
    for name, path in inputs.items():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
        if str(path) not in cache:
            cache[str(path)] = sha256(path)
        fingerprints[name] = {"path": str(path), "sha256": cache[str(path)]}
    args.checkpoint_sha256 = {"semantic_evaluator": fingerprints["semantic_evaluator_checkpoint"]["sha256"],
                             "clatr_evaluator": fingerprints["clatr_checkpoint"]["sha256"]}
    source_files = sorted(path for root in ("src", "config", "scripts", "third_parties") for path in (args.project_dir / root).rglob("*")
                          if path.is_file() and path.suffix in (".py", ".yaml", ".yml"))
    source = {str(path.relative_to(args.project_dir)): sha256(path) for path in source_files}
    inventory = [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    manifest = {"protocol": 1, "inputs": fingerprints, "source_sha256": source,
                "dataset": {"path": str(args.dataset_path), "sample_count": len(files),
                            "inventory_sha256": hashlib.sha256(json.dumps(inventory).encode()).hexdigest(),
                            "parameter_dictionary_sha256": sha256(args.dataset_path / "parameter_dictionary.msgpack"),
                            "split_seed": 42, "split_counts": {k: len(v) for k, v in splits.items()},
                            "test_indices_sha256": hashlib.sha256(json.dumps(splits["test"]).encode()).hexdigest()},
                "baseline_environment": {key: environment.get(key) for key in PATH_ENV},
                "device": args.device, "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
                "test_fraction": args.test_fraction,
                "python": sys.executable, "torch_version": torch.__version__, "native_lengths": LENGTHS,
                "stages": build_stages(args), "architecture_ablations": "unavailable: separately trained ablation checkpoints were not supplied"}
    return manifest, environment


def expected_modes(model):
    if model != "lenscraft":
        return {"prompt_generation", "prompt_generation_no_norm", "prompt_generation_norm_lenscraft_init"}
    return {"prompt_generation", "reconstruction", "hybrid_generation"} | {
        f"{mode}_k{k}" for mode in ("key_framing", "key_framing+prompt") for k in (1, 2, 4, 8, 26)}


def validate_output(args, stage, identities):
    if stage["kind"] == "aggregate":
        paths = [args.output_dir / "tables" / name for name in (
            "all_metrics.csv", "table1_sota.md", "table2_multimodal.md", "table3_ablation.md", "table4_efficiency.md",
            "table5_baseline_normalization.md", "table6_keyframes.md", "table7_geometry.md")]
    else:
        model = "lens_craft" if stage["model"] == "lenscraft" else stage["model"]
        prefix = "metrics" if stage["kind"] == "evaluation" else "efficiency"
        paths = [path for path in (args.output_dir / "results").glob(f"{prefix}_{model}*.json")
                 if stage["kind"] != "evaluation" or path.stem.endswith("_" + stage["set"])]
        if len(paths) != 1:
            raise ValueError(f"Expected one {stage['name']} output, found {len(paths)}")
        payload = json.loads(paths[0].read_text())
        if payload.get("model_type") != model:
            raise ValueError("Output model does not match stage")
        if stage["kind"] == "evaluation":
            if payload.get("set") != stage["set"] or set(payload.get("metrics", {})) != expected_modes(stage["model"]):
                raise ValueError("Missing/unexpected evaluation modes or subset")
            if payload.get("clatr_backend") != "native":
                raise ValueError("Native CLaTr required")
            if payload.get("test_movement_types") != COHORTS[stage["set"]]:
                raise ValueError("Unexpected held-out movement cohort")
            if payload.get("test_fraction") != args.test_fraction:
                raise ValueError("Unexpected held-out test fraction")
            counts = payload.get("test_sample_counts", {})
            for key, expected in getattr(args, "expected_test_counts", {}).items():
                if counts.get(key) != expected:
                    raise ValueError(f"Unexpected test sample count: {key}")
            for mode, metrics in payload["metrics"].items():
                for name in LEARNED_METRICS:
                    value = metrics.get(f"{mode}/{name}")
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"Missing/non-finite {mode}/{name}")
                    if args.n_boot:
                        boot = payload.get("bootstrap_std", {}).get(mode, {}).get(f"{mode}/{name}")
                        if not isinstance(boot, list) or len(boot) != 2 or not all(
                            isinstance(v, (int, float)) and math.isfinite(v) for v in boot
                        ):
                            raise ValueError(f"Missing/non-finite bootstrap estimate {mode}/{name}")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not math.isfinite(value):
                        raise ValueError(f"Non-finite metric {key}")
            for evaluator in ("semantic_evaluator", "clatr_evaluator"):
                identity = payload.get("evaluation_provenance", {}).get(evaluator, {})
                fingerprint = identity.get("fingerprint")
                if not isinstance(fingerprint, str) or not fingerprint:
                    raise ValueError(f"Missing {evaluator} fingerprint")
                expected_path = args.semantic_evaluator_checkpoint if evaluator == "semantic_evaluator" else args.clatr_checkpoint
                if Path(identity.get("checkpoint", {}).get("path", "")).resolve() != expected_path.resolve():
                    raise ValueError(f"Unexpected {evaluator} checkpoint")
                expected_hash = getattr(args, "checkpoint_sha256", {}).get(evaluator)
                if expected_hash and identity.get("checkpoint", {}).get("sha256") != expected_hash:
                    raise ValueError(f"Unexpected {evaluator} checkpoint hash")
                if evaluator in identities and identities[evaluator] != fingerprint:
                    raise ValueError(f"Different {evaluator} fingerprints")
                identities[evaluator] = fingerprint
        else:
            if payload.get("batch_size") != 16:
                raise ValueError("Efficiency must use common batch size 16")
            for key in ("inference_time_batch_s", "inference_time_batch_std_s", "inference_time_per_traj_s"):
                value = payload.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid efficiency value {key}")
    for path in paths:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing/empty output {path}")
    return {str(path): sha256(path) for path in paths}


def main(argv=None):
    args = parse_args(argv)
    stages = build_stages(args)
    if args.dry_run:
        for stage in stages:
            print(stage["name"] + ":\n" + shlex.join(stage["command"]))
        return 0
    manifest, environment = preflight(args, stages)
    stages = manifest["stages"]
    manifest_path, status_path = args.output_dir / "run_manifest.json", args.output_dir / "status.json"
    if args.resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume refused: manifest differs (inputs, source, dataset, or commands); use a new output directory")
        status = json.loads(status_path.read_text())
        previous_pid = status.get("pid")
        if status.get("status") == "running" and previous_pid:
            try:
                os.kill(previous_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError(f"Previous runner PID {previous_pid} is still alive")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Output directory must be empty; use --resume only for identical runs")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        status = {"created_at": now(), "stages": {stage["name"]: {"status": "pending"} for stage in stages}}
    for directory in ("logs", "results", "tables"):
        (args.output_dir / directory).mkdir(exist_ok=True)
    identities = {}
    for stage in stages:
        record = status["stages"][stage["name"]]
        if record["status"] == "complete":
            if validate_output(args, stage, identities) != record.get("output_sha256"):
                raise ValueError(f"Resume refused: outputs changed for {stage['name']}")
    status.update(status="running", pid=os.getpid(), updated_at=now())
    write_json(status_path, status)
    for stage in stages:
        record = status["stages"][stage["name"]]
        if record["status"] == "complete":
            continue
        log_path = args.output_dir / "logs" / (stage["name"] + ".log")
        record.update(status="running", started_at=now(), log=str(log_path))
        status.update(current_stage=stage["name"], updated_at=now())
        write_json(status_path, status)
        print(f"Starting {stage['name']}; log: {log_path}", flush=True)
        start = time.monotonic()
        try:
            with log_path.open("a") as log:
                log.write(f"\n[{now()}] {shlex.join(stage['command'])}\n")
                log.flush()
                child = subprocess.Popen(stage["command"], cwd=args.project_dir, env=environment, stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = child.pid
                write_json(status_path, status)
                returncode = child.wait()
            record["returncode"] = returncode
            if returncode:
                raise RuntimeError(f"Stage exited with code {returncode}")
            record["output_sha256"] = validate_output(args, stage, identities)
            record["status"] = "complete"
        except (Exception, KeyboardInterrupt) as error:
            if 'child' in locals() and child.poll() is None:
                child.terminate()
                child.wait()
            record.update(status="failed", error=str(error))
            status["status"] = "failed"
        record.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - start, 3))
        status.update(updated_at=now(), evaluator_fingerprints=identities)
        write_json(status_path, status)
        if record["status"] == "failed":
            print(f"Failed {stage['name']}: {record['error']}; see {log_path}", file=sys.stderr)
            return 1
    ablation = args.output_dir / "tables/table3_ablation.md"
    note = "\nArchitecture ablations unavailable: separately trained ablation checkpoints were not supplied. Conditioning-mode and baseline-normalization comparisons are reported separately.\n"
    if note not in ablation.read_text():
        ablation.write_text(ablation.read_text() + note)
    status["stages"]["aggregate"]["output_sha256"] = validate_output(args, stages[-1], identities)
    status.update(status="complete", current_stage=None, updated_at=now())
    write_json(status_path, status)
    print(f"Complete: {args.output_dir / 'tables'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"Paper evaluation preflight failed: {error}", file=sys.stderr)
        sys.exit(2)
