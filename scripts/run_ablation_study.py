#!/usr/bin/env python3
"""Train one scheduled control and eight ablations with fixed external scorers.

This is a component study of commit 96a9cd7's corrected loss/rotation recipe,
not reproduction of the PDF's original loss weights or raw-rotation training.
Source a curated runtime.env before invoking; no shell environment is evaluated
by this script. --resume requires identical source, inputs, and configuration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

VARIANTS = {
    "full_scheduled": {},
    "without_clip": {"training.loss_module.losses_list.clip": 0},
    "without_first_frame": {"training.loss_module.losses_list.first_frame": 0},
    "without_relative": {"training.loss_module.losses_list.relative": 0},
    "without_speed": {"training.loss_module.losses_list.speed": 0},
    "without_cycle": {"training.loss_module.losses_list.cycle": 0,
                      "training.use_cycle_consistency": False},
    "without_teacher": {"training.teacher_forcing_schedule.memory_initial_ratio": 1.0,
                        "training.teacher_forcing_schedule.memory_final_ratio": 1.0},
    "without_volume": {"training.model.module.use_subject_volume": False},
    "without_noise": {"training.noise.initial_std": 0.0, "training.noise.final_std": 0.0},
}
COHORTS = {"static": ["static"], "dynamic": ["circular", "zigzag", "linear", "spiral",
    "figureEight", "wave", "pendulum", "orbital", "bounce"]}
METRICS = ("clatr_score", "fcd", "precision", "recall", "density", "coverage", "clip_score")
EPOCHS, BATCH_SIZE, SEED, TEST_FRACTION = 100, 128, 42, 0.1
ENV_KEYS = ("CLIP_EMBEDDINGS_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TORCH_HOME",
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def override(key, value):
    if isinstance(value, Path):
        value = str(value)
    return key + "=" + (str(value).lower() if isinstance(value, bool) else json.dumps(value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-dir", "run-dir", "dataset-path", "split-manifest",
                 "semantic-checkpoint", "semantic-config", "clatr-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--source-revision", default="96a9cd7", help="Declared frozen base revision; all source files are also hashed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing or validating assets")
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.num_workers < 0 or not (args.device == "cpu" or
            args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("workers must be nonnegative; device must be cpu or cuda:N")
    return args


def train_overrides(args, variant, resume=None):
    train_dir = args.run_dir / variant / "train"
    values = {
        "seed": SEED, "trainer.max_epochs": EPOCHS, "patience": EPOCHS,
        "data.batch_size": BATCH_SIZE, "data.num_workers": args.num_workers,
        "data.val_size": 0.2, "data.test_size": 0.2,
        "data.dataset.config.data_path": args.dataset_path,
        "+data.dataset.config.allowed_movement_types": [],
        "training.conditioning.enabled": False,
        "training.validation_conditioning.enabled": True,
        "training.validation_conditioning.max_batches": 4,
        "checkpoint_monitor": "val_conditioning_score",
        "training.model.module._target_": "models.ablation_model.AblationLensCraft",
        "+training.model.module.use_subject_volume": True,
        "initialize_from_checkpoint": None, "resume_checkpoint": resume,
        "trainer.accelerator": "cpu" if args.device == "cpu" else "gpu",
        "trainer.devices": 1 if args.device == "cpu" else [int(args.device[5:])],
        "+callbacks.val_checkpoint.save_last": True,
        "+callbacks.val_checkpoint.dirpath": train_dir / "lightning_logs/version_0/checkpoints",
        "+callbacks.train_checkpoint.dirpath": train_dir / "lightning_logs/version_0/checkpoints",
        "+trainer.logger._target_": "lightning.pytorch.loggers.CSVLogger",
        "+trainer.logger.save_dir": train_dir, "+trainer.logger.name": "lightning_logs",
        "+trainer.logger.version": 0, "+trainer.enable_progress_bar": False,
        "hydra.run.dir": train_dir,
    }
    for key, value in VARIANTS[variant].items():
        values[("+" + key) if "+" + key in values else key] = value
    return [override(key, value) for key, value in values.items()]


def training_command(args, variant, resume=None):
    return [sys.executable, str(args.project_dir / "src/train.py"),
            *train_overrides(args, variant, resume)]


def evaluation_command(args, variant, cohort, checkpoint):
    values = {
        "data.dataset.config.data_path": args.dataset_path,
        "+data.dataset.config.allowed_movement_types": [], "data.val_size": 0.2,
        "data.test_size": 0.2, "data.batch_size": BATCH_SIZE,
        "data.num_workers": args.num_workers, "+seed": SEED,
        "training.model.inference.checkpoint_path": checkpoint,
        "training.model.inference.config": args.run_dir / variant / "train/.hydra/config.yaml",
        "ref_model.inference.checkpoint_path": args.semantic_checkpoint,
        "ref_model.inference.config": args.semantic_config,
        "clatr_backend": "native", "clatr_native_checkpoint_path": args.clatr_checkpoint,
        "test_fraction": TEST_FRACTION, "test_movement_types": COHORTS[cohort],
        "+variant": variant, "+eval_set": cohort, "+n_boot": 500,
        "keyframes.counts": [], "caption_top1_metric": False, "tsne": False,
        "device": args.device, "output_dir": args.run_dir / "results",
        "trajectory_cache_dir": args.run_dir / variant / "trajectory_cache",
        "hydra.run.dir": args.run_dir / variant / "evaluation" / cohort,
    }
    return [sys.executable, str(args.project_dir / "src/test.py"),
            *[override(key, value) for key, value in values.items()]]


def _get(config, dotted):
    for part in dotted.split("."):
        config = config[part]
    return config


def validate_recipe(config, variant):
    """Reject changed defaults and silent no-op Teacher/Noise/Volume studies."""
    expected = {
        "training.conditioning.enabled": False, "training.validation_conditioning.enabled": True,
        "training.use_merged_memory": False, "training.decode_mode": "single_step",
        "training.model.module._target_": "models.ablation_model.AblationLensCraft",
        "training.model.module.use_subject_volume": True,
        "training.model.module.keyframe_pose_conditioning": False,
        "training.loss_module.losses_list.clip": 1000, "training.loss_module.losses_list.first_frame": 2,
        "training.loss_module.losses_list.relative": 16, "training.loss_module.losses_list.speed": 80,
        "training.loss_module.losses_list.cycle": 100, "training.use_cycle_consistency": True,
        "training.loss_module.losses_list.rotation_absolute": 2,
        "training.loss_module.losses_list.rotation_raw": 1,
        "training.noise.initial_std": 1.0, "training.noise.final_std": 0.0,
        "training.mask.initial_ratio": 0.1, "training.mask.final_ratio": 0.8,
        "training.teacher_forcing_schedule.memory_initial_ratio": 0.7,
        "training.teacher_forcing_schedule.memory_final_ratio": 1.0,
        "trainer.max_epochs": EPOCHS, "patience": EPOCHS, "seed": SEED,
        "checkpoint_monitor": "val_conditioning_score", "data.batch_size": BATCH_SIZE,
    }
    expected.update(VARIANTS[variant])
    for key, value in expected.items():
        if _get(config, key) != value:
            raise ValueError(f"{variant}: incompatible/no-op study setting {key}: expected {value!r}")


def normalized_config(config):
    # Hydra rewrites this one field during legitimate full-state continuation.
    result = dict(config)
    result["resume_checkpoint"] = None
    return result


def preflight(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import torch
    sys.path.insert(0, str(args.project_dir / "src"))
    from data.simulation.metadata import resolve_dataset_root
    args.dataset_path = resolve_dataset_root(args.dataset_path)
    environment = os.environ.copy()
    environment.update(SIMULATION_DATA_PATH=str(args.dataset_path), PYTHONUNBUFFERED="1",
        HYDRA_FULL_ERROR="1", TEST_CHECKPOINT_PATH=str(args.semantic_checkpoint),
        TEST_CONFIG_PATH=str(args.semantic_config),
        SEMANTIC_EVALUATOR_CHECKPOINT_PATH=str(args.semantic_checkpoint),
        SEMANTIC_EVALUATOR_CONFIG_PATH=str(args.semantic_config),
        CLATR_NATIVE_CHECKPOINT_PATH=str(args.clatr_checkpoint))
    # Hydra composition uses environment resolvers; these explicit paths also
    # prevent stale TEST_* shell variables from leaking into saved training YAML.
    os.environ.update({key: value for key, value in environment.items() if key in (
        "SIMULATION_DATA_PATH", "TEST_CHECKPOINT_PATH", "TEST_CONFIG_PATH",
        "SEMANTIC_EVALUATOR_CHECKPOINT_PATH", "SEMANTIC_EVALUATOR_CONFIG_PATH",
        "CLATR_NATIVE_CHECKPOINT_PATH")})
    paths = {name: getattr(args, name) for name in (
        "semantic_checkpoint", "semantic_config", "clatr_checkpoint", "split_manifest")}
    paths["parameter_dictionary"] = args.dataset_path / "parameter_dictionary.msgpack"
    identities = {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()}
    files = sorted(args.dataset_path.glob("simulation_*.msgpack"))
    if not files:
        raise ValueError("No simulation samples")
    indices = torch.randperm(len(files), generator=torch.Generator().manual_seed(SEED)).tolist()
    train_n, val_n = int((1 - .2 - .2) * len(files)), int(.2 * len(files))
    splits = {"train": indices[:train_n], "validation": indices[train_n:train_n + val_n],
              "test": indices[train_n + val_n:]}
    saved = json.loads(args.split_manifest.read_text())
    if any(saved.get(key) != value for key, value in splits.items()):
        raise ValueError("Split manifest differs from seed-42 60/20/20 training inventory")
    if args.device != "cpu" and (not torch.cuda.is_available() or
            int(args.device[5:]) >= torch.cuda.device_count()):
        raise ValueError(f"Requested device unavailable: {args.device}")
    configurations = {}
    for variant in VARIANTS:
        with initialize_config_dir(version_base=None, config_dir=str(args.project_dir / "config")):
            cfg = compose(config_name="config", overrides=train_overrides(args, variant))
        config = OmegaConf.to_container(cfg, resolve=True)
        validate_recipe(config, variant)
        configurations[variant] = normalized_config(config)
    sources = {str(path.relative_to(args.project_dir)): digest(path)
        for folder in ("src", "config", "scripts", "third_parties")
        for path in sorted((args.project_dir / folder).rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml")}
    inventory = [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    manifest = {"protocol": 1, "declared_base_revision": args.source_revision,
        "description": "Scheduled-policy one-factor study of corrected 96a9cd7 weights/rotation losses; NOT exact paper-method reproduction. Teacher removal fixes text-memory ratio at one. Volume removal retains a constant token. Keyframe losses have no known frames under this scheduled policy.",
        "variants": VARIANTS, "source_sha256": sources, "inputs": identities,
        "configurations": configurations, "environment": {key: environment.get(key) for key in ENV_KEYS},
        "python": sys.executable, "torch": torch.__version__, "device": args.device,
        "dataset": {"path": str(args.dataset_path), "sample_count": len(files),
            "inventory_sha256": hashlib.sha256(json.dumps(inventory).encode()).hexdigest(),
            "train_count": train_n, "validation_count": val_n, "test_count": len(splits["test"]),
            "fractional_test_count": math.ceil(len(splits["test"]) * TEST_FRACTION),
            "split_seed": SEED, "test_fraction": TEST_FRACTION},
        "evaluation": {"cohorts": COHORTS, "n_boot": 500,
            "modes": ["prompt_generation", "reconstruction", "hybrid_generation"]}}
    return manifest, environment


def verify_artifacts(records):
    for name, expected in records.items():
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f"Recorded output changed or missing: {name}")


def saved_config(args, variant, expected):
    from omegaconf import OmegaConf
    path = args.run_dir / variant / "train/.hydra/config.yaml"
    actual = normalized_config(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    if actual != expected:
        raise ValueError(f"Saved training configuration changed: {variant}")
    return path


def preserve_resume_metrics(args, variant, checkpoint):
    """CSVLogger can replace metrics.csv when reusing version 0 on resume."""
    if checkpoint is None:
        return None
    train = args.run_dir / variant / "train"
    metrics = train / "lightning_logs/version_0/metrics.csv"
    record = {"resumed_at": now(), "checkpoint": str(checkpoint)}
    if metrics.is_file():
        directory = train / "continuations"
        directory.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        destination = directory / f"metrics_before_resume_{stamp}.csv"
        if destination.exists():
            raise ValueError(f"Refusing to overwrite continuation metrics: {destination}")
        shutil.copy2(metrics, destination)
        record.update(metrics_path=str(destination), metrics_sha256=digest(destination))
    else:
        record["metrics_status"] = "No previous metrics.csv was present"
    return record


def validate_training(args, variant, manifest):
    import torch
    directory = args.run_dir / variant / "train/lightning_logs/version_0/checkpoints"
    last = directory / "last.ckpt"
    state = torch.load(last, map_location="cpu", weights_only=False)
    expected_steps = EPOCHS * math.ceil(manifest["dataset"]["train_count"] / BATCH_SIZE)
    if state.get("epoch") != EPOCHS - 1 or state.get("global_step") != expected_steps:
        raise ValueError(f"Incomplete 100-epoch training: {variant}, epoch={state.get('epoch')}, step={state.get('global_step')}")
    optim_steps = [float(value["step"]) for optimizer in state.get("optimizer_states", [])
        for value in optimizer.get("state", {}).values() if "step" in value]
    if not optim_steps or not all(math.isfinite(value) for value in optim_steps) or max(optim_steps) <= 0:
        raise ValueError("No finite optimizer updates in final checkpoint")
    callbacks = [value for value in state.get("callbacks", {}).values()
        if isinstance(value, dict) and value.get("monitor") == "val_conditioning_score"]
    if len(callbacks) != 1:
        raise ValueError("Expected one fixed-score checkpoint callback")
    callback = callbacks[0]
    best = Path(callback["best_model_path"]).resolve()
    if best.parent != directory.resolve() or not best.name.startswith("best-val-model-"):
        raise ValueError("Best checkpoint points outside this variant's own run")
    score = float(callback["best_model_score"])
    if not math.isfinite(score):
        raise ValueError("Non-finite best conditioning score")
    config = saved_config(args, variant, manifest["configurations"][variant])
    return {"checkpoint": str(best), "score": score, "global_step": expected_steps,
        "optimizer_max_step": max(optim_steps), "output_sha256": {
            str(path): digest(path) for path in (best, last, config)}}


def validate_evaluation(args, variant, cohort, manifest, identities):
    path = args.run_dir / "results" / f"metrics_lens_craft_{variant}_{cohort}.json"
    value = json.loads(path.read_text())
    if (value.get("variant") != variant or value.get("set") != cohort or
            value.get("model_type") != "lens_craft" or value.get("clatr_backend") != "native" or
            value.get("test_fraction") != TEST_FRACTION or
            value.get("test_movement_types") != COHORTS[cohort]):
        raise ValueError(f"Wrong ablation/cohort/protocol: {path}")
    counts = value.get("test_sample_counts", {})
    if (counts.get("original_holdout") != manifest["dataset"]["test_count"] or
            counts.get("fractional_holdout") != manifest["dataset"]["fractional_test_count"] or
            not 0 < counts.get("cohort", 0) <= counts["fractional_holdout"]):
        raise ValueError("Evaluation holdout counts differ")
    if set(value.get("metrics", {})) != set(manifest["evaluation"]["modes"]):
        raise ValueError("Missing/unexpected conditioning modes")
    for mode, metrics in value["metrics"].items():
        for name in METRICS:
            key = f"{mode}/{name}"
            if not isinstance(metrics.get(key), (float, int)) or not math.isfinite(metrics[key]):
                raise ValueError(f"Missing/nonfinite metric: {key}")
            bootstrap = value.get("bootstrap_std", {}).get(mode, {}).get(key)
            if not isinstance(bootstrap, list) or len(bootstrap) != 2 or not all(math.isfinite(x) for x in bootstrap):
                raise ValueError(f"Missing/nonfinite bootstrap: {key}")
    for name, field in (("semantic_evaluator", "semantic_checkpoint"), ("clatr_evaluator", "clatr_checkpoint")):
        identity = value.get("evaluation_provenance", {}).get(name, {})
        if identity.get("checkpoint", {}).get("sha256") != manifest["inputs"][field]["sha256"]:
            raise ValueError(f"Changed fixed evaluator: {name}")
        fingerprint = identity.get("fingerprint")
        if not fingerprint or name in identities and identities[name] != fingerprint:
            raise ValueError(f"Inconsistent evaluator fingerprint: {name}")
        identities[name] = fingerprint
    return {str(path): digest(path)}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        import shlex
        for variant in VARIANTS:
            print(shlex.join(training_command(args, variant)))
            for cohort in COHORTS:
                print(shlex.join(evaluation_command(args, variant, cohort, "SELECTED_BEST_CHECKPOINT")))
        return 0
    manifest, environment = preflight(args)
    manifest_path, status_path = args.run_dir / "run_manifest.json", args.run_dir / "status.json"
    if args.resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume refused: source, configuration, environment or inputs differ")
        status = json.loads(status_path.read_text())
        for pid in (status.get("pid"), status.get("child_pid")):
            if pid and alive(pid):
                raise ValueError(f"Previous runner/child PID {pid} is still alive")
    else:
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise ValueError("New study requires an empty run directory; use --resume for recovery")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        status = {"created_at": now(), "variants": {variant: {
            stage: {"status": "pending"} for stage in ("training", *COHORTS)
        } for variant in VARIANTS}}
    identities = {}
    for variant, stages in status["variants"].items():
        for stage, record in stages.items():
            if record["status"] == "complete":
                verify_artifacts(record["output_sha256"])
                if stage != "training":
                    validate_evaluation(args, variant, stage, manifest, identities)
    (args.run_dir / "results").mkdir(exist_ok=True)
    (args.run_dir / "logs").mkdir(exist_ok=True)
    status.update(status="running", pid=os.getpid(), child_pid=None, evaluator_fingerprints=identities)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    current_record = None
    child = None
    try:
        for variant in VARIANTS:
            stages = status["variants"][variant]
            for stage in ("training", *COHORTS):
                current_record = stages[stage]
                if current_record["status"] == "complete":
                    continue
                if stage == "training":
                    last = args.run_dir / variant / "train/lightning_logs/version_0/checkpoints/last.ckpt"
                    if last.exists():
                        saved_config(args, variant, manifest["configurations"][variant])
                        # A crash after the last optimizer epoch can be finalized
                        # without invoking Lightning again on a completed loop.
                        try:
                            result = validate_training(args, variant, manifest)
                        except ValueError:
                            result = None
                        if result is not None:
                            current_record.update(status="complete", finished_at=now(), **result)
                            write_json(status_path, status)
                            continue
                    resume_checkpoint = last if last.exists() else None
                    command = training_command(args, variant, resume_checkpoint)
                    continuation = preserve_resume_metrics(args, variant, resume_checkpoint)
                    if continuation is not None:
                        current_record.setdefault("continuations", []).append(continuation)
                else:
                    command = evaluation_command(args, variant, stage, stages["training"]["checkpoint"])
                log_path = args.run_dir / "logs" / f"{variant}_{stage}.log"
                current_record.update(status="running", started_at=now(), command=command, log=str(log_path))
                status.update(current_variant=variant, current_stage=stage, updated_at=now())
                write_json(status_path, status)
                print(f"Starting {variant}/{stage}; log {log_path}", flush=True)
                started = time.monotonic()
                with log_path.open("a") as log:
                    log.write(f"\n[{now()}] {json.dumps(command)}\n")
                    log.flush()
                    child = subprocess.Popen(command, cwd=args.project_dir, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    status["child_pid"] = child.pid
                    write_json(status_path, status)
                    returncode = child.wait()
                status["child_pid"] = None
                if returncode:
                    raise RuntimeError(f"{variant}/{stage} exited with {returncode}")
                result = (validate_training(args, variant, manifest) if stage == "training" else
                    {"output_sha256": validate_evaluation(args, variant, stage, manifest, identities)})
                current_record.update(status="complete", finished_at=now(),
                    elapsed_seconds=round(time.monotonic() - started, 3), **result)
                status.update(updated_at=now(), evaluator_fingerprints=identities)
                write_json(status_path, status)
        status.update(status="complete", current_variant=None, current_stage=None, pid=None)
        write_json(status_path, status)
        print(f"Complete: {args.run_dir / 'results'}", flush=True)
        return 0
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if current_record is not None:
            current_record.update(status="failed", error=str(error), finished_at=now())
        status.update(status="failed", error=str(error), updated_at=now(), pid=None, child_pid=None)
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, KeyboardInterrupt) as error:
        print(f"Ablation study stopped: {error}", file=sys.stderr)
        sys.exit(130 if isinstance(error, KeyboardInterrupt) else 1)
