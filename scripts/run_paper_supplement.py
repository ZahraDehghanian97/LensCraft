#!/usr/bin/env python3
"""Run missing E.T. variants, random-K conditioning, and equal-shape timing.

Use the original run's runtime.env and LensCraftVenv. This runner extends a
frozen source snapshot using only this script and testing/paper_protocol.py.
Existing result files and checkpoints are never overwritten.
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
import statistics
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def overrides_with(original, replacements):
    result = [value for value in original if value.split("=", 1)[0].lstrip("+") not in replacements]
    result.extend(value for value in replacements.values() if value is not None)
    return result


def override(key, value):
    return f"{key}={json.dumps(str(value))}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-run-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eff-batch-size", type=int, default=1)
    parser.add_argument("--eff-frames", type=int, default=300)
    parser.add_argument("--eff-warmup", type=int, default=3)
    parser.add_argument("--eff-runs", type=int, default=10)
    parser.add_argument("--sections", default="random_k,et,efficiency")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("random_k", "efficiency"))
    parser.add_argument("--worker-tag")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    for name in ("project_dir", "run_dir", "base_run_dir"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if min(args.batch_size, args.eff_batch_size, args.eff_frames) < 1 or args.num_workers < 0:
        parser.error("Batch sizes/frame count must be positive and workers nonnegative")
    if args.eff_warmup < 0 or args.eff_runs < 2:
        parser.error("Efficiency requires nonnegative warmup and at least two runs")
    if not args.worker and args.base_run_dir is None:
        parser.error("--base-run-dir is required for the runner")
    return args


def base_overrides(base_manifest, model, cohort):
    stage = next(stage for stage in base_manifest["stages"]
                 if stage.get("kind") == "evaluation" and stage.get("model") == model and stage.get("set") == cohort)
    return stage["command"][2:]


def build_stages(args, base_manifest):
    sections = set(args.sections.split(","))
    if sections - {"random_k", "et", "efficiency"}:
        raise ValueError("Unknown supplement section")
    stages = []
    script = str(args.project_dir / "scripts/run_paper_supplement.py")

    def worker_command(kind, name):
        # Random-K workers do not use efficiency settings. Retain their original
        # command identity when amending only the efficiency length protocol.
        size, frames, warmup, runs = ((args.eff_batch_size, args.eff_frames, args.eff_warmup, args.eff_runs)
                                      if kind == "efficiency" else (1, 30, 3, 10))
        return [sys.executable, script, "--worker", kind, "--worker-tag", name,
                "--project-dir", str(args.project_dir), "--run-dir", str(args.run_dir),
                "--eff-batch-size", str(size), "--eff-frames", str(frames),
                "--eff-warmup", str(warmup), "--eff-runs", str(runs), "--"]

    def common(name, batch_size):
        return {"output_dir": override("output_dir", args.run_dir / "results"),
                "hydra.run.dir": override("hydra.run.dir", args.run_dir / "hydra" / name),
                "device": override("device", args.device),
                "data.batch_size": f"data.batch_size={batch_size}",
                "data.num_workers": f"data.num_workers={args.num_workers}",
                "trajectory_cache_dir": override("trajectory_cache_dir", args.run_dir / "trajectory_cache")}

    if "random_k" in sections:
        for cohort in ("static", "dynamic"):
            name = f"random_k_{cohort}"
            replacements = common(name, args.batch_size)
            replacements.update(variant="+variant=paper_random_k", trajectory_cache="trajectory_cache=false")
            command = worker_command("random_k", name) + overrides_with(base_overrides(base_manifest, "lenscraft", cohort), replacements)
            stages.append({"name": name, "kind": "random_k", "cohort": cohort,
                           "command": command, "output": f"metrics_lens_craft_paper_random_k_{cohort}.json"})
    if "et" in sections:
        for et_type in ("adaln", "incontext"):
            for cohort in ("static", "dynamic"):
                name = f"et_{et_type}_{cohort}"
                replacements = common(name, args.batch_size)
                replacements["training.model.inference.et_type"] = f"training.model.inference.et_type={et_type}"
                command = [sys.executable, str(args.project_dir / "src/test.py")] + overrides_with(base_overrides(base_manifest, "et", cohort), replacements)
                stages.append({"name": name, "kind": "et", "cohort": cohort,
                               "et_type": et_type, "command": command, "output": f"metrics_et_{et_type}_{cohort}.json"})
    if "efficiency" in sections:
        for model, et_type in (("lenscraft", None), ("ccdm", None), ("et", "ca"), ("et", "adaln"), ("et", "incontext"), ("gendop", None)):
            tag = ("lens_craft" if model == "lenscraft" else model) + ("_" + et_type if et_type else "")
            name = f"efficiency_{tag}"
            replacements = common(name, args.eff_batch_size)
            replacements.update({"data.num_workers": "data.num_workers=0", "eval_set": None,
                                 "test_movement_types": "test_movement_types=null",
                                 "training.model.data_format.seq_length": f"training.model.data_format.seq_length={args.eff_frames}"})
            if et_type:
                replacements["training.model.inference.et_type"] = f"training.model.inference.et_type={et_type}"
            if model == "ccdm":
                replacements["training.model.inference.seq_len"] = f"+training.model.inference.seq_len={args.eff_frames}"
            command = worker_command("efficiency", tag) + overrides_with(base_overrides(base_manifest, model, "static"), replacements)
            stages.append({"name": name, "kind": "efficiency", "command": command,
                           "output": f"efficiency_paper_{tag}.json"})
    return stages


def _timing_inputs(cfg, args):
    import torch
    from data.datamodule import CameraTrajectoryDataModule
    from data.dataset_type import resolve_dataset_type
    from data.convertor.convertor import convert_to_target
    from data.sim_format import to_simulation_format
    from data.simulation.utils import structured_conditioning_from_batch
    from models.factory import load_model, model_type_from_cfg
    from testing.paper_protocol import configure_native_length
    from utils.device import move_batch_to_device

    device = torch.device(cfg.device)
    model_type = model_type_from_cfg(cfg)
    dm = CameraTrajectoryDataModule(dataset_config=cfg.data.dataset.config,
        batch_size=args.eff_batch_size, num_workers=0, val_size=cfg.data.val_size,
        test_size=cfg.data.test_size, test_fraction=cfg.test_fraction)
    dm.setup()
    batch = move_batch_to_device(next(iter(dm.test_dataloader())), device)
    if len(batch["text_prompts"]) != args.eff_batch_size:
        raise ValueError("Insufficient held-out inputs for requested efficiency batch")
    dataset_type = resolve_dataset_type(cfg.data.dataset.config["_target_"])
    model = load_model(cfg, model_type, device)
    length_protocol = configure_native_length(model, model_type, args.eff_frames)
    limitations = ["Text conditioning extraction and input format conversion are excluded from timing and FLOPs.",
                  "Adapter-native output decoding/postprocessing is included; model loading and metric evaluation are excluded.",
                  "Native output representations differ across models (poses/features/matrices).",
                  "Measurements share server GPUs with other jobs and may include resource contention."]
    if model_type == "lens_craft":
        _, subject, volume, padding = to_simulation_format(batch, dataset_type, target_len=args.eff_frames)
        caption = structured_conditioning_from_batch(batch)
        def generate():
            return model.generate_camera_trajectory(subject_trajectory=subject, subject_volume=volume,
                camera_trajectory=None, padding_mask=padding, memory_teacher_forcing_ratio=1.0,
                caption_embedding=caption)["reconstructed"]
        if length_protocol["extrapolates_checkpoint_native_length"]:
            limitations.append(f"LensCraft generates {args.eff_frames} frames with unchanged weights trained at {length_protocol['checkpoint_native_frames']} frames; this is a length-extrapolation timing benchmark, not a quality evaluation at the trained length.")
        return generate, model, limitations, length_protocol

    trajectory, subject, _, padding = convert_to_target(dataset_type, model_type,
        batch["camera_trajectory"], batch["subject_trajectory"], batch["subject_volume"],
        batch["padding_mask"], args.eff_frames)
    prompts = batch["text_prompts"]
    if model_type == "ccdm":
        rewritten = list(prompts)
        for old, new in model._PROMPT_REWRITES:
            rewritten = [prompt.replace(old, new) for prompt in rewritten]
        embedding = model.clip_embedder.extract_clip_embeddings(rewritten, return_seq=False)
        model.clip_embedder.extract_clip_embeddings = lambda *a, **kw: embedding
        limitations.append(f"CCDM generates {model.seq_len} frames using {model.n_T} diffusion steps; checkpoint was trained at native length 300.")
    elif model_type == "et":
        features = model._generate_caption_feat(prompts)
        model._generate_caption_feat = lambda *a, **kw: features
        if length_protocol["extrapolates_checkpoint_native_length"]:
            limitations.append(f"E.T. CA latent-length metadata changed to {args.eff_frames} without changing weights; released checkpoint was trained at 300 frames.")
    elif model_type == "gendop":
        cached = {prompt: model.model.encode_cond([prompt]) for prompt in prompts}
        model.model.encode_cond = lambda texts: cached[texts[0]]
        limitations.append("GenDoP text encoder plus learned text-conditioning projection are precomputed; native generation supports only batch 1.")
        limitations.append(f"GenDoP fixed-length decoding sets min_new_tokens=max_new_tokens={args.eff_frames * 10} in its autoregressive sampler. This explicitly changes its EOS stopping rule; all coordinate tokens are actually generated, with no adapter padding or trimming.")
        if length_protocol["extrapolates_checkpoint_native_length"]:
            limitations.append(f"GenDoP requests {args.eff_frames} frames via native pose-length/token-count metadata with unchanged 30-frame-trained weights. Every call must emit exactly {args.eff_frames * 10} raw coordinate tokens before adapter postprocessing.")
    def generate():
        return model.generate_using_text(prompts, subject, trajectory, padding)
    return generate, model, limitations, length_protocol


def efficiency_worker(cfg, args):
    import torch
    from models.factory import model_type_from_cfg
    from testing.paper_protocol import (equal_length_support, validate_native_shape,
                                        profile_generation_flops, UnsupportedGenerationLength)
    model_type = model_type_from_cfg(cfg)
    et_type = cfg.training.model.inference.get("et_type") if model_type == "et" else None
    supported, reason = equal_length_support(model_type, et_type, args.eff_frames, args.eff_batch_size)
    payload = {"model_type": model_type, "et_type": et_type,
               "status": "unsupported" if not supported else "pending", "reason": reason,
               "batch_size": args.eff_batch_size, "seq_length": args.eff_frames,
               "protocol": {"same_length_same_batch": supported, "seq_length": args.eff_frames,
                            "batch_size": args.eff_batch_size, "text_preprocessing": "excluded",
                            "scope": "actual native adapter generation; conditioning precomputed; no output resampling"},
               "inference_time_batch_s": None, "inference_time_batch_std_s": None,
               "inference_time_per_traj_s": None, "inference_time_per_traj_std_s": None,
               "gflops_per_traj": None, "flop_count_is_lower_bound": True}
    if supported:
        torch.manual_seed(42)
        device = torch.device(cfg.device)
        generate, model, limitations, length_protocol = _timing_inputs(cfg, args)
        payload.update(limitations=limitations, generation_length_protocol=length_protocol,
                       torch_version=torch.__version__, device=str(device),
                       device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        try:
            with torch.no_grad():
                first = generate()
                validate_native_shape(first, args.eff_frames, args.eff_batch_size)
                payload["actual_output_shape"] = list(first.shape)
                for _ in range(args.eff_warmup):
                    generate()
                durations = []
                for _ in range(args.eff_runs):
                    sync()
                    start = time.perf_counter()
                    result = generate()
                    sync()
                    durations.append(time.perf_counter() - start)
                    validate_native_shape(result, args.eff_frames, args.eff_batch_size)
            flops = profile_generation_flops(generate, args.eff_batch_size)
        except UnsupportedGenerationLength as error:
            payload.update(status="unsupported", reason=str(error))
            payload["protocol"]["same_length_same_batch"] = False
        else:
            payload.update(status="complete", n_warmup=args.eff_warmup,
                n_runs=args.eff_runs, inference_time_batch_s=statistics.mean(durations),
                inference_time_batch_std_s=statistics.stdev(durations),
                inference_time_per_traj_s=statistics.mean(durations) / args.eff_batch_size,
                inference_time_per_traj_std_s=statistics.stdev(durations) / args.eff_batch_size,
                timing_samples_s=durations)
            payload.update(flops)
    output = args.run_dir / "results" / f"efficiency_paper_{args.worker_tag}.json"
    write_json(output, payload)
    print(f"Wrote {output}", flush=True)


def worker(args):
    from dotenv import load_dotenv
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    sys.path.insert(0, str(args.project_dir / "src"))
    os.chdir(args.project_dir)
    load_dotenv(args.project_dir / ".env", override=False)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    with initialize_config_dir(version_base=None, config_dir=str(args.project_dir / "config")):
        cfg = compose(config_name="test", overrides=overrides)
    config_path = args.run_dir / "hydra" / str(args.worker_tag) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path, resolve=True)
    if args.worker == "random_k":
        import test as evaluator
        from testing.paper_protocol import install_random_keyframe_protocol
        if cfg.trajectory_cache:
            raise ValueError("Random-K worker requires trajectory_cache=false")
        install_random_keyframe_protocol(evaluator, args.run_dir / "random_k_trajectories" / str(cfg.eval_set))
        evaluator.main.__wrapped__(cfg)
    else:
        efficiency_worker(cfg, args)
    return 0


def validate_result(path, stage, base_manifest, base_run):
    payload = json.loads(path.read_text())
    if stage["kind"] == "efficiency":
        if payload.get("status") not in ("complete", "unsupported"):
            raise ValueError("Efficiency did not complete or explain unsupported shape")
        if payload["status"] == "complete":
            if not payload["protocol"]["same_length_same_batch"]:
                raise ValueError("Efficiency output violates shared length/batch protocol")
            if payload["actual_output_shape"][:2] != [payload["batch_size"], payload["seq_length"]]:
                raise ValueError("Efficiency output shape mismatch")
            if not math.isfinite(payload["inference_time_batch_s"]) or payload["inference_time_batch_s"] <= 0:
                raise ValueError("Invalid efficiency time")
        elif not payload.get("reason"):
            raise ValueError("Unsupported efficiency must explain the limitation")
    else:
        expected = json.loads((base_run / "evaluation/results" / f"metrics_lens_craft_{stage['cohort']}.json").read_text())
        for key in ("test_sample_counts", "test_fraction", "test_movement_types", "set", "clatr_backend"):
            if payload.get(key) != expected.get(key):
                raise ValueError(f"Supplement changed held-out protocol: {key}")
        for name, field in (("semantic_evaluator", "semantic_evaluator_checkpoint"), ("clatr_evaluator", "clatr_checkpoint")):
            observed = payload["evaluation_provenance"][name]
            expected_hash = base_manifest["inputs"][field]["sha256"]
            if expected_hash != observed.get("checkpoint", {}).get("sha256"):
                raise ValueError(f"Supplement changed fixed evaluator: {name}")
        if stage["kind"] == "random_k":
            if set(payload["metrics"]) != {"key_framing_random_1_10", "key_framing+prompt_random_1_10"}:
                raise ValueError("Missing random-K modes")
            if payload["keyframe_protocol"]["sample_count"] != payload["test_sample_counts"]["cohort"]:
                raise ValueError("Random-K sample count mismatch")
        elif payload.get("et_type") != stage["et_type"]:
            raise ValueError("E.T. architecture does not match stage")
        for row in payload["metrics"].values():
            for key, value in row.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"Nonfinite output metric: {key}")
    return sha256(path)


def run(args):
    from dotenv import load_dotenv
    load_dotenv(args.project_dir / ".env", override=False)
    base_path = args.base_run_dir / "evaluation/run_manifest.json"
    base = json.loads(base_path.read_text())
    stages = build_stages(args, base)
    if args.dry_run:
        for stage in stages:
            print(stage["name"] + ":\n" + shlex.join(stage["command"]))
        return 0
    environment = os.environ.copy()
    environment.update({key: value for key, value in base["baseline_environment"].items() if value})
    environment.update(PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
    for relative, expected_hash in base["source_sha256"].items():
        path = args.project_dir / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Frozen base-run source missing or changed: {relative}")
    for entry in base["inputs"].values():
        if not Path(entry["path"]).is_file() or sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Original run input missing or changed: {entry['path']}")
    extra_checkpoints = {}
    if any(stage.get("kind") == "et" for stage in stages):
        checkpoint_dir = Path(environment["DIRECTOR_PROJECT_DIR"]) / "checkpoints/director"
        for et_type in ("adaln", "incontext"):
            path = checkpoint_dir / f"{et_type}-mixed-e449.ckpt"
            if not path.is_file():
                raise ValueError(f"Missing E.T. architecture checkpoint: {path}")
            extra_checkpoints[et_type] = {"path": str(path), "sha256": sha256(path)}
    added = [args.project_dir / "scripts/run_paper_supplement.py", args.project_dir / "src/testing/paper_protocol.py"]
    manifest = {"protocol": "paper-supplement-v1", "base_run_dir": str(args.base_run_dir),
                "base_manifest_sha256": sha256(base_path), "device": args.device,
                "new_source_sha256": {str(path.relative_to(args.project_dir)): sha256(path) for path in added},
                "stages": stages, "extra_checkpoints": extra_checkpoints,
                "created_by_python": sys.executable}
    manifest_path, status_path = args.run_dir / "supplement_manifest.json", args.run_dir / "supplement_status.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume refused: source, inputs, or stage commands changed")
        status = json.loads(status_path.read_text())
        if status.get("status") == "running" and status.get("pid"):
            try:
                os.kill(status["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("Previous supplement runner is still running")
    else:
        if manifest_path.exists() or status_path.exists():
            raise ValueError("Supplement already exists; use --resume for the same protocol")
        write_json(manifest_path, manifest)
        status = {"created_at": now(), "stages": {stage["name"]: {"status": "pending"} for stage in stages}}
    for directory in ("logs", "results", "hydra"):
        (args.run_dir / directory).mkdir(parents=True, exist_ok=True)
    for stage in stages:
        record = status["stages"][stage["name"]]
        if record["status"] == "complete":
            if validate_result(args.run_dir / "results" / stage["output"], stage, base, args.base_run_dir) != record.get("output_sha256"):
                raise ValueError("Resume refused: completed outputs changed")
    status.update(status="running", pid=os.getpid(), updated_at=now())
    write_json(status_path, status)
    for stage in stages:
        record = status["stages"][stage["name"]]
        if record["status"] == "complete":
            continue
        log_path = args.run_dir / "logs" / f"{stage['name']}.log"
        record.update(status="running", started_at=now(), log=str(log_path))
        status.update(current_stage=stage["name"], updated_at=now())
        write_json(status_path, status)
        print(f"Starting {stage['name']}; log: {log_path}", flush=True)
        started = time.monotonic()
        child = None
        try:
            with log_path.open("a") as log:
                log.write(f"\n[{now()}] {shlex.join(stage['command'])}\n")
                log.flush()
                child = subprocess.Popen(stage["command"], cwd=args.project_dir, env=environment,
                                         stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = child.pid
                write_json(status_path, status)
                returncode = child.wait()
                if returncode:
                    raise RuntimeError(f"Stage exited with code {returncode}")
            record.update(status="complete", output_sha256=validate_result(
                args.run_dir / "results" / stage["output"], stage, base, args.base_run_dir))
        except (Exception, KeyboardInterrupt) as error:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            record.update(status="failed", error=str(error))
            status["status"] = "failed"
        record.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        status["updated_at"] = now()
        write_json(status_path, status)
        if record["status"] == "failed":
            print(f"Failed {stage['name']}: {record['error']}; see {log_path}", file=sys.stderr)
            return 1
    status.update(status="complete", current_stage=None, updated_at=now())
    write_json(status_path, status)
    return 0


if __name__ == "__main__":
    arguments = parse_args()
    sys.exit(worker(arguments) if arguments.worker else run(arguments))
