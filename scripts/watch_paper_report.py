#!/usr/bin/env python3
"""Refresh the paper report as measurements arrive, until automatic work ends.

Human-study completion is intentionally excluded from automatic finality. The
watcher never loads model checkpoints. CSV changes trigger progress refreshes.
The adjacent build_paper_report.py remains responsible for report validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BUILDER = Path(__file__).with_name("build_paper_report.py")
VARIANTS = ("full_scheduled", "without_clip", "without_first_frame", "without_relative",
            "without_speed", "without_cycle", "without_teacher", "without_volume", "without_noise")
JSON_NAMES = {"status.json", "supplement_status.json", "run_manifest.json", "supplement_manifest.json",
              "manifest.json", "qualitative_manifest.json", "training_summary.json", "split_indices.json"}
ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".pdf"}
SKIP_DIRS = {".git", "__pycache__", "checkpoints", "trajectory_cache", "random_k_trajectories",
             "source", "code", "server_backup", "test_dependencies"}


class InputChanged(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-pdf", type=Path)
    parser.add_argument("--extra-results-dir", type=Path, action="append", default=[])
    parser.add_argument("--ablation-dir", type=Path)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--once", action="store_true", help="Perform one refresh attempt, without waiting")
    args = parser.parse_args(argv)
    if not 0 < args.interval < float("inf"):
        parser.error("--interval must be finite and positive")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    args.extra_results_dir = [path.expanduser().resolve() for path in args.extra_results_dir]
    if args.output_dir in input_roots(args):
        parser.error("Report output must be separate from input roots")
    return args


def input_roots(args):
    roots = [args.base_run_dir, *args.extra_results_dir]
    if args.ablation_dir is not None:
        roots.append(args.ablation_dir)
    return list(dict.fromkeys(path.resolve() for path in roots))


def watcher_paths(args):
    parent, name = args.output_dir.parent, args.output_dir.name
    return {"status": parent / f"{name}_watch_status.json",
            "lock": parent / f"{name}.watch.lock", "log": parent / f"{name}_watch.log"}


def excluded(path, args):
    path = path.resolve()
    return path == args.output_dir or args.output_dir in path.parents or path in {
        *watcher_paths(args).values(), Path(__file__).resolve()}


def relevant(path):
    return (path.name in JSON_NAMES or
            path.suffix == ".json" and path.name.startswith(("metrics_", "efficiency")) or
            path.suffix.lower() in ASSET_SUFFIXES | {".yaml", ".yml", ".csv"})


def discover_inputs(args):
    files = set()
    for root in input_roots(args):
        if not root.is_dir():
            continue
        for directory, children, names in os.walk(root, followlinks=False):
            directory = Path(directory)
            children[:] = [name for name in children if name not in SKIP_DIRS and not excluded(directory / name, args)]
            for name in names:
                path = directory / name
                if relevant(path) and not excluded(path, args) and path.is_file():
                    files.add(path.resolve())
    # Referenced figure assets may legitimately be outside the supplied root.
    for path in tuple(files):
        if path.name == "qualitative_manifest.json":
            for figure in read_json(path).get("figures", []):
                asset = (path.parent / figure.get("path", "")).resolve()
                if asset.is_file() and not excluded(asset, args):
                    files.add(asset)
    for path in (args.paper_pdf, BUILDER):
        if path is not None:
            files.add(Path(path).resolve())
    return sorted(files)


def file_identity(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise InputChanged(f"Input changed while reading: {path}")
    return {"path": str(path), "size_bytes": after.st_size, "sha256": digest.hexdigest()}


def snapshot(args):
    files = [file_identity(path) for path in discover_inputs(args)]
    payload = {"roots": [{"path": str(path), "exists": path.exists()} for path in input_roots(args)], "files": files}
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def automatic_completion(args, files):
    """Require successful planned jobs and actual figures; human data is separate."""
    result = {"ablation_complete": False, "supplement_complete": False,
              "qualitative_complete": False, "human_study_excluded": True}
    ablation_path = args.ablation_dir / "status.json" if args.ablation_dir is not None else None
    if ablation_path is not None and ablation_path.is_file():
        ablation = read_json(ablation_path)
        variants = ablation.get("variants", {})
        result["ablation_complete"] = (ablation.get("status") == "complete" and
            set(VARIANTS).issubset(variants) and all(
                variants[variant].get(stage, {}).get("status") == "complete"
                for variant in variants for stage in ("training", "static", "dynamic")))
        result["ablation_status"] = ablation.get("status")
    supplement_paths = [Path(record["path"]) for record in files if Path(record["path"]).name == "supplement_status.json"]
    supplements = []
    for path in supplement_paths:
        data = read_json(path)
        stages = data.get("stages", {})
        manifest_path = path.with_name("supplement_manifest.json")
        expected = None
        if manifest_path.is_file():
            expected = {stage["name"] for stage in read_json(manifest_path).get("stages", [])}
        complete = (data.get("status") == "complete" and bool(stages) and
            bool(expected) and expected == set(stages) and
            all(stage.get("status") == "complete" for stage in stages.values()))
        supplements.append({"path": str(path), "status": data.get("status"), "complete": complete})
    result["supplements"] = supplements
    result["supplement_complete"] = bool(supplements) and all(row["complete"] for row in supplements)
    figures = set()
    for record in files:
        path = Path(record["path"])
        if path.name != "qualitative_manifest.json":
            continue
        for figure in read_json(path).get("figures", []):
            name = str(figure.get("id", "")).lower().replace(" ", "")
            asset = path.parent / figure.get("path", "")
            if (name in ("figure4", "figure5", "figure6") and asset.is_file() and asset.stat().st_size > 0 and
                    (name != "figure6" or figure.get("protocol", {}).get("conflicting_prompt") is True)):
                figures.add(name)
    result["figures"] = sorted(figures)
    result["qualitative_complete"] = figures == {"figure4", "figure5", "figure6"}
    result["complete"] = all(result[key] for key in ("ablation_complete", "supplement_complete", "qualitative_complete"))
    return result


def build_command(args):
    command = [sys.executable, str(BUILDER), "--base-run-dir", str(args.base_run_dir),
               "--output-dir", str(args.output_dir)]
    for path in args.extra_results_dir:
        command += ["--extra-results-dir", str(path)]
    if args.ablation_dir is not None:
        command += ["--ablation-dir", str(args.ablation_dir)]
    if args.paper_pdf is not None:
        command += ["--paper-pdf", str(args.paper_pdf)]
    return command


def run_builder(args, log_path):
    with log_path.open("a", encoding="utf-8") as log:
        command = build_command(args)
        log.write(f"\n[{now()}] {json.dumps(command)}\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Report builder exited with {result.returncode}; see {log_path}")


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another report watcher holds {path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def poll(args, state, *, builder=run_builder):
    """One attempt; failed attempts never advance last_success or finality."""
    paths = watcher_paths(args)
    started = time.monotonic()
    state.update(last_checked_at=now(), status="watching", automatic_complete=False)
    try:
        before = snapshot(args)
        completion = automatic_completion(args, before["files"])
        state["automatic_completion"] = completion
        previous = state.get("last_success") or {}
        artifacts_exist = all((args.output_dir / name).is_file() for name in ("REPORT.html", "REPORT.md", "report_manifest.json"))
        need_build = (before["fingerprint"] != previous.get("input_fingerprint") or not artifacts_exist or
                      completion["complete"] and not previous.get("final_build"))
        if need_build:
            state.update(status="building", last_attempt_at=now())
            write_json(paths["status"], state)
            builder(args, paths["log"])
            after = snapshot(args)
            if before["fingerprint"] != after["fingerprint"]:
                raise InputChanged("Inputs changed during report generation; a fresh build will be retried")
            state["last_success"] = {"at": now(), "generation_seconds": round(time.monotonic() - started, 3),
                "input_fingerprint": before["fingerprint"], "input_file_count": len(before["files"]),
                "final_build": completion["complete"]}
            state["successful_builds"] = state.get("successful_builds", 0) + 1
            print(f"[{now()}] Report refreshed; final automatic build: {completion['complete']}", flush=True)
        state.update(status="complete" if completion["complete"] else "watching",
                     automatic_complete=completion["complete"])
        write_json(paths["status"], state)
        return completion["complete"], True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        state.update(status="retry_pending", automatic_complete=False,
            last_failure={"at": now(), "error": str(error), "attempt_seconds": round(time.monotonic() - started, 3)})
        write_json(paths["status"], state)
        print(f"[{now()}] Report refresh postponed: {error}", flush=True)
        return False, False


def main(argv=None):
    args = parse_args(argv)
    paths = watcher_paths(args)
    with exclusive_lock(paths["lock"]):
        previous = {}
        if paths["status"].is_file():
            try:
                previous = read_json(paths["status"])
            except (OSError, ValueError):
                pass
        state = {"schema_version": 1, "started_at": now(), "pid": os.getpid(),
                 "output_dir": str(args.output_dir), "interval_seconds": args.interval,
                 "log_path": str(paths["log"]), "human_study_excluded_from_finality": True,
                 **{key: previous[key] for key in ("last_success", "last_failure", "successful_builds") if key in previous}}
        try:
            while True:
                complete, success = poll(args, state)
                if complete or args.once:
                    state.update(pid=None, stopped_at=now())
                    write_json(paths["status"], state)
                    return 0 if success else 1
                time.sleep(args.interval)
        except KeyboardInterrupt:
            state.update(status="stopped", pid=None, stopped_at=now())
            write_json(paths["status"], state)
            return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
