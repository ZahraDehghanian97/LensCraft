#!/usr/bin/env python3
"""Reconstruct CCDM's released six-motion evaluator; never uses CLaTr.

This is a newly seeded evaluator, not the authors' original checkpoint. The
released 300-frame architecture/recipe differs from the paper's 60-frame
experiment. Run in a fresh directory; resuming requires --resume explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from testing.ccdm_native import CCDMNativeClassifier  # noqa: E402


LABEL_MAP = {"static": 0, "push_in": 1, "pull_out": 2, "pan": 3, "boom": 4, "orbit": 5}
MOTION_PHRASES = {
    "is static": "static", "pushes in to the character": "push_in",
    "pulls out to the character": "pull_out", "pans to the character": "pan",
    "booms up": "boom", "booms down": "boom", "rotates around the character": "orbit",
}
MOTION_RE = re.compile(r"\bthe camera (" + "|".join(map(re.escape, MOTION_PHRASES)) + r")(?=[.!?]|$)", re.I)
FRAMES = 300
PROTOCOL = "ccdm-native-classifier-reconstruction-v1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def motion_label(info):
    """Require exactly one native motion sentence, preserving the full caption."""
    if isinstance(info, str):
        prompt = info
    elif isinstance(info, (list, tuple, np.ndarray)) and len(info) and all(isinstance(s, str) for s in info):
        prompt = " ".join(info)
    else:
        raise ValueError("native caption must be text or a nonempty sequence of strings")
    matches = MOTION_RE.findall(prompt)
    if len(matches) != 1:
        raise ValueError(f"native caption has {len(matches)} recognized motion sentences; expected exactly one: {prompt!r}")
    return LABEL_MAP[MOTION_PHRASES[matches[0].lower()]], prompt


def load_corpus(data_path):
    raw = np.load(data_path, allow_pickle=True).item()
    if not isinstance(raw, dict) or "cam" not in raw or "info" not in raw:
        raise ValueError("native data.npy must contain cam and info")
    if len(raw["cam"]) != len(raw["info"]) or len(raw["cam"]) < 10:
        raise ValueError("native cam/info counts must match and include at least 10 samples")
    cameras, labels, prompts = [], [], []
    for index, (camera, info) in enumerate(zip(raw["cam"], raw["info"])):
        camera = np.asarray(camera)
        if (camera.ndim != 2 or camera.shape[1] != 5 or not 1 <= len(camera) <= FRAMES
                or camera.dtype.kind not in "fiu" or not np.isfinite(camera).all()):
            raise ValueError(f"invalid native5D trajectory at source index {index}")
        try:
            label, prompt = motion_label(info)
        except ValueError as exc:
            raise ValueError(f"source index {index}: {exc}") from exc
        cameras.append(camera)
        labels.append(label)
        prompts.append(prompt)
    if set(labels) != set(LABEL_MAP.values()):
        raise ValueError("corpus must contain all six native motion classes; no samples are dropped")
    return cameras, np.asarray(labels, dtype=np.int64), list(raw["info"]), prompts


def split_indices(count, seed=42):
    """New deterministic random 9:1 split; no unpublished original IDs implied."""
    if count < 10:
        raise ValueError("at least 10 samples are required")
    order = np.random.RandomState(seed).permutation(count)
    boundary = int(0.9 * count)
    return order[:boundary], order[boundary:]


def prepare_corpus(data_path, output, seed=42):
    cameras, labels, infos, prompts = load_corpus(data_path)
    train, test = split_indices(len(cameras), seed)
    # Match classify.py: normalization includes all unpadded train+test frames.
    # Concatenation follows split order, exactly as the released preparation.
    joined = np.concatenate([cameras[int(i)] for i in np.concatenate((train, test))], axis=0)
    mean, std = np.mean(joined, axis=0), np.std(joined, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("nonfinite normalization statistics")
    del joined
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    schema = {}
    for name, indices in (("train", train), ("test", test)):
        schema[name + "_cam"] = [cameras[int(i)] for i in indices]
        schema[name + "_label"] = labels[indices]
        schema[name + "_info"] = [infos[int(i)] for i in indices]
        schema[name + "_source_indices"] = indices
    np.save(output / "data.npy", schema, allow_pickle=True)
    np.save(output / "Mean_Std.npy", {"Mean": mean, "Std": std}, allow_pickle=True)
    write_json(output / "label_mapping.json", {
        "label_map": LABEL_MAP, "motion_phrases": MOTION_PHRASES,
        "mapping_provenance": "new explicit native-caption map; original unpublished numeric mapping unknown",
    })
    write_json(output / "source_indices.json", {"train": train.tolist(), "test": test.tolist()})
    with (output / "source_labels.jsonl").open("w") as stream:
        test_set = set(map(int, test))
        for i, (label, info, prompt) in enumerate(zip(labels, infos, prompts)):
            stream.write(json.dumps({"source_index": i, "source_id": f"ccdm-{i:06d}",
                "label": int(label), "motion": list(LABEL_MAP)[int(label)],
                "split": "test" if i in test_set else "train",
                "raw_caption": info.tolist() if isinstance(info, np.ndarray) else info,
                "prompt": prompt}, ensure_ascii=False) + "\n")
    manifest = {
        "protocol": PROTOCOL, "reconstructed": True, "source_path": str(Path(data_path).resolve()),
        "source_sha256": digest(data_path), "samples": len(cameras),
        "split": {"seed": seed, "algorithm": "numpy.RandomState(seed).permutation; first floor(0.9*N) train",
                  "train_count": len(train), "test_count": len(test), "original_split_recovered": False},
        "class_counts": {name: dict(sorted(Counter(map(int, labels[idx])).items()))
                         for name, idx in (("all", np.arange(len(labels))), ("train", train), ("test", test))},
        "label_map": LABEL_MAP, "frames": FRAMES,
        "native_lengths": {"min": min(map(len, cameras)), "max": max(map(len, cameras))},
        "normalization": {"policy": "all unpadded train+test native frames, population std (released recipe)",
                          "test_data_included": True, "padding": "repeat normalized final frame to300",
                          "mean": mean.tolist(), "std": std.tolist(), "epsilon": 1e-8},
        "differences_from_original_paper": [
            "New seed42 split and explicit numeric label map; original classifier checkpoint/split unavailable.",
            "Released classifier uses4encoderlayers/300frames; paper AppendixB specifies6encoderlayers and evaluation reports60frames.",
            "No test-based checkpoint selection; fixed final scheduled epoch.",
            "Deterministic random seeds and resumable checkpoints added.",
        ],
        "artifacts": {name: digest(output / name) for name in
                      ("data.npy", "Mean_Std.npy", "source_indices.json", "source_labels.jsonl", "label_mapping.json")},
    }
    write_json(output / "data_manifest.json", manifest)
    return schema, mean, std, manifest


class NativeDataset(Dataset):
    def __init__(self, cameras, labels, mean, std):
        self.cameras, self.labels = cameras, labels
        self.mean, self.std = np.asarray(mean), np.asarray(std)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        values = (self.cameras[index] - self.mean[None, :]) / (self.std[None, :] + 1e-8)
        values = np.concatenate((values, np.repeat(values[-1:], FRAMES - len(values), axis=0)))
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite normalized sample {index}")
        return torch.from_numpy(values), int(self.labels[index])


def seed_runtime(seed, device):
    # Initial seed is a declared reconstruction choice. Preserve the release's
    # shared global RNG for loader shuffling and dropout; no epoch reseeding.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def run(args):
    output = Path(args.output_dir).resolve()
    if (args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or args.num_workers < 0
            or (args.threads is not None and args.threads < 1)):
        raise ValueError("epochs, batch size, learning rate and threads must be positive; workers nonnegative")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("max_train_batches must be positive")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("output directory is not empty; choose a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".training.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("training lock exists; verify the recorded process before explicit stale-lock removal") from exc
    try:
        os.write(lock_fd, str(os.getpid()).encode())
        os.close(lock_fd)
        return _run_locked(args, output)
    finally:
        lock.unlink(missing_ok=True)


def _run_locked(args, output):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.precision != "fp32":
        raise ValueError("released classifier training uses FP32; mixed precision is unsupported")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    seed_runtime(args.seed, device)
    config = {key: getattr(args, key) for key in
              ("seed", "epochs", "batch_size", "learning_rate", "precision", "max_train_batches", "num_workers", "threads")}
    config.update({"protocol": PROTOCOL, "reconstructed": True, "frames": FRAMES,
                   "training_reference": "authors' released classify.py; paper/code architecture discrepancy unresolved",
                   "upstream_classify_sha256": digest(Path(__file__).resolve().parents[1] /
                       "third_parties/Camera-control/[2024][EG]Text+keyframe/classify.py"),
                   "trainer_source_sha256": digest(Path(__file__)),
                   "classifier_source_sha256": digest(Path(__file__).resolve().parents[1] / "src/testing/ccdm_native.py"),
                   "optimizer": "torch.optim.Adam defaults", "lr_schedule": "lr0*(1-epoch_index/epochs)",
                   "adam_hyperparameters": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
                   "checkpoint_selection": "fixed last scheduled epoch; test accuracy diagnostic only",
                   "deterministic_algorithms_override": None, "tf32_override": None,
                   "runtime_defaults_note": "PyTorch runtime defaults retained; historical PyTorch/CUDA environment is not recovered",
                   "shuffle": "DataLoader(shuffle=True), shared global torch RNG, no epoch reseeding",
                   "released_training_recipe_match": (args.epochs == 1000 and args.batch_size == 128
                       and args.learning_rate == 1e-4 and args.num_workers == 5
                       and args.max_train_batches is None),
                   "smoke": (args.epochs != 1000 or args.batch_size != 128 or args.learning_rate != 1e-4
                       or args.num_workers != 5 or args.max_train_batches is not None),
                   "device": str(device), "torch_version": str(torch.__version__), "numpy_version": np.__version__})
    if args.resume:
        if not (output / "last.pt").is_file():
            progress_path = output / "progress.json"
            if not progress_path.is_file() or json.loads(progress_path.read_text()).get("state") != "prepared":
                raise ValueError("--resume requires last.pt or an explicitly prepared run")
        old_config = json.loads((output / "training_config.json").read_text())
        if old_config != config:
            raise ValueError("resume configuration differs from the original training configuration")
        manifest = json.loads((output / "data_manifest.json").read_text())
        if digest(args.data_path) != manifest["source_sha256"]:
            raise ValueError("source corpus changed since preparation")
        for name, expected in manifest["artifacts"].items():
            if digest(output / name) != expected:
                raise ValueError(f"prepared artifact changed: {name}")
        schema = np.load(output / "data.npy", allow_pickle=True).item()
        normalization = np.load(output / "Mean_Std.npy", allow_pickle=True).item()
        mean, std = normalization["Mean"], normalization["Std"]
    else:
        schema, mean, std, manifest = prepare_corpus(args.data_path, output, args.seed)
        write_json(output / "training_config.json", config)
    if args.prepare_only:
        write_json(output / "progress.json", {"state": "prepared", "reconstructed": True, "epochs_completed": 0})
        return {"state": "prepared", "train_samples": manifest["split"]["train_count"], "test_samples": manifest["split"]["test_count"]}
    datasets = {name: NativeDataset(schema[name + "_cam"], schema[name + "_label"], mean, std) for name in ("train", "test")}
    criterion = torch.nn.CrossEntropyLoss()
    model = CCDMNativeClassifier(frames=FRAMES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch, updates = 0, 0
    if args.resume and (output / "last.pt").is_file():
        # Own resumable state is trusted and contains Python/NumPy RNG state.
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["data_manifest_sha256"] != digest(output / "data_manifest.json"):
            raise ValueError("checkpoint provenance mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, updates = saved["epochs_completed"], saved["updates"]
        _restore_rng(saved["rng"])
        # A crash may occur after the checkpoint but before the metric/log writes.
        history = []
        lines = (output / "history.jsonl").read_text().splitlines() if (output / "history.jsonl").exists() else []
        for line_index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line_index != len(lines) - 1:
                    raise ValueError("corrupt training history before the final append")
                break  # A process interruption may truncate its last log append.
            if record["epoch"] <= start_epoch:
                history.append(record)
        if not history or history[-1]["epoch"] < start_epoch:
            history.append(saved["last_metrics"])
        (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    result_dir = output / "result"
    result_dir.mkdir(exist_ok=True)
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    started = time.monotonic()
    for ep in range(start_epoch, args.epochs):
        epoch_started = time.monotonic()
        lr = args.learning_rate * (1 - ep / args.epochs)
        optimizer.param_groups[0]["lr"] = lr
        model.train()
        loss_total, train_correct, train_count = 0.0, 0, 0
        for batch_index, (camera, label) in enumerate(loader):
            camera, label = camera.to(device), label.to(device)
            logits = model(camera)
            optimizer.zero_grad()
            loss = criterion(logits, label)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite classifier training loss")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(label)
            train_correct += int((logits.argmax(1) == label).sum())
            train_count += len(label)
            updates += 1
            if args.max_train_batches is not None and batch_index + 1 >= args.max_train_batches:
                break
        model.eval()
        test_correct, test_count = 0, 0
        confusion = np.zeros((6, 6), dtype=np.int64)
        with torch.inference_mode():
            for camera, label in test_loader:
                camera, label = camera.to(device), label.to(device)
                prediction = model(camera).argmax(1)
                test_correct += int((prediction == label).sum())
                test_count += len(label)
                np.add.at(confusion, (label.cpu().numpy(), prediction.cpu().numpy()), 1)
        row = {"epoch": ep + 1, "epoch_index": ep, "learning_rate": lr, "updates": updates,
               "train_samples_seen": train_count, "train_loss": loss_total / train_count,
               "train_accuracy": train_correct / train_count, "test_samples": test_count,
               "test_accuracy": test_correct / test_count, "test_confusion_matrix": confusion.tolist(),
               "epoch_seconds": time.monotonic() - epoch_started, "reconstructed": True,
               "test_used_for_checkpoint_selection": False}
        atomic_torch_save(output / "last.pt", {"config": config, "epochs_completed": ep + 1,
            "updates": updates, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": _rng_state(), "data_manifest_sha256": digest(output / "data_manifest.json"), "last_metrics": row})
        atomic_torch_save(result_dir / "latest.pth", model.state_dict())
        if ep % 100 == 0:
            atomic_torch_save(result_dir / f"model_{ep}.pth", model.state_dict())
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(output / "progress.json", {"state": "training", "epochs_completed": ep + 1,
            "target_epochs": args.epochs, "elapsed_this_process_seconds": time.monotonic() - started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), **row})
        print(json.dumps(row, allow_nan=False), flush=True)
    # If resuming a completed run, retain its existing last epoch metrics.
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    atomic_torch_save(result_dir / "latest.pth", last["model"])
    completion = {"state": "complete", "protocol": PROTOCOL, "reconstructed": True,
        "epochs_completed": args.epochs, "updates": last["updates"], "smoke": config["smoke"],
        "final_test_accuracy": last["last_metrics"]["test_accuracy"],
        "checkpoint_selection": config["checkpoint_selection"], "checkpoint": str(result_dir / "latest.pth"),
        "checkpoint_sha256": digest(result_dir / "latest.pth"), "frames": FRAMES,
        "data_manifest_sha256": digest(output / "data_manifest.json"), "training_config": config,
        "original_paper_evaluator": False}
    write_json(result_dir / "evaluator_manifest.json", {
        "protocol": PROTOCOL, "classifier_origin": "retrained", "reconstructed": True,
        "classifier_checkpoint": {"path": str(result_dir / "latest.pth"),
                                  "sha256": completion["checkpoint_sha256"]},
        "classifier_data": {"path": str(output / "data.npy"), "sha256": digest(output / "data.npy")},
        "normalization": {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()},
        "frames": FRAMES, "label_map": LABEL_MAP,
        "training": {"configuration": config, "epochs_completed": args.epochs,
                     "updates": last["updates"], "data_manifest_path": str(output / "data_manifest.json"),
                     "data_manifest_sha256": completion["data_manifest_sha256"],
                     "checkpoint_selection": config["checkpoint_selection"],
                     "normalization_policy": manifest["normalization"]["policy"],
                     "differences_from_original_paper": manifest["differences_from_original_paper"]},
    })
    write_json(output / "completion.json", completion)
    write_json(output / "progress.json", completion)
    return completion


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-path", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--epochs", type=int, default=1000)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--precision", choices=("fp32",), default="fp32")
    result.add_argument("--num-workers", type=int, default=5)
    result.add_argument("--threads", type=int, default=None, help="Optional runtime setting; release leaves it unspecified")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--max-train-batches", type=int, help="Smoke only: cap batches each epoch; recorded in provenance")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), allow_nan=False), flush=True)
