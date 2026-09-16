#!/usr/bin/env python3
"""Download report inputs from an experiment, excluding weights and trajectory caches.

SSH uses the user's configured host alias. No remote files are modified. Build
the report separately after synchronization, using build_paper_report.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import tempfile


PATTERNS = (
    "project_commit.txt", "smoke_status.json", "*_driver.log",
    "report_watch_status.json", "supplement/supplement_status.json",
    "supplement/supplement_manifest.json", "supplement/results/*.json",
    "supplement/supplementary_efficiency/*.json",
    "supplement/*sampling*.json", "ablation/status.json",
    "ablation/run_manifest.json", "ablation/results/*.json",
    "ablation/*/train/.hydra/config.yaml",
    "ablation/*/train/lightning_logs/version_0/metrics.csv",
    "ablation/*/train/continuations/*.csv", "qualitative/**/*",
)


def remote_command(root):
    if not PurePosixPath(root).is_absolute():
        raise ValueError("Remote run directory must be absolute")
    program = (
        "import pathlib,sys,tarfile\n"
        f"root=pathlib.Path({root!r})\n"
        "if not root.is_dir(): raise FileNotFoundError(root)\n"
        f"patterns={PATTERNS!r}\n"
        "files=sorted({p for pattern in patterns for p in root.glob(pattern) "
        "if p.is_file() and not p.is_symlink()})\n"
        "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:\n"
        " for path in files: archive.add(path,arcname=str(path.relative_to(root)),recursive=False)\n"
    )
    return "python3 -c " + shlex.quote(program)


def extract_snapshot(archive_path, destination):
    destination = Path(destination).resolve()
    inventory = destination / ".synced_paper_inputs.json"
    previous = json.loads(inventory.read_text()) if inventory.is_file() else {}
    for name in previous:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or destination not in (destination / name).resolve().parents:
            raise ValueError(f"Unsafe previous sync inventory path: {name}")
    # Validate every entry before writing any input. Status files are installed
    # after their measurements so an interrupted sync cannot claim new success.
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        current = {}
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe report archive entry: {member.name}")
            target = destination.joinpath(*name.parts).resolve()
            if destination not in target.parents:
                raise ValueError(f"Archive entry escapes destination: {member.name}")
        members.sort(key=lambda item: ("status" in item.name, item.name))
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    source = archive.extractfile(member)
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        handle.write(chunk)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(target)
            current[member.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        # Remove only unchanged files downloaded by this program that the server
        # has since archived, e.g. superseded equal-shape timing records. Local
        # files not owned by the previous sync inventory are left in place.
        for name, checksum in previous.items():
            path = destination / name
            if name not in current and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == checksum:
                path.unlink()
        temporary_inventory = inventory.with_suffix(".tmp")
        temporary_inventory.write_text(json.dumps(current, indent=2) + "\n")
        temporary_inventory.replace(inventory)
    return len(members)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="me_DML4_proxy")
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--local-run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.host.startswith("-") or any(char.isspace() for char in args.host):
        parser.error("Use a configured SSH host alias")
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=2",
               args.host, remote_command(args.remote_run_dir)]
    with tempfile.TemporaryDirectory(prefix="lenscraft-report-sync-") as directory:
        archive = Path(directory) / "inputs.tar.gz"
        with archive.open("wb") as handle:
            subprocess.run(command, stdout=handle, check=True)
        count = extract_snapshot(archive, args.local_run_dir)
    print(json.dumps({"synced_files": count, "local_run_dir": str(args.local_run_dir.resolve())}))


if __name__ == "__main__":
    main()
