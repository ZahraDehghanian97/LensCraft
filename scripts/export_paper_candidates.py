#!/usr/bin/env python3
"""Render saved model results into a browsable collection of paper candidates."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visualization.data import ComparisonSample, SCHEMA, save_result
from visualization.export import export_figure, keyframe_constraints


def sample_cohort(sample):
    """Prefer the recorded evaluation cohort; otherwise inspect full subject poses."""
    cohort = sample.metadata.get("cohort")
    if cohort in ("static", "dynamic"):
        return cohort
    if sample.subject is None:
        raise ValueError(f"Sample '{sample.sample_id}' has neither a cohort nor subject poses.")
    return "static" if np.allclose(sample.subject, sample.subject[0], rtol=0, atol=1e-6) else "dynamic"


def sample_task(sample):
    task = sample.metadata.get("paper_task")
    if task in ("prompt", "keyframe", "conflict"):
        return task
    if sample.metadata.get("conflicting_prompt") or any("conflict" in name for name in sample.trajectories):
        return "conflict"
    if any("key_framing" in name for name in sample.trajectories):
        return "keyframe"
    return "prompt"


def sample_fingerprint(sample, cohort, task):
    """Collapse repeated copies, keeping distinct predictions and actual inputs."""
    digest = hashlib.sha256()
    digest.update(json.dumps([sample.dataset, sample.sample_id, sample.prompt, cohort, task],
                             ensure_ascii=False).encode())
    arrays = [("camera:" + name, value) for name, value in sample.trajectories.items()]
    arrays.extend((("subject", sample.subject), ("volume", sample.volume)))
    for name, value in sorted(arrays):
        digest.update(name.encode())
        if value is None:
            digest.update(b"null")
        else:
            values = np.asarray(value, dtype="<f8")
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes())
    for method in [None, *sorted(sample.trajectories)]:
        indices, poses, source = keyframe_constraints(sample, method)
        digest.update(json.dumps([method, indices, source]).encode())
        if poses is not None:
            digest.update(np.asarray(poses, dtype="<f8").tobytes())
    return digest.hexdigest()


def collect_candidates(paths):
    """Read every sample in portable files/directories, excluding metadata sidecars."""
    files = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            for entry in sorted(path.rglob("*.json")):
                files.setdefault(entry, False)
        elif path.is_file():
            files[path] = True
        else:
            raise FileNotFoundError(path)
    candidates = {}
    for path, explicit in files.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            if explicit:
                raise ValueError(f"Expected a portable comparison bundle: {path}")
            continue
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, record in enumerate(payload["samples"]):
            sample = ComparisonSample(**record)
            if sample.dataset == "demo" or sample.metadata.get("demo"):
                raise ValueError(f"Synthetic demo '{sample.sample_id}' cannot be used as a paper model-result candidate.")
            references = {"gt", "reference", "ground truth", "ground_truth"}
            if not any(name.lower() not in references for name in sample.trajectories):
                raise ValueError(f"Sample '{sample.sample_id}' contains no model output.")
            cohort, task = sample_cohort(sample), sample_task(sample)
            if cohort == "dynamic" and sample.subject is None:
                raise ValueError(f"Dynamic sample '{sample.sample_id}' needs subject poses.")
            fingerprint = sample_fingerprint(sample, cohort, task)
            candidate = candidates.setdefault(fingerprint, dict(sample=sample, cohort=cohort, task=task,
                                                                  fingerprint=fingerprint, sources=[]))
            candidate["sources"].append({"path": str(path), "sample_index": index, "sha256": source_hash})
    if not candidates:
        raise ValueError("No portable model-result samples were found.")
    return sorted(candidates.values(), key=lambda item: (item["cohort"], item["task"],
                                                        item["sample"].sample_id, item["fingerprint"]))


def build_candidates(results, output_dir, *, dpi=300, overwrite=False):
    """Write one comparison per sample/protocol, without generating predictions."""
    if dpi < 1:
        raise ValueError("dpi must be positive.")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ValueError("Candidate output already exists; choose a new folder or pass --overwrite.")
    candidates = collect_candidates(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    entries = []
    for candidate in candidates:
        sample, cohort, task = candidate["sample"], candidate["cohort"], candidate["task"]
        counters[(cohort, task)] += 1
        # Keep shortlist IDs stable when more bundles are added to a gallery.
        identifier = f"{cohort[0].upper()}-{task[0].upper()}-{candidate['fingerprint'][:10]}"
        sample.metadata["paper_candidate_id"] = identifier
        caption = f"{identifier}  |  {cohort.title()} subject  |  " + {
            "prompt": "Text conditioning", "keyframe": "Keyframe conditioning", "conflict": "Source / prompt conflict"
        }[task]
        written = export_figure(sample, output_dir / "figures" / identifier,
                                figure_mode=cohort, include_input=cohort == "static" and task == "keyframe",
                                show_keyframes=task == "keyframe", dpi=dpi, title=caption)
        bundle = save_result(sample, output_dir / "bundles" / f"{identifier}.json")
        raw_prompt = sample.metadata.get("raw_prompt")
        movement = sample.metadata.get("camera_motion", "")
        if not movement and isinstance(raw_prompt, dict):
            movement = raw_prompt.get("movement", {}).get("type", "")
        entry = dict(id=identifier, sample_id=sample.sample_id, dataset=sample.dataset,
                     cohort=cohort, task=task, prompt=sample.prompt, camera_motion=movement,
                     methods=list(sample.trajectories), keyframes=sample.keyframes,
                     fingerprint=candidate["fingerprint"], sources=candidate["sources"],
                     assets={name: str(path.relative_to(output_dir)) for name, path in written.items()},
                     bundle=str(bundle.relative_to(output_dir)))
        entries.append(entry)
        print(f"{identifier}: {sample.sample_id}", flush=True)
    from visualization.gallery import write_gallery
    gallery = write_gallery(entries, output_dir)
    manifest = dict(schema="lenscraft.paper-candidates.v1", created_at=datetime.now(timezone.utc).isoformat(),
                    source="saved actual model outputs; no model inference performed by the renderer",
                    candidate_count=len(entries),
                    unique_sample_count=len({(entry["dataset"], entry["sample_id"]) for entry in entries}),
                    counts={f"{cohort}/{task}": count for (cohort, task), count in sorted(counters.items())},
                    dpi=dpi, gallery=gallery.name, candidates=entries)
    (output_dir / "candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        f"# Paper figure candidates\n\n{len(entries)} figures covering {manifest['unique_sample_count']} distinct scenes.\n\n"
        "Open [index.html](index.html) to filter, compare and shortlist figures. Download the selection JSON from that page to retain candidate IDs and source bundles.\n\n"
        "Each figure includes PNG, PDF, SVG, figure metadata, and a portable comparison bundle. Dynamic figures use five time columns; static figures compare complete camera trajectories.\n\n"
        "These examples are for qualitative selection, not a representative quantitative evaluation. The manifest records all source files and hashes; predictions are preserved as recorded.\n",
        encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True, type=Path, help="Portable bundle files or directories; every sample is included")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_candidates(args.results, args.output_dir, dpi=args.dpi, overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Paper candidates: {error}\n")
    print(json.dumps({key: result[key] for key in ("candidate_count", "unique_sample_count", "counts", "gallery")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
