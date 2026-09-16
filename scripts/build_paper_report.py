#!/usr/bin/env python3
"""Build an auditable, paper-ordered report using only the Python standard library.

This reads measurements, never the manuscript's numerical claims. Missing
experiments remain pending. Run --help or see docs/paper_report.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

METRICS = ("fcd", "precision", "recall", "density", "coverage", "clip_score", "clatr_score")
METRIC_LABELS = ("FID/FCD (CLaTr) ↓", "P ↑", "R ↑", "D ↑", "C ↑", "CS ↑", "CLaTr ↑")
MODELS = ("gendop", "ccdm", "et_adaln", "et_incontext", "et_ca", "lens_craft")
MODEL_LABELS = {"gendop": "GenDOP", "ccdm": "CCDM", "et_ca": "E.T. (ca)",
                "et_adaln": "E.T. (adaln)", "et_incontext": "E.T. (incontext)", "lens_craft": "LensCraft"}
VARIANTS = ("full_scheduled", "without_clip", "without_first_frame", "without_relative", "without_speed",
            "without_cycle", "without_teacher", "without_volume", "without_noise")
VARIANT_LABELS = {"full_scheduled": "Matched schedule-enabled control", "without_clip": "Without CLIP loss",
                  "without_first_frame": "Without first-frame loss", "without_relative": "Without relative loss",
                  "without_speed": "Without speed loss", "without_cycle": "Without cycle loss",
                  "without_teacher": "Without teacher schedule", "without_volume": "Without subject volume",
                  "without_noise": "Without input noise"}
CRITERIA = ("prompt_alignment", "naturalness", "geometric_stability", "movement_correctness", "framing_correctness")
HUMAN_COLUMNS = ("scene_id", "evaluator_id", "model", *CRITERIA, "rank")
MISSING = "—"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value):
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else MISSING
    return str(value)


def canonical_variant(value):
    if not value or value == "paper_random_k":
        return "main"
    value = str(value)
    if value in ("full_control", "scheduled_full", "full_scheduled"):
        return "full_scheduled"
    if value.startswith("no_"):
        return "without_" + value[3:]
    return value


def model_id(data):
    name = data.get("model_type", "unknown")
    if name == "lenscraft":
        name = "lens_craft"
    return "et_" + data.get("et_type", "ca") if name == "et" else name


def canonical_mode(mode):
    if "random_1_10" in mode.lower() or "random1to10" in mode.lower():
        return "P+KF_RANDOM_1_10" if "prompt" in mode.lower() or mode.startswith("P+") else "KF_RANDOM_1_10"
    return {"prompt_generation": "P", "reconstruction": "ST", "hybrid_generation": "P+ST"}.get(mode, mode)


def read_measurements(roots):
    """Reject incomparable evaluators/cohorts and conflicting duplicate values."""
    records, sources, fingerprints, cohort_counts = {}, {}, None, {}
    expected_fraction = None
    holdout_counts = None
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("metrics_*.json")):
            absolute = str(path.resolve())
            if absolute in sources:
                continue
            payload = path.read_bytes()
            data = json.loads(payload)
            if not isinstance(data.get("metrics"), dict):
                continue
            provenance = data.get("evaluation_provenance", {})
            fp = {key: provenance.get(key, {}).get("fingerprint") for key in ("semantic_evaluator", "clatr_evaluator")}
            if not all(fp.values()):
                raise ValueError(f"Missing required evaluator fingerprints: {path}")
            if fingerprints is not None and fp != fingerprints:
                raise ValueError(f"Evaluator fingerprint mismatch: {path}")
            fingerprints = fp
            fraction = data.get("test_fraction")
            counts = data.get("test_sample_counts", {})
            cohort = data.get("set")
            if not finite(fraction) or not 0 < fraction <= 1 or cohort not in ("static", "dynamic", "all"):
                raise ValueError(f"Missing/invalid test fraction or cohort: {path}")
            if expected_fraction is not None and fraction != expected_fraction:
                raise ValueError(f"Test fraction mismatch: {path}")
            expected_fraction = fraction
            if not all(isinstance(counts.get(k), int) and counts[k] > 0 for k in ("original_holdout", "fractional_holdout", "cohort")):
                raise ValueError(f"Missing/invalid sample counts: {path}")
            current_holdout = (counts["original_holdout"], counts["fractional_holdout"])
            if holdout_counts is not None and holdout_counts != current_holdout:
                raise ValueError(f"Holdout counts mismatch: {path}")
            holdout_counts = current_holdout
            if counts["cohort"] > counts["fractional_holdout"] or abs(counts["fractional_holdout"] - counts["original_holdout"] * fraction) > 1:
                raise ValueError(f"Inconsistent fractional sample counts: {path}")
            if cohort in cohort_counts and cohort_counts[cohort] != counts["cohort"]:
                raise ValueError(f"Cohort sample counts mismatch: {path}")
            cohort_counts[cohort] = counts["cohort"]
            sources[absolute] = {"path": absolute, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
                                 "evaluation_provenance": provenance, "generation_provenance": data.get("generation_provenance", {}),
                                 "keyframe_protocol": data.get("keyframe_protocol", {}), "test_sample_counts": counts}
            variant = canonical_variant(data.get("variant"))
            for mode, values in data["metrics"].items():
                if not isinstance(values, dict):
                    raise ValueError(f"Invalid metrics for {mode}: {path}")
                values = {key.rsplit("/", 1)[-1]: val for key, val in values.items()}
                for key, val in values.items():
                    if val is not None and not finite(val):
                        raise ValueError(f"Non-finite/non-numeric measurement {mode}/{key}: {path}")
                bootstrap = {key.rsplit("/", 1)[-1]: val for key, val in data.get("bootstrap_std", {}).get(mode, {}).items()}
                for key, stats in bootstrap.items():
                    if not isinstance(stats, list) or len(stats) != 2 or not all(finite(v) for v in stats) or stats[1] < 0:
                        raise ValueError(f"Invalid bootstrap mean/SD {mode}/{key}: {path}")
                if "RANDOM_1_10" in canonical_mode(mode):
                    k_protocol = data.get("keyframe_protocol", {})
                    if k_protocol.get("version") != "paper-random-k-v1" or k_protocol.get("same_masks_across_modes") is not True:
                        raise ValueError(f"Random 1-10 keyframes lack explicit shared sampling protocol: {path}")
                    if not k_protocol.get("sampling_manifest_sha256"):
                        raise ValueError(f"Random keyframe sampling manifest hash missing: {path}")
                record = {"model": model_id(data), "variant": variant, "cohort": cohort, "mode": canonical_mode(mode),
                          "raw_mode": mode, "values": values, "bootstrap": bootstrap, "n": counts["cohort"], "sources": [absolute]}
                key = (record["model"], variant, cohort, record["mode"])
                if key in records:
                    previous = records[key]
                    if previous["values"] != values or previous["bootstrap"] != bootstrap:
                        raise ValueError(f"Conflicting duplicate measurement {key}: {previous['sources'][0]} / {path}")
                    previous["sources"].append(absolute)
                else:
                    records[key] = record
    if not records:
        raise ValueError("No valid metrics_*.json measurements found")
    if "static" in cohort_counts and "dynamic" in cohort_counts and sum(cohort_counts[c] for c in ("static", "dynamic")) != holdout_counts[1]:
        raise ValueError("Static + dynamic counts do not equal fractional holdout")
    return records, sources, {"evaluator_fingerprints": fingerprints, "test_fraction": expected_fraction,
                              "original_holdout": holdout_counts[0], "fractional_holdout": holdout_counts[1], "cohorts": cohort_counts}


def human_study(path):
    """Accept partial valid data, but never call it a completed paper study."""
    if path is None:
        return [], {"status": "pending", "ratings": 0, "scenes": 0, "evaluators": 0, "complete": False}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(HUMAN_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError("Human study CSV lacks required columns")
        rows = list(reader)
    seen, groups = set(), defaultdict(list)
    allowed = {"lens_craft", "ccdm", "et_ca", "gendop"}
    for row in rows:
        key = tuple(row[k] for k in ("scene_id", "evaluator_id", "model"))
        if not all(key) or key in seen or row["model"] not in allowed:
            raise ValueError(f"Invalid/duplicate human rating: {key}")
        seen.add(key)
        for field in (*CRITERIA, "rank"):
            value = float(row[field])
            maximum = 4 if field == "rank" else 10
            if not math.isfinite(value) or not 1 <= value <= maximum or (field == "rank" and int(value) != value):
                raise ValueError(f"Invalid human {field}: {key}")
            row[field] = value
        groups[key[:2]].append(row)
    for key, group in groups.items():
        ranks = [r["rank"] for r in group]
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"Tied/duplicate ranks for scene/evaluator {key}")
    scenes = {r["scene_id"] for r in rows}
    raters = {r["evaluator_id"] for r in rows}
    complete = len(scenes) == 50 and len(raters) == 30 and len(rows) == 6000 and all(len(g) == 4 for g in groups.values())
    result = []
    for model in ("gendop", "ccdm", "et_ca", "lens_craft"):
        selected = [r for r in rows if r["model"] == model]
        result.append([MODEL_LABELS[model], *[sum(r[k] for r in selected) / len(selected) if selected else None for k in (*CRITERIA, "rank")], len(selected)])
    return result, {"status": "complete" if complete else "partial", "ratings": len(rows), "scenes": len(scenes),
                    "evaluators": len(raters), "complete": complete,
                    "sampling_and_evaluator_background": "Not independently verified; requires study protocol documentation"}


def collect_efficiency(roots):
    fair, native, sources = [], [], {}
    fair_signature = None
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("efficiency*.json")):
            if str(path.resolve()) in sources:
                continue
            payload = path.read_bytes()
            data = json.loads(payload)
            if "model_type" not in data:
                continue
            sources[str(path.resolve())] = {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}
            protocol = data.get("protocol", data.get("benchmark_protocol", {}))
            length = data.get("seq_length", protocol.get("seq_length"))
            batch = data.get("batch_size", protocol.get("batch_size"))
            row = {"model": model_id(data), "data": data, "source": str(path.resolve()), "seq_length": length, "batch_size": batch}
            if data.get("report_role") == "supplementary":
                if not data.get("original_record", {}).get("sha256"):
                    raise ValueError(f"Historical efficiency lacks original record hash: {path}")
                for key in ("inference_time_batch_s", "inference_time_batch_std_s", "gflops_per_traj"):
                    value = data.get(key)
                    if value is not None and (not finite(value) or value < 0 or data.get("status") == "unsupported"):
                        raise ValueError(f"Invalid historical efficiency measurement: {path}")
                native.append(row)
                continue
            if data.get("status") == "unsupported":
                if any(data.get(key) is not None for key in ("inference_time_batch_s", "inference_time_batch_std_s", "gflops_per_traj")):
                    raise ValueError(f"Unsupported benchmark contains measured-looking values: {path}")
                fair.append(row)
            elif protocol.get("same_length_same_batch") is True:
                if not isinstance(length, int) or length < 1 or not isinstance(batch, int) or batch < 1:
                    raise ValueError(f"Fair benchmark lacks explicit frame/batch size: {path}")
                signature = (length, batch)
                if fair_signature is not None and fair_signature != signature:
                    raise ValueError(f"Fair benchmark length/batch mismatch: {path}")
                fair_signature = signature
                actual_shape = data.get("actual_output_shape")
                if actual_shape and (actual_shape[0] != batch or actual_shape[1] != length):
                    raise ValueError(f"Fair benchmark actual output differs from requested shape: {path}")
                for key in ("inference_time_batch_s", "inference_time_batch_std_s", "gflops_per_traj"):
                    if data.get(key) is not None and (not finite(data[key]) or data[key] < 0):
                        raise ValueError(f"Invalid fair benchmark measurement {key}: {path}")
                fair.append(row)
            else:
                native.append(row)
    return fair, native, sources


def metric_row(record, prefix):
    if record is None:
        return [*prefix, *([None] * len(METRICS)), None, "pending"]
    complete = all(record["values"].get(k) is not None for k in METRICS)
    return [*prefix, *[record["values"].get(k) for k in METRICS], record["n"], "measured" if complete else "partial"]


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def latex_escape(value):
    value = number(value)
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "↓": r"$\downarrow$", "↑": r"$\uparrow$", "—": "--"}
    return "".join(mapping.get(char, char) for char in value)


def latex_table(columns, rows, caption, label):
    lines = [r"\begin{table*}[htbp]", r"\centering", r"\small", r"\caption{" + latex_escape(caption) + "}",
             r"\label{" + label + "}", r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
             " & ".join(latex_escape(c) for c in columns) + r" \\", r"\hline"]
    lines.extend(" & ".join(latex_escape(v) for v in row) + r" \\" for row in rows)
    return "\n".join([*lines, r"\hline", r"\end{tabular}", r"\end{table*}", ""])


class Report:
    def __init__(self):
        self.blocks = []
        self.tables = {}

    def heading(self, title, level=2):
        self.blocks.append(("heading", (title, level)))

    def paragraph(self, text):
        self.blocks.append(("paragraph", text))

    def table(self, identifier, columns, rows, caption=""):
        self.tables[identifier] = {"columns": list(columns), "rows": rows, "caption": caption}
        self.blocks.append(("table", identifier))

    def image(self, path, caption):
        self.blocks.append(("image", (path, caption)))

    def links(self, items):
        self.blocks.append(("links", items))

    def render(self, output):
        md, body = [], []
        for kind, content in self.blocks:
            if kind == "heading":
                text, level = content
                md.append("#" * level + " " + text)
                body.append(f"<h{level}>{html.escape(text)}</h{level}>")
            elif kind == "paragraph":
                md.append(content)
                body.append("<p>" + html.escape(content) + "</p>")
            elif kind == "image":
                path, caption = content
                md.append(f"![{caption}]({path})")
                body.append(f'<figure><img src="{html.escape(path, quote=True)}" alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
            elif kind == "links":
                md.append(" · ".join(f"[{label}]({path})" for label, path in content))
                body.append('<p class="artifact-links">' + " · ".join(f'<a href="{html.escape(path, quote=True)}">{html.escape(label)}</a>' for label, path in content) + "</p>")
            else:
                table = self.tables[content]
                cols, rows = table["columns"], table["rows"]
                safe = lambda v: number(v).replace("|", "\\|").replace("\n", " ")
                md.append("\n".join(["| " + " | ".join(map(safe, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |",
                                      *["| " + " | ".join(map(safe, row)) + " |" for row in rows]]))
                body.append('<div class="table-scroll"><table dir="ltr"><thead><tr>' + "".join("<th>" + html.escape(str(c)) + "</th>" for c in cols) + "</tr></thead><tbody>" +
                            "".join("<tr>" + "".join("<td>" + html.escape(number(v)) + "</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>")
                write_csv(output / "tables" / (content + ".csv"), cols, rows)
                if content.startswith("table") and content[5:6].isdigit():
                    (output / "tables" / (content + ".tex")).write_text(latex_table(cols, rows, table["caption"], content), encoding="utf-8")
        (output / "REPORT.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
        css = """body{font-family:Tahoma,Arial,sans-serif;background:#f3f5f8;color:#182436;margin:0;line-height:1.9}main{max-width:1400px;margin:30px auto;background:white;padding:45px 55px;border-top:7px solid #12677a;box-shadow:0 4px 30px #10203012}h1{font-size:30px;color:#125567}h2{margin-top:48px;padding-bottom:10px;border-bottom:2px solid #dae5eb;font-size:23px;color:#145568}h3{font-size:19px;margin-top:32px}p{max-width:1100px}.table-scroll{overflow:auto;margin:20px 0 30px}table{border-collapse:collapse;width:100%;font:13px/1.7 Tahoma,Arial,sans-serif;font-variant-numeric:tabular-nums}th{background:#164f62;color:white;font-weight:600;white-space:nowrap}td,th{padding:10px 11px;border-bottom:1px solid #dce3e8;text-align:center}tbody tr:nth-child(even){background:#f2f6f8}td:first-child{text-align:left}figure{margin:25px 0}img{max-width:100%}figcaption{font-size:14px;color:#536575}footer{font-size:12px;color:#647584;margin-top:40px}@media print{body{background:white}main{padding:0;margin:0;box-shadow:none;max-width:none}h2,h3{break-after:avoid}tr,figure{break-inside:avoid}table{font-size:9px}td,th{padding:4px}thead{display:table-header-group}.table-scroll{overflow:visible}a{color:inherit}}@page{size:A4 landscape;margin:14mm}"""
        css += "@media print{table{table-layout:fixed;width:100%}th,td{white-space:normal;overflow-wrap:anywhere;word-break:normal}p{overflow-wrap:anywhere}.table-scroll{max-width:100%}thead{display:table-header-group}tr{break-inside:avoid;page-break-inside:avoid}}"
        css += "@media print{figure{margin:5mm 0;text-align:center}figure img{max-height:150mm;width:auto;max-width:100%;object-fit:contain}figcaption{font-size:10px}}"
        document = '<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>گزارش کامل ارزیابی LensCraft</title><style>' + css + '</style></head><body><main>' + "\n".join(body) + '<footer>تولید خودکار از فایل‌های اندازه‌گیری؛ جزئیات منشأ و هش‌ها در report_manifest.json</footer></main></body></html>'
        (output / "REPORT.html").write_text(document, encoding="utf-8")


def auxiliary_sources(roots):
    result = {}
    for root in roots:
        if not root.exists():
            continue
        for name in ("status.json", "run_manifest.json", "supplement_manifest.json", "supplement_status.json", "manifest.json", "training_summary.json", "split_indices.json", "config.yaml"):
            for path in sorted(root.rglob(name)):
                result[str(path.resolve())] = {"path": str(path.resolve()), "sha256": digest(path)}
    return result


def verify_recorded_hashes(roots, sources):
    by_name = defaultdict(list)
    for path, record in sources.items():
        by_name[Path(path).name].append(record["sha256"])
    checked = 0
    def completed_hashes(node):
        if isinstance(node, dict):
            if node.get("status") == "complete" and isinstance(node.get("output_sha256"), dict):
                yield from node["output_sha256"].items()
            for value in node.values():
                yield from completed_hashes(value)
        elif isinstance(node, list):
            for value in node:
                yield from completed_hashes(value)

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("status.json"):
            data = read_json(path)
            for source, recorded in completed_hashes(data):
                if Path(source).name in by_name:
                    if recorded not in by_name[Path(source).name]:
                        raise ValueError(f"Recorded output hash mismatch: {source} in {path}")
                    checked += 1
        for path in root.rglob("supplement_status.json"):
            manifest_path = path.with_name("supplement_manifest.json")
            if not manifest_path.is_file():
                continue
            manifest = read_json(manifest_path)
            stage_outputs = {stage["name"]: stage["output"] for stage in manifest.get("stages", [])}
            for name, stage in read_json(path).get("stages", {}).items():
                recorded, output_name = stage.get("output_sha256"), stage_outputs.get(name)
                if stage.get("status") == "complete" and isinstance(recorded, str) and output_name in by_name:
                    if recorded not in by_name[output_name]:
                        raise ValueError(f"Recorded supplement output hash mismatch: {output_name}")
                    checked += 1
    return checked


def resolved_scalar_config(path):
    """Extract scalar mapping entries from resolved Hydra YAML, without YAML dependencies.

    This intentionally does not interpret lists, tags, aliases or interpolation.
    It is only a display helper; the original file hash remains authoritative.
    """
    if not path.is_file():
        return {}
    result, stack = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\s*)([A-Za-z_][\w.-]*):(?:\s+(.*))?$", line)
        if not match:
            continue
        indent, key, raw = len(match[1]), match[2], match[3]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        name = ".".join([entry[1] for entry in stack] + [key])
        if raw is None:
            stack.append((indent, key))
        else:
            raw = raw.split(" #", 1)[0].strip()
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                value = raw
            result[name] = value
    return result


def ablation_logged_progress(directory):
    """Summarize recorded progress, including preserved pre-resume CSV logs."""
    progress, sources = {}, {}
    for variant in VARIANTS:
        train = directory / variant / "train"
        files = sorted(set(train.glob("lightning_logs/version_*/metrics.csv")) |
                       set((train / "continuations").glob("*.csv")))
        epochs, steps, scores = [], [], []
        for path in files:
            payload = path.read_bytes()
            sources[str(path.resolve())] = {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
            reader = csv.DictReader(payload.decode("utf-8-sig").splitlines())
            for row in reader:
                for key, target in (("epoch", epochs), ("step", steps), ("val_conditioning_score", scores)):
                    raw = row.get(key)
                    if raw in (None, ""):
                        continue
                    try:
                        value = float(raw)
                    except (ValueError, TypeError) as error:
                        raise ValueError(f"Invalid logged {key} in {path}") from error
                    if not math.isfinite(value) or (key in ("epoch", "step") and (value < 0 or value != int(value))):
                        raise ValueError(f"Invalid logged {key} in {path}")
                    target.append(int(value) if key in ("epoch", "step") else value)
        progress[variant] = {"last_logged_epoch": max(epochs) if epochs else None,
                             "last_logged_step": max(steps) if steps else None,
                             "best_val_conditioning_score": min(scores) if scores else None,
                             "csv_files": len(files)}
    return progress, sources


def qualitative_error_table(bundle, figure_id):
    """Display recorded errors from the verified portable bundle; never impute."""
    rows, notes = [], []

    def value(record, key):
        measured = record.get(key)
        if measured is not None and (not finite(measured) or measured < 0 or (key == "frame_count" and (measured != int(measured) or measured == 0))):
            raise ValueError(f"Invalid qualitative error measurement: {key}")
        return measured

    position_comparisons = []
    rotation_comparisons = []
    for sample in bundle.get("samples", []):
        sample_id = str(sample.get("sample_id", "unknown"))
        label = sample_id.rsplit("_", 1)[-1] if len(sample_id) > 40 else sample_id
        metadata = sample.get("metadata", {})
        if figure_id == "figure5":
            errors = metadata.get("known_pose_errors", {})
            rows.append([label, metadata.get("cohort"), *[value(errors, key) for key in ("frame_count", "position_error_world_units", "rotation_error_deg")]])
        elif figure_id == "figure6":
            errors = metadata.get("conflict_errors", {})
            source, output = errors.get("source_to_target", {}), errors.get("output_to_target", {})
            source_n, output_n = value(source, "frame_count"), value(output, "frame_count")
            if source_n is not None and output_n is not None and source_n != output_n:
                raise ValueError("Qualitative source/output errors have different frame counts")
            source_position, output_position = value(source, "position_error_world_units"), value(output, "position_error_world_units")
            source_rotation, output_rotation = value(source, "rotation_error_deg"), value(output, "rotation_error_deg")
            rows.append([label, output_n, source_position, output_position, source_rotation, output_rotation])
            if source_position is not None and output_position is not None:
                position_comparisons.append(output_position < source_position)
            if source_rotation is not None and output_rotation is not None:
                rotation_comparisons.append(output_rotation < source_rotation)
                if output_rotation > source_rotation:
                    notes.append("در نمونهٔ " + label + "، خطای زاویهٔ خروجی (" + number(output_rotation) + " درجه) از خطای مسیر منبع (" + number(source_rotation) + " درجه) بیشتر شده است؛ بنابراین بهبود در همهٔ مؤلفه‌ها رخ نداده است.")
    if figure_id == "figure5":
        if any(any(value is not None and value > 0 for value in row[-2:]) for row in rows):
            notes.append("در نمونه‌های نمایش‌داده‌شده خطای غیرصفر روی قیدهای واقعی وجود دارد؛ این شکل ادعای عبور دقیق از همهٔ کی‌فریم‌ها را تأیید نمی‌کند. این خطاها مربوط به همین نمونه‌های تصویری‌اند و میانگین کل مجموعهٔ تست نیستند.")
    elif figure_id == "figure6":
        if position_comparisons:
            notes.insert(0, "در " + str(sum(position_comparisons)) + " از " + str(len(position_comparisons)) + " نمونهٔ دارای هر دو اندازه‌گیری، خطای موقعیت خروجی نسبت به مسیر منبع کمتر است. این نتیجه فقط به آزمایش کنترل‌شدهٔ وارونگی زمانی مربوط است.")
    return rows, notes


def build_report(args):
    roots = [args.base_run_dir, *(args.extra_results_dir or [])]
    if args.ablation_dir:
        roots.append(args.ablation_dir)
    output = args.output_dir.resolve()
    if any(output == root.resolve() or root.resolve() in output.parents and output.name == "results" for root in roots):
        raise ValueError("Use a separate report output directory")
    records, sources, protocol = read_measurements(roots)
    fair, native, efficiency_sources = collect_efficiency(roots)
    sources.update(efficiency_sources)
    checked_hashes = verify_recorded_hashes(roots, sources)
    sources.update(auxiliary_sources(roots))
    training_path = args.base_run_dir / "training_summary.json"
    training = read_json(training_path) if training_path.exists() else {}
    resolved_training = resolved_scalar_config(args.base_run_dir / "train/.hydra/config.yaml")
    if args.paper_pdf:
        sources[str(args.paper_pdf.resolve())] = {"path": str(args.paper_pdf.resolve()), "sha256": digest(args.paper_pdf), "role": "protocol reference only; manuscript numbers not imported"}
    human_rows, human = human_study(args.human_study_csv)
    if args.human_study_csv:
        sources[str(args.human_study_csv.resolve())] = {"path": str(args.human_study_csv.resolve()), "sha256": digest(args.human_study_csv)}
    output.mkdir(parents=True, exist_ok=True)
    report = Report()
    report.heading("گزارش کامل ارزیابی LensCraft مطابق ساختار مقاله", 1)
    report.paragraph("زمان ساخت گزارش (UTC): " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    report.paragraph("این گزارش از اندازه‌گیری‌های موجود ساخته شده است و پس از پایان آزمایش‌های تازه با همان دستور بازتولید می‌شود. خط تیره یعنی نتیجه موجود نیست؛ هیچ عددی از نسخهٔ مقاله به‌عنوان نتیجهٔ اجرای فعلی وارد نشده است.")
    report.heading("داده، پروتکل و وضعیت تکمیل — بخش ۴.۱ مقاله، صفحات ۱۵–۱۶")
    dataset = training.get("protocol", {}).get("dataset", {})
    protocol_rows = [["Dataset samples", dataset.get("sample_count")], ["Training / validation / holdout", json.dumps(dataset.get("split_counts", {}))],
                     ["Test fraction", protocol["test_fraction"]], ["Evaluated test samples", protocol["fractional_holdout"]],
                     ["Static subject", protocol["cohorts"].get("static")], ["Dynamic subject", protocol["cohorts"].get("dynamic")],
                     ["Split seed", dataset.get("split_seed")], ["Split indices SHA256", dataset.get("test_indices_sha256")],
                     *[[name + " fingerprint", fp] for name, fp in protocol["evaluator_fingerprints"].items()], ["Verified recorded source hashes", checked_hashes]]
    report.table("protocol", ["Item", "Recorded value"], protocol_rows)
    report.paragraph("ثابت و متحرک به حرکت سوژه اشاره دارند. در حالت‌های LensCraft مسیر و ابعاد سوژه موجود است؛ حالت P شرط متنی دارد و قید مسیر مرجع دوربین یا کی‌فریم مصرف نمی‌کند. ورودی baselineها یکسان با LensCraft نیست: GenDoP از متن استفاده می‌کند و CCDM اطلاعات سوژه را در تبدیل/هم‌ترازی خروجی به کار می‌برد. همهٔ نتایج تجمیع‌شده اثرانگشت یکسان هر دو ارزیاب و تعداد یکسان نمونه‌های هر گروه دارند. انتخاب نمونه‌ها، seed و هش split از اجرای مبنا گزارش شده‌اند؛ فایل‌های قدیمی متریک به‌تنهایی فهرست شناسهٔ نمونه‌ها را ذخیره نکرده‌اند.")
    report.paragraph("FID/FCD در این گزارش فاصلهٔ Fréchet در فضای ویژگی CLaTr است، نه Inception تصویری. CS مقدار بومی clip_score بدون ضرب در ۱۰۰ است؛ ستون جداگانهٔ CS×100 در CSV خام ارائه می‌شود. CS در این اجرا به ارزیاب معنایی منجمد LensCraft وابسته است و داوری انسانی مستقل نیست. Density می‌تواند از ۱ بیشتر باشد. ستون CLaTr مقاله فاقد عدد است و مبنای مقایسهٔ عددی قرار نگرفته است.")
    lookup = lambda model, variant, cohort, mode: records.get((model, variant, cohort, mode))
    t1 = [metric_row(lookup(m, "main", c, "P"), [c, MODEL_LABELS[m]]) for c in ("static", "dynamic") for m in MODELS]
    t2 = [metric_row(lookup("lens_craft", "main", c, mode), [c, mode]) for c in ("static", "dynamic") for mode in ("P", "KF_RANDOM_1_10", "ST", "P+KF_RANDOM_1_10", "P+ST")]
    t3 = [metric_row(lookup("lens_craft", v, "dynamic", "P"), [VARIANT_LABELS[v]]) for v in VARIANTS]
    efficiency_rows = []
    for model in MODELS:
        matched = [r for r in fair if r["model"] == model]
        if len(matched) > 1:
            raise ValueError(f"Conflicting duplicate fair efficiency records: {model}")
        row = matched[0] if matched else {}
        data = row.get("data", {})
        vals = [data.get(k) for k in ("inference_time_batch_s", "inference_time_batch_std_s", "gflops_per_traj")]
        status = "unsupported" if data.get("status") == "unsupported" else "measured" if row and all(finite(v) for v in vals) else "pending"
        efficiency_rows.append([MODEL_LABELS[model], *vals, "lower bound" if data.get("flop_count_is_lower_bound") else "unspecified" if row else None,
                                row.get("batch_size"), row.get("seq_length"), status])
    if not human_rows:
        human_rows = [[MODEL_LABELS[m], *([None] * 6), 0] for m in ("gendop", "ccdm", "et_ca", "lens_craft")]
    completeness = []
    for identifier, rows in (("Table 1", t1), ("Table 2", t2), ("Table 3", t3), ("Table 4", efficiency_rows)):
        measured = sum(r[-1] == "measured" for r in rows)
        completeness.append({"section": identifier, "status": "complete" if measured == len(rows) else "partial" if measured else "pending", "measured": measured, "required": len(rows)})
    completeness.append({"section": "Table 5", "status": human["status"], "measured": human["ratings"], "required": 6000})
    figures = {}
    for root in roots:
        if root.exists():
            for path in root.rglob("qualitative_manifest.json"):
                sources[str(path.resolve())] = {"path": str(path.resolve()), "sha256": digest(path)}
                for figure in read_json(path).get("figures", []):
                    identifier = str(figure.get("id", "")).lower().replace(" ", "")
                    asset = path.parent / figure.get("path", "")
                    if identifier not in ("figure4", "figure5", "figure6") or not asset.is_file():
                        continue
                    if identifier in figures:
                        raise ValueError(f"Duplicate qualitative figure: {identifier}")
                    if identifier == "figure6" and figure.get("protocol", {}).get("conflicting_prompt") is not True:
                        continue
                    asset_dir = output / "assets"
                    asset_dir.mkdir(exist_ok=True)
                    target = asset_dir / (identifier + asset.suffix.lower())
                    if asset.resolve() != target.resolve():
                        shutil.copyfile(asset, target)
                    record = {**figure, "report_path": str(target.relative_to(output)), "source": str(asset.resolve()), "report_assets": {}}
                    for kind, attachment in figure.get("assets", {}).items():
                        attachment_path = path.parent / attachment["path"]
                        if not attachment_path.is_file():
                            raise ValueError(f"Qualitative attachment missing; sync the full export directory: {attachment_path}")
                        attachment_sha = digest(attachment_path)
                        if attachment.get("sha256") and attachment["sha256"] != attachment_sha:
                            raise ValueError(f"Qualitative attachment hash mismatch: {attachment_path}")
                        copied_path = asset_dir / identifier / attachment_path.name
                        copied_path.parent.mkdir(parents=True, exist_ok=True)
                        if copied_path.resolve() != attachment_path.resolve():
                            shutil.copyfile(attachment_path, copied_path)
                        record["report_assets"][kind] = str(copied_path.relative_to(output))
                        sources[str(attachment_path.resolve())] = {"path": str(attachment_path.resolve()), "sha256": attachment_sha}
                    if figure.get("bundle"):
                        bundle = path.parent / figure["bundle"]
                        if not bundle.is_file():
                            raise ValueError(f"Portable qualitative bundle missing: {bundle}")
                        bundle_sha = digest(bundle)
                        if figure.get("bundle_sha256") and figure["bundle_sha256"] != bundle_sha:
                            raise ValueError(f"Portable qualitative bundle hash mismatch: {bundle}")
                        copied_bundle = asset_dir / identifier / "trajectories.json"
                        copied_bundle.parent.mkdir(parents=True, exist_ok=True)
                        if copied_bundle.resolve() != bundle.resolve():
                            shutil.copyfile(bundle, copied_bundle)
                        record["report_assets"]["trajectories"] = str(copied_bundle.relative_to(output))
                        sources[str(bundle.resolve())] = {"path": str(bundle.resolve()), "sha256": bundle_sha}
                    figures[identifier] = record
                    sources[str(asset.resolve())] = {"path": str(asset.resolve()), "sha256": digest(asset)}
    for i in (4, 5, 6):
        completeness.append({"section": f"Figure {i}", "status": "available" if f"figure{i}" in figures else "pending", "measured": int(f"figure{i}" in figures), "required": 1})
    report.table("completion", ["Paper item", "Status", "Available", "Required"], [[r[k] for k in ("section", "status", "measured", "required")] for r in completeness])
    report.heading("جدول ۱ — مقایسهٔ تولید از متن، بخش ۴.۲.۱، صفحهٔ ۱۷")
    report.paragraph("ردیف baselineها فقط prompt_generation نرمال‌شده و بدون مقداردهی اولیه از LensCraft است. نسخه‌های E.T. با نام واقعی معماری مشخص‌اند؛ مقاله نگاشت A/B/C را تعریف نکرده است. همهٔ اعداد برآورد اصلی‌اند؛ میانگین و انحراف معیار bootstrap، با تفکیک روشن از مقدار اصلی، در پیوست آمده‌اند.")
    report.table("table1_sota", ["Subject", "Model", *METRIC_LABELS, "N", "Status"], t1, "Prompt-only evaluation; normalized baselines without LensCraft initialization. CS uses its native scale.")
    for cohort, label in (("static", "سوژهٔ ثابت"), ("dynamic", "سوژهٔ متحرک")):
        available = [lookup(model, "main", cohort, "P") for model in MODELS]
        available = [row for row in available if row is not None and all(row["values"].get(k) is not None for k in ("fcd", "clatr_score"))]
        if available:
            best_fcd = min(available, key=lambda row: row["values"]["fcd"])
            best_clatr = max(available, key=lambda row: row["values"]["clatr_score"])
            report.paragraph("در نتایج موجودِ " + label + "، کمترین FCD متعلق به " + MODEL_LABELS.get(best_fcd["model"], best_fcd["model"]) + " با مقدار " + number(best_fcd["values"]["fcd"]) + " و بیشترین CLaTr متعلق به " + MODEL_LABELS.get(best_clatr["model"], best_clatr["model"]) + " با مقدار " + number(best_clatr["values"]["clatr_score"]) + " است. این نتیجه به همین checkpointها، نمونه‌های تست و پروتکل وابسته است؛ ردیف‌های در انتظار در این مقایسه شرکت ندارند.")
    report.heading("جدول ۲ — ورودی‌های چندوجهی، بخش ۴.۲.۲، صفحهٔ ۱۸")
    report.paragraph("P متن، KF کی‌فریم دوربین و ST کل مسیر مرجع دوربین است. آزمایش اصلی مقاله K تصادفی بین ۱ و ۱۰ می‌خواهد؛ جدول‌های K ثابت جایگزین آن نشده‌اند. در ST و P+ST، مسیر واقعی دوربین ورودی است و نتیجه بازسازی را می‌سنجد؛ با تولید صرفاً از متن شرایط ورودی یکسان نیست. مقایسهٔ کی‌فریم‌های ثابت در انتهای گزارش قرار دارد.")
    report.table("table2_multimodal", ["Subject", "Condition", *METRIC_LABELS, "N", "Status"], t2, "Multimodal conditioning. KF uses random 1 to 10 keyframes; ST supplies the complete reference camera trajectory.")
    report.heading("جدول ۳ — حذف اجزا، بخش ۴.۲.۳، صفحهٔ ۱۹")
    report.paragraph("این جدول فقط کنترل جدید با زمان‌بندی فعال و هشت آموزش مستقل هم‌شرایط را می‌پذیرد. مدل اصلی جدول‌های ۱–۲ با سیاست mixed-conditioning، کنترل این جدول محسوب نمی‌شود. گروه متحرک و ورودی متن انتخاب شده‌اند: مقاله گروه را صریح نمی‌گوید، اما عدد کنترل آن با Dynamic/P برابر است؛ این انتخاب یک استنباط مستند است.")
    report.paragraph("پیکربندی مقاله: ۱۰۰ epoch، batch128، AdamW با lr=1e-4 و betas=(0.9,0.95)، وزن‌های init/relative/speed/CLIP/cycle برابر 1/8/20/50/5. مقادیر واقعیِ استخراج‌شده از config مدل اصلی در پیوست آموزش درج شده‌اند. سیاست صریح conditioning زمان‌بندی‌های قدیمی teacher/noise را غیرفعال می‌کند؛ حذف صرفِ پارامتر زمان‌بندی در آن سیاست ابلیشن واقعی نیست. پیاده‌سازی فعلی ممکن است SO(3) را در اهداف دوران آموزش نیز استفاده کند، درحالی‌که مقاله SVD را فقط برای استنتاج توصیف می‌کند. فایل‌های resolved config و manifest آموزش مرجع دقیق مداخلات‌اند؛ نتایج را بازتولید دقیق همهٔ جزئیات مقاله نمی‌نامیم.")
    report.table("table3_ablation", ["Variant", *METRIC_LABELS, "N", "Status"], t3, "Dynamic prompt-only ablations, using a separately trained matched schedule-enabled control; cohort inferred from manuscript control row.")
    if args.ablation_dir and (args.ablation_dir / "status.json").is_file():
        ablation_status = read_json(args.ablation_dir / "status.json")
        logged_progress, logged_sources = ablation_logged_progress(args.ablation_dir)
        sources.update(logged_sources)
        report.table("ablation_execution", ["Variant", "Training", "Static evaluation", "Dynamic evaluation", "Last logged epoch (0-based)", "Last logged step (0-based)", "Best logged val score ↓"],
                     [[variant, *[ablation_status.get("variants", {}).get(variant, {}).get(stage, {}).get("status", "pending")
                                  for stage in ("training", "static", "dynamic")],
                       *[logged_progress[variant][key] for key in ("last_logged_epoch", "last_logged_step", "best_val_conditioning_score")]] for variant in VARIANTS])
        report.paragraph("epoch و step ستون‌های ثبت‌شده در CSV و از صفر هستند؛ این اعداد آخرین لاگ موجود را نشان می‌دهند و شمارندهٔ لحظه‌ای اجرای GPU نیستند. بهترین score از لاگ اصلی و فایل‌های حفظ‌شدهٔ پیش از ادامهٔ آموزش گرفته می‌شود؛ مقدار ثبت‌نشده خط تیره است. هش تک‌تک CSVهای خوانده‌شده در manifest گزارش ذخیره شده است.")
    report.heading("جدول ۴ — هزینهٔ محاسباتی، بخش ۴.۲.۴، صفحهٔ ۲۰")
    report.paragraph("فقط رکورد دارای تصریح same_length_same_batch و طول/اندازهٔ batch یکسان وارد این جدول می‌شود. زمان، میانگین و انحراف معیار کل batch برحسب ثانیه است؛ GFLOPs به‌ازای مسیر گزارش می‌شود. زمان هر مسیر در batch با تأخیر یک درخواست برابر نیست. اندازه‌گیری‌های قدیمی با طول بومی متفاوت در پیوست‌اند. شمارش FLOPs باید همراه دامنهٔ اپراتورهای پشتیبانی‌شده و هزینهٔ پردازش متن تفسیر شود.")
    report.table("table4_efficiency", ["Model", "Batch time mean (s)", "Batch time SD (s)", "GFLOPs / trajectory", "FLOPs scope", "Batch", "Frames", "Status"], efficiency_rows, "Matched frame length and batch size efficiency. Timing is per batch; FLOPs are per trajectory; observed graph counts may be lower bounds.")
    for row in fair:
        report.paragraph(MODEL_LABELS.get(row["model"], row["model"]) + " benchmark protocol: " + json.dumps(row["data"].get("protocol", row["data"].get("benchmark_protocol", {})), ensure_ascii=False, sort_keys=True))
        data = row["data"]
        details = {k: data[k] for k in ("status", "reason", "limitations", "flop_count_is_lower_bound", "flops_protocol") if k in data}
        if details:
            report.paragraph("محدودیت و دامنهٔ اندازه‌گیری: " + json.dumps(details, ensure_ascii=False, sort_keys=True))
    report.heading("جدول ۵ — مطالعهٔ انسانی، بخش ۴.۳.۱، صفحات ۲۱–۲۲")
    report.paragraph("پروتکل مقاله ۵۰ صحنهٔ تصادفی، ۳۰ ارزیاب، چهار مدل و پنج امتیاز ۱ تا ۱۰ به‌علاوهٔ رتبهٔ ۱ تا ۴ است؛ در مجموع ۶۰۰۰ سطر ارزیابی. Winning Rate در مقاله در واقع میانگین رتبه است و کمتر بهتر است. نمرات خودکار جایگزین داوری انسانی نشده‌اند. وضعیت دادهٔ انسانی: " + human["status"] + ". حتی کامل بودن ساختار CSV، تصادفی بودن انتخاب صحنه‌ها یا پیشینهٔ ارزیاب‌ها را اثبات نمی‌کند.")
    report.table("table5_human_study", ["Model", "Prompt alignment", "Naturalness", "Geometric stability", "Movement correctness", "Framing correctness", "Mean rank ↓", "Ratings"], human_rows, "Human ratings only; mean rank is lower-is-better. Partial or missing studies are not presented as completed.")
    write_csv(output / "human_study_template.csv", HUMAN_COLUMNS, [])
    (output / "human_study_protocol.json").write_text(json.dumps({"expected_scenes": 50, "expected_evaluators": 30, "expected_models": ["gendop", "ccdm", "et_ca", "lens_craft"], "criteria": list(CRITERIA), "score_range": [1, 10], "rank_range": [1, 4], "expected_rating_rows": 6000, "actual": human}, ensure_ascii=False, indent=2), encoding="utf-8")
    report.heading("نتایج کیفی — شکل‌های ۴ تا ۶، بخش‌های ۴.۳.۲ تا ۴.۳.۴")
    figure_notes = {4: "مقایسهٔ تولید از متن بین مدل‌ها، مسیر ایده‌آل و متن؛ صفحهٔ ۲۳.", 5: "مقایسهٔ P+KF، P و مسیر ایده‌آل با نمایش کی‌فریم‌های واقعی؛ صفحهٔ ۲۴.", 6: "ویرایش مسیر مرجع با متن عمداً متعارض؛ بازسازی P+ST با متن سازگار جای این آزمایش را نمی‌گیرد؛ صفحات ۲۵–۲۶."}
    for i in (4, 5, 6):
        report.heading("شکل " + str(i), 3)
        report.paragraph(figure_notes[i])
        figure = figures.get(f"figure{i}")
        if figure:
            report.image(figure["report_path"], figure.get("caption", figure_notes[i]))
            if figure.get("report_assets"):
                report.links([(kind.upper(), path) for kind, path in figure["report_assets"].items()])
            report.paragraph("پروتکل شکل: " + json.dumps(figure.get("protocol", {}), ensure_ascii=False, sort_keys=True))
            bundle_path = figure.get("report_assets", {}).get("trajectories")
            if bundle_path and i in (5, 6):
                error_rows, error_notes = qualitative_error_table(read_json(output / bundle_path), f"figure{i}")
                if error_rows:
                    columns = (["Sample (ID suffix)", "Subject", "Known frames", "Known position error ↓", "Known rotation error (deg) ↓"] if i == 5 else
                               ["Sample (ID suffix)", "Frames", "Source→target position ↓", "Output→target position ↓", "Source→target rotation (deg) ↓", "Output→target rotation (deg) ↓"])
                    report.table(f"figure{i}_recorded_errors", columns, error_rows)
                    report.paragraph("خطاهای موقعیت در واحد جهان دیتاست و زاویه‌ها برحسب درجه‌اند؛ شناسهٔ کامل هر نمونه و مسیرهای واقعی در فایل TRAJECTORIES همان شکل موجود است.")
                for note in error_notes:
                    report.paragraph(note)
        else:
            report.paragraph("در انتظار خروجی تصویری مستندِ این آزمایش؛ نمونه یا تصویر ساخته‌شده بدون خروجی مدل گزارش نشده است.")
    report.heading("پیوست تکمیلی — تحلیل‌های بیشتر از جدول‌ها و شکل‌های اصلی مقاله")
    report.paragraph("تمام نتایج اضافی از این نقطه به بعد آمده‌اند. شکل‌های روش ۱–۳ و شکل A.7 مقاله اندازه‌گیری جدید نیستند. جدول A.6 قواعد تولید shot-size را شرح می‌دهد و با جدول عملکرد مدل اشتباه گرفته نمی‌شود. Appendix B.1/B.2 به جزئیات آموزش و معیارها مربوط‌اند؛ manifest این گزارش منبع بازتولید اجرای واقعی است.")
    report.heading("الف — تعداد ثابت کی‌فریم‌ها", 3)
    fixed = [r for r in records.values() if re.search(r"_k\d+$", r["mode"])]
    fixed.sort(key=lambda r: (r["variant"], r["cohort"], int(re.search(r"_k(\d+)$", r["mode"]).group(1)), r["mode"]))
    report.table("appendix_fixed_keyframes", ["Variant", "Subject", "Mode", *METRIC_LABELS, "N", "Status"], [metric_row(r, [r["variant"], r["cohort"], r["mode"]]) for r in fixed])
    known_keys = ("keyframe_position_error", "keyframe_rotation_error_deg", "hidden_position_error", "hidden_rotation_error_deg")
    report.table("appendix_keyframe_errors", ["Variant", "Subject", "Mode", *[k for metric in known_keys for k in (metric, metric + "_count")]], [[r["variant"], r["cohort"], r["mode"], *[r["values"].get(k) for metric in known_keys for k in (metric, metric + "_count")]] for r in fixed])
    report.heading("ب — هندسه و کادربندی", 3)
    report.paragraph("موقعیت در واحد جهان دیتاست و زاویه برحسب درجه است. خطاهای کادر نسبت به اندازهٔ تصویر هستند. شاخص خروج از کادر برش عمدی close-up را هم حساب می‌کند؛ بنابراین درصد آن نرخ شکست عمومی نیست. خطاها با وزن فریم معتبر تجمیع شده‌اند؛ شمارش هر معیار در جدول کامل خام موجود است.")
    geometry = ("position_error", "rotation_error_deg", "bbox_size_error", "bbox_center_error", "out_of_frame_rate", "behind_camera_rate", "near_plane_violation_rate", "invalid_generated_pose_rate")
    main_records = sorted([r for r in records.values() if r["variant"] == "main"], key=lambda r: (r["model"], r["cohort"], r["mode"]))
    report.table("appendix_geometry", ["Model", "Subject", "Mode", *geometry], [[MODEL_LABELS.get(r["model"], r["model"]), r["cohort"], r["mode"], *[r["values"].get(k) for k in geometry]] for r in main_records])
    report.heading("پ — نرمال‌سازی و مقداردهی اولیهٔ baselineها", 3)
    report.paragraph("این حالت‌ها تبدیل خروجی‌اند، نه آموزش جداگانه. حالت دارای LensCraft init از موقعیت پیش‌بینی‌شدهٔ LensCraft استفاده می‌کند و فقط در پیوست آمده است؛ موقعیت مسیر را جابه‌جا می‌کند و اصلاح دوران محسوب نمی‌شود.")
    norm = [r for r in main_records if r["model"] != "lens_craft"]
    report.table("appendix_normalization", ["Model", "Subject", "Mode", *METRIC_LABELS, "N", "Status"], [metric_row(r, [MODEL_LABELS.get(r["model"], r["model"]), r["cohort"], r["raw_mode"]]) for r in norm])
    report.heading("ت — سرعت در طول بومی مدل‌ها", 3)
    lengths = training.get("protocol", {}).get("native_lengths", {})
    native_rows = []
    for row in native:
        data, model = row["data"], row["model"]
        if data.get("report_role") == "supplementary":
            continue
        length_key = "lenscraft" if model == "lens_craft" else "et" if model.startswith("et_") else model
        native_rows.append([MODEL_LABELS.get(model, model), row["seq_length"] or lengths.get(length_key), row["batch_size"], *[data.get(k) for k in ("inference_time_batch_s", "inference_time_batch_std_s", "inference_time_per_traj_s", "gflops_per_traj", "n_warmup", "n_runs")]])
    report.table("appendix_native_efficiency", ["Model", "Native frames", "Batch", "Batch mean (s)", "Batch SD (s)", "Trajectory throughput time (s)", "GFLOPs/traj", "Warmups", "Runs"], native_rows)
    historical = [row for row in native if row["data"].get("report_role") == "supplementary"]
    if historical:
        report.heading("اندازه‌گیری‌های تکمیلی با پروتکل پیشین", 3)
        report.paragraph("این اندازه‌گیری‌ها از پروتکل پیشین با طول مشترک ۳۰ فریم نگه‌داری شده‌اند و وارد جدول اصلی هزینه نمی‌شوند. مقادیر FLOPs با شمارش اپراتورهای پشتیبانی‌شده، کران پایین‌اند؛ روش شمارش و سیاست توقف تولید در فایل منبع هر رکورد ثبت شده است. نتایج پروتکل‌های متفاوت را نباید بی‌قید با هم مقایسه کرد.")
        report.table("appendix_previous_efficiency", ["Model", "Frames", "Batch", "Batch mean (s)", "Batch SD (s)", "GFLOPs/traj", "FLOPs scope", "Status"],
                     [[MODEL_LABELS.get(row["model"], row["model"]), row["seq_length"], row["batch_size"],
                       *[row["data"].get(key) for key in ("inference_time_batch_s", "inference_time_batch_std_s", "gflops_per_traj")],
                       "lower bound" if row["data"].get("flop_count_is_lower_bound") else "recorded scope",
                       row["data"].get("status", "measured")] for row in historical])
    report.heading("ث — آموزش و انتخاب checkpoint", 3)
    train_rows = [[key, val.get("count"), *[val.get(stage, {}).get("value") for stage in ("first", "best", "last")]] for key, val in training.get("summary", {}).items()]
    report.table("appendix_training", ["Metric", "Count", "First", "Best recorded", "Last"], train_rows)
    selected_config = {k: v for k, v in resolved_training.items() if k.startswith(("training.loss_module.losses_list.", "training.teacher_forcing_schedule.", "training.noise.", "training.mask.", "training.optimizer.")) or k in ("training.conditioning.enabled", "trainer.max_epochs", "data.batch_size", "data.dataloader.batch_size")}
    report.table("appendix_resolved_training_config", ["Resolved configuration key", "Actual value"], [[k, v] for k, v in selected_config.items()])
    report.paragraph("مدل مبنا با val_conditioning_score انتخاب شده است؛ بهترین val_loss الزاماً checkpoint منتخب نیست. verified_launch.json وضعیت تاریخی زمان آغاز کار است و نشان‌دهندهٔ وضعیت زندهٔ فعلی نیست.")
    provenance_rows, provenance_links, seen_manifests = [], [], set()
    for root in roots:
        if not root.exists():
            continue
        manifests = sorted(set(root.rglob("run_manifest.json")) | set(root.rglob("supplement_manifest.json")))
        for path in manifests:
            if path.resolve() in seen_manifests:
                continue
            seen_manifests.add(path.resolve())
            data = read_json(path)
            checksum = digest(path)
            label = f"{path.parent.name}/{path.name}"
            target = output / "provenance" / f"{path.parent.name}_{checksum[:12]}.json"
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
            provenance_links.append((label, str(target.relative_to(output))))
            provenance_rows.append([label, data.get("declared_base_revision", data.get("git_head", data.get("project_commit"))),
                                    len(data.get("stages", data.get("variants", {}))), checksum])
    if provenance_rows:
        report.table("appendix_provenance", ["Manifest", "Base revision", "Stages / variants", "SHA256"], provenance_rows)
        report.paragraph("فایل‌های کامل منشأ شامل فرمان‌ها، تنظیمات هر آموزش، هش کد و checkpointها از پیوندهای زیر در دسترس‌اند؛ مقادیر کامل آن‌ها در فایل حفظ شده است.")
        report.links(provenance_links)
    report.heading("ج — همهٔ معیارها، عدم‌قطعیت و منشأ", 3)
    report.paragraph("فایل all_metrics.csv تمام مقادیر، شمارش فریم‌ها، میانگین و انحراف معیار bootstrap و مسیر منبع را با دقت کامل ذخیره می‌کند. انحراف معیار bootstrap فاصلهٔ اطمینان ۹۵٪ نیست و میانگین bootstrap ممکن است با برآورد اصلی تفاوت داشته باشد. جدول زیر معیارهای توزیعی تمام حالت‌ها و نسخه‌ها را ارائه می‌کند؛ مقدار خام و میانگین bootstrap به‌صورت جدا ذخیره شده‌اند.")
    all_records = sorted(records.values(), key=lambda r: (r["variant"], r["model"], r["cohort"], r["mode"]))
    report.table("appendix_all_distribution", ["Model", "Variant", "Subject", "Mode", *METRIC_LABELS, "N", "Status"], [metric_row(r, [r["model"], r["variant"], r["cohort"], r["mode"]]) for r in all_records])
    boot_rows = []
    long_rows = []
    for row in all_records:
        for key, value in sorted(row["values"].items()):
            bootstrap = row["bootstrap"].get(key, [])
            mean, std = bootstrap if isinstance(bootstrap, list) and len(bootstrap) == 2 else (None, None)
            if std is not None:
                boot_rows.append([row["model"], row["variant"], row["cohort"], row["mode"], key, value, mean, std])
            long_rows.append([row["model"], row["variant"], row["cohort"], row["mode"], key, value, mean, std, value * 100 if key == "clip_score" and value is not None else None, row["n"], " | ".join(row["sources"])])
    report.table("appendix_bootstrap", ["Model", "Variant", "Subject", "Mode", "Metric", "Point estimate", "Bootstrap mean", "Bootstrap SD"], boot_rows)
    report.render(output)
    write_csv(output / "all_metrics.csv", ["model", "variant", "cohort", "mode", "metric", "value", "bootstrap_mean", "bootstrap_std", "cs_times_100_display_only", "sample_count", "source_files"], long_rows)
    summary = {"schema_version": 1, "protocol": protocol, "completion": completeness, "tables": report.tables,
               "resolved_main_training_config": selected_config,
               "measurements": all_records, "human_study": human, "qualitative_figures": figures,
               "metric_display": {"fcd": "FID/FCD in CLaTr feature space", "clip_score": "native scale, no multiplication", "cs_times_100": "display-only derived column in all_metrics.csv"}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for source in sources.values():
        if digest(source["path"]) != source["sha256"]:
            raise ValueError(f"Input changed during report build; retry with a stable snapshot: {source['path']}")
    manifest = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "generator": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
                "inputs": sorted(sources.values(), key=lambda r: r["path"]), "input_roots": [str(p.resolve()) for p in roots],
                "validated_recorded_output_hashes": checked_hashes, "evaluator_fingerprints": protocol["evaluator_fingerprints"], "completion": completeness,
                "outputs": [{"path": str(p.relative_to(output)), "sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file() and p.name != "report_manifest.json" and p != output / "REPORT.pdf"],
                "pdf_note": "REPORT.pdf is an optional separately rendered snapshot; this generator does not create or validate it."}
    (output / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-run-dir", type=Path, required=True)
    p.add_argument("--extra-results-dir", type=Path, action="append", default=[])
    p.add_argument("--ablation-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--paper-pdf", type=Path)
    p.add_argument("--human-study-csv", type=Path)
    return p


def main():
    args = parser().parse_args()
    try:
        summary = build_report(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Report validation failed: {exc}") from exc
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "completion": summary["completion"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
