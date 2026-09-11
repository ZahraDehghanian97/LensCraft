from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional

from testing.keyframes import parse_keyframe_mode

METRIC_SUFFIX = {
    "FID": "fcd",
    "P": "precision",
    "R": "recall",
    "D": "density",
    "C": "coverage",
    "CS": "clip_score",
    "Clatr": "clatr_score",
}
METRIC_COLS = ["FID", "P", "R", "D", "C", "CS", "Clatr"]

MODEL_LABEL = {"lens_craft": "LensCraft", "ccdm": "CCDM", "et": "E.T.", "gendop": "GenDoP"}
MODEL_ORDER = ["lens_craft", "ccdm", "et", "gendop"]
SET_ORDER = ["all", "static", "dynamic"]
NO_NORM_MODE = "prompt_generation_no_norm"
NORM_MODE = "prompt_generation"
NORM_LENSCRAFT_INIT_MODE = "prompt_generation_norm_lenscraft_init"

BASELINE_MODE_SPECS = (
    (NO_NORM_MODE, "no norm, no init", "no", "none"),
    (NORM_MODE, "norm, no init", "yes", "none"),
    (
        NORM_LENSCRAFT_INIT_MODE,
        "norm + LensCraft init",
        "yes",
        "LensCraft",
    ),
)

MODE_TO_INPUT = {
    "prompt_generation": "P",
    "reconstruction": "ST",
    "key_framing": "KF",
    "key_framing+prompt": "P + KF",
    "hybrid_generation": "P + ST",
}
MODE_ORDER = ["prompt_generation", "reconstruction", "key_framing",
              "key_framing+prompt", "hybrid_generation"]

GEOMETRY_METRICS = (
    "position_error", "rotation_error_deg",
    "keyframe_position_error", "keyframe_rotation_error_deg",
    "hidden_position_error", "hidden_rotation_error_deg",
    "invalid_generated_pose_rate",
    "bbox_size_error", "bbox_center_error",
    "subject_bbox_width", "subject_bbox_height",
    "subject_bbox_center_x", "subject_bbox_center_y",
    "out_of_frame_rate", "behind_camera_rate", "near_plane_violation_rate",
)


def validate_evaluator_provenance(metric_runs) -> Dict[str, str]:
    """Prevent comparative tables from silently combining different scorers."""
    identities = {}
    for evaluator in ("semantic_evaluator", "clatr_evaluator"):
        fingerprints = []
        for run in metric_runs:
            provenance = run.get("evaluation_provenance") or {}
            identity = provenance.get(evaluator)
            if identity is not None and (
                not isinstance(identity, dict)
                or not isinstance(identity.get("fingerprint"), str)
                or not identity["fingerprint"]
            ):
                raise ValueError(f"Invalid {evaluator} fingerprint in evaluation provenance")
            fingerprints.append(identity["fingerprint"] if identity is not None else None)
        recorded = {fingerprint for fingerprint in fingerprints if fingerprint is not None}
        if recorded and None in fingerprints:
            raise ValueError(
                f"Cannot compare modern and legacy results: {evaluator} fingerprint "
                "is missing from some runs. Re-evaluate them with the same fixed evaluator."
            )
        if len(recorded) > 1:
            raise ValueError(
                f"Cannot compare different {evaluator} fingerprints. "
                "Re-evaluate all runs with the same fixed evaluator."
            )
        if recorded:
            identities[evaluator] = next(iter(recorded))
    if identities and "semantic_evaluator" not in identities:
        raise ValueError("Modern evaluation results must record a semantic_evaluator fingerprint")
    return identities


def _provenance_note(metric_runs) -> str:
    identities = validate_evaluator_provenance(metric_runs)
    if not metric_runs:
        return ""
    if not identities:
        return (
            "\n\nLegacy results: evaluator identities were not recorded; "
            "semantic-score comparability cannot be verified."
        )
    return "\n\nFixed evaluator fingerprints: " + "; ".join(
        f"{name}: `{fingerprint}`" for name, fingerprint in identities.items()
    ) + "."


def _set_name(run) -> str:
    return str(run.get("set") or "all").lower()


def _set_names(metric_runs):
    present = {_set_name(run) for run in metric_runs}
    return [name for name in SET_ORDER if name in present] + sorted(present - set(SET_ORDER))


def _ordered_runs(metric_runs):
    for set_name in _set_names(metric_runs):
        yield from sorted(
            (run for run in metric_runs if _set_name(run) == set_name),
            key=lambda run: (
                MODEL_ORDER.index(run["model_type"])
                if run.get("model_type") in MODEL_ORDER else len(MODEL_ORDER),
                str(run.get("model_type") or ""),
                str(run.get("et_type") or ""),
                str(run.get("variant") or ""),
            ),
        )


def _method_label(run):
    model = run.get("model_type", "unknown")
    label = MODEL_LABEL.get(model, model)
    return f"{label} ({run['et_type']})" if run.get("et_type") else label


def _input_label(mode):
    base_mode, count = parse_keyframe_mode(mode)
    label = MODE_TO_INPUT.get(base_mode, base_mode)
    return f"{label} (K={count})" if count is not None and mode != base_mode else label


def _requested_keyframes(run, mode):
    base_mode, count = parse_keyframe_mode(mode)
    if count is None:
        return None
    if mode != base_mode:
        return count
    # Legacy unsuffixed modes used a different count; never infer K=4 for them.
    return (run.get("keyframe_protocol") or {}).get("num_keyframes")


def _ordered_modes(metrics):
    def key(mode):
        base_mode, count = parse_keyframe_mode(mode)
        return (
            MODE_ORDER.index(base_mode) if base_mode in MODE_ORDER else len(MODE_ORDER),
            count or 0,
            mode,
        )
    return sorted(metrics, key=key)


def _fmt(v: Optional[float], nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _fmt_pm(mean: Optional[float], std: Optional[float], nd: int = 3) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def _get(metrics: Dict[str, Dict[str, float]], mode: str, suffix: str) -> Optional[float]:
    block = metrics.get(mode)
    if not block:
        return None
    return block.get(f"{mode}/{suffix}")


def _get_std(boot: Dict[str, Any], mode: str, suffix: str) -> Optional[float]:
    if not boot:
        return None
    block = boot.get(mode)
    if not block:
        return None
    entry = block.get(f"{mode}/{suffix}")
    if entry is None:
        return None
    try:
        return float(entry[1])
    except (TypeError, IndexError, ValueError):
        return None


def _row_cells(run: Dict[str, Any], mode: str) -> List[str]:
    metrics = run.get("metrics", {})
    boot = run.get("bootstrap_std", {})
    return [
        _fmt_pm(_get(metrics, mode, METRIC_SUFFIX[c]),
                _get_std(boot, mode, METRIC_SUFFIX[c]))
        for c in METRIC_COLS
    ]


def _load(results_dir: str, prefix: str) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json"))):
        try:
            with open(path) as f:
                runs.append(json.load(f))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: could not read {path}: {exc}")
    return runs


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_table1(metric_runs) -> str:
    """SOTA comparison, including every available baseline result mode."""
    note = _provenance_note(metric_runs)
    idx = {(_set_name(r), r.get("model_type")): r
           for r in metric_runs if not r.get("variant")}
    rows = []
    for s in _set_names(metric_runs):
        for m in MODEL_ORDER:
            run = idx.get((s, m))
            if run is None:
                continue
            metrics = run.get("metrics", {})
            label = MODEL_LABEL.get(m, m)
            if m == "lens_craft":
                if NORM_MODE in metrics:
                    rows.append([s.capitalize(), label]
                                + _row_cells(run, NORM_MODE))
                continue
            for mode, mode_label, _normalized, _initial_position in BASELINE_MODE_SPECS:
                if mode in metrics:
                    rows.append([s.capitalize(), f"{label} ({mode_label})"]
                                + _row_cells(run, mode))
    return md_table(["Set", "Methods"] + METRIC_COLS, rows) + note


def build_table2(metric_runs) -> str:
    note = _provenance_note(metric_runs)
    rows = []
    for run in _ordered_runs(metric_runs):
        if run.get("model_type") != "lens_craft" or run.get("variant"):
            continue
        metrics = run.get("metrics", {})
        for mode in _ordered_modes(metrics):
            base_mode, _count = parse_keyframe_mode(mode)
            if base_mode in MODE_ORDER:
                rows.append([_set_name(run).capitalize(), _input_label(mode)]
                            + _row_cells(run, mode))
    return md_table(["Set", "Input(s)"] + METRIC_COLS, rows) + note


def build_table3(metric_runs, mode: str) -> str:
    note = _provenance_note(metric_runs)
    rows = []
    for r in metric_runs:
        if r.get("variant") and r.get("model_type") == "lens_craft":
            rows.append([str(r.get("variant"))] + _row_cells(r, mode))
    return md_table(["Variant"] + METRIC_COLS, rows) + note


def build_table4(eff_runs) -> str:
    by_model = {r.get("model_type"): r for r in eff_runs}
    rows = []
    for m in MODEL_ORDER:
        r = by_model.get(m)
        if r is None:
            continue
        per_traj = r.get("inference_time_per_traj_s")
        per_traj_std = r.get("inference_time_per_traj_std_s")
        if per_traj_std is None:
            batch_std = r.get("inference_time_batch_std_s")
            bs = r.get("batch_size")
            if batch_std is not None and bs:
                per_traj_std = batch_std / bs
        rows.append([MODEL_LABEL.get(m, m),
                     _fmt_pm(per_traj, per_traj_std, 4),
                     _fmt(r.get("gflops_per_traj"), 2)])
    return md_table(["Model", "Inference Time (s)", "FLOPs (G)"], rows)


def build_table5(metric_runs) -> str:
    note = _provenance_note(metric_runs)
    idx = {(_set_name(r), r.get("model_type")): r
           for r in metric_runs if not r.get("variant")}
    rows = []
    for s in _set_names(metric_runs):
        for m in MODEL_ORDER:
            if m == "lens_craft":
                continue
            run = idx.get((s, m))
            if run is None:
                continue
            metrics = run.get("metrics", {})
            for mode, _mode_label, normalized, initial_position in BASELINE_MODE_SPECS:
                if mode in metrics:
                    rows.append([
                        s.capitalize(),
                        MODEL_LABEL.get(m, m),
                        normalized,
                        initial_position,
                    ] + _row_cells(run, mode))
    return md_table(
        ["Set", "Methods", "Sim. norm.", "Initial position"] + METRIC_COLS,
        rows,
    ) + note


def build_keyframe_table(metric_runs) -> str:
    """Report constrained-frame compliance separately from hidden-frame quality."""
    note = _provenance_note(metric_runs)
    rows = []
    geometry = (
        "keyframe_position_error", "keyframe_rotation_error_deg",
        "hidden_position_error", "hidden_rotation_error_deg",
    )
    for run in _ordered_runs(metric_runs):
        metrics = run.get("metrics", {})
        for mode in _ordered_modes(metrics):
            base_mode, count = parse_keyframe_mode(mode)
            if count is None:
                continue
            requested = _requested_keyframes(run, mode)
            row = [
                _set_name(run).capitalize(), _method_label(run),
                str(run.get("variant") or "-"), MODE_TO_INPUT[base_mode],
                str(requested) if requested is not None else "-",
            ]
            row += [_fmt(_get(metrics, mode, suffix)) for suffix in geometry]
            row += [
                _fmt(_get(metrics, mode, f"{suffix}_count"), 0)
                for suffix in ("keyframe_position_error", "hidden_position_error")
            ]
            row += [
                _fmt_pm(_get(metrics, mode, suffix),
                        _get_std(run.get("bootstrap_std", {}), mode, suffix))
                for suffix in ("clip_score", "clatr_score")
            ]
            rows.append(row)
    return md_table([
        "Set", "Method", "Variant", "Input(s)", "K",
        "Keyframe position", "Keyframe angle (deg)",
        "Hidden position", "Hidden angle (deg)",
        "Keyframe frames", "Hidden frames", "CS", "Clatr",
    ], rows) + (
        "\n\nPosition errors use dataset world units; angles use SO(3) geodesic degrees. "
        "K is requested; actual valid keyframes are capped by sequence length. "
        "Frame counts are the denominators of the position-error means. "
        "A dash denotes an unavailable metric or an unrecorded legacy K."
    ) + note


def build_geometry_table(metric_runs) -> str:
    note = _provenance_note(metric_runs)
    rows = []
    suffixes = (
        "position_error", "rotation_error_deg", "bbox_size_error",
        "bbox_center_error", "out_of_frame_rate", "invalid_generated_pose_rate",
    )
    for run in _ordered_runs(metric_runs):
        metrics = run.get("metrics", {})
        for mode in _ordered_modes(metrics):
            if not any(f"{mode}/{name}" in metrics[mode] for name in GEOMETRY_METRICS):
                continue
            rows.append([
                _set_name(run).capitalize(), _method_label(run),
                str(run.get("variant") or "-"), mode,
            ] + [_fmt(_get(metrics, mode, suffix)) for suffix in suffixes])
    return md_table([
        "Set", "Method", "Variant", "Mode", "Position error", "Angle error (deg)",
        "BBox size error", "BBox center error", "Out-of-frame rate", "Invalid pose rate",
    ], rows) + (
        "\n\nPosition errors use dataset world units. BBox errors are fractions of "
        "image dimensions; rates lie in [0, 1]. Projection uses shared reference "
        "intrinsics. Missing calibration is reported as unavailable (-). "
        "All metric denominators and additional geometry measures are in all_metrics.csv."
    ) + note


def write_flat_csv(metric_runs, path: str) -> None:
    identities = validate_evaluator_provenance(metric_runs)
    header = [
        "model", "set", "variant", "mode", "et_type", "requested_keyframes",
        "keyframe_seed", "evaluator_status", "semantic_evaluator_fingerprint",
        "clatr_evaluator_fingerprint",
    ]
    for c in METRIC_COLS:
        header += [c, f"{c}_std"]
    for suffix in GEOMETRY_METRICS:
        header += [suffix, f"{suffix}_count"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in _ordered_runs(metric_runs):
            metrics = r.get("metrics", {})
            boot = r.get("bootstrap_std", {})
            for mode in _ordered_modes(metrics):
                row = [
                    r.get("model_type"), _set_name(r), r.get("variant"), mode,
                    r.get("et_type"), _requested_keyframes(r, mode),
                    (r.get("keyframe_protocol") or {}).get("seed"),
                    "fixed" if identities else "legacy_unverified",
                    identities.get("semantic_evaluator"),
                    identities.get("clatr_evaluator"),
                ]
                for c in METRIC_COLS:
                    suffix = METRIC_SUFFIX[c]
                    row.append(_fmt(_get(metrics, mode, suffix)))
                    row.append(_fmt(_get_std(boot, mode, suffix)))
                for suffix in GEOMETRY_METRICS:
                    # Preserve full numeric precision for downstream analysis.
                    row.append(_get(metrics, mode, suffix))
                    row.append(_get(metrics, mode, f"{suffix}_count"))
                w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="table_results")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--table3-mode", default="prompt_generation",
                    help="which generation mode to report per ablation variant")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)

    metric_runs = _load(args.results_dir, "metrics_")
    eff_runs = _load(args.results_dir, "efficiency_")

    validate_evaluator_provenance(metric_runs)
    write_flat_csv(metric_runs, os.path.join(out_dir, "all_metrics.csv"))

    tables = {
        "table1_sota.md": build_table1(metric_runs),
        "table2_multimodal.md": build_table2(metric_runs),
        "table3_ablation.md": build_table3(metric_runs, args.table3_mode),
        "table4_efficiency.md": build_table4(eff_runs),
        "table5_baseline_normalization.md": build_table5(metric_runs),
        "table6_keyframes.md": build_keyframe_table(metric_runs),
        "table7_geometry.md": build_geometry_table(metric_runs),
    }
    for name, content in tables.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(content + "\n")
        print(f"\n=== {name} ===\n{content}")

    print(f"\nFlat dump: {os.path.join(out_dir, 'all_metrics.csv')}")


if __name__ == "__main__":
    main()
