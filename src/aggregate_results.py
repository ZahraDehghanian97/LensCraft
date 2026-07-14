from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional

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
SET_ORDER = ["static", "dynamic"]
NO_NORM_MODE = "prompt_generation_no_norm"

MODE_TO_INPUT = {
    "prompt_generation": "P",
    "reconstruction": "ST",
    "key_framing": "KF",
    "key_framing+prompt": "P + KF",
    "hybrid_generation": "P + ST",
}
MODE_ORDER = ["prompt_generation", "reconstruction", "key_framing",
              "key_framing+prompt", "hybrid_generation"]


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
    """SOTA comparison. When a baseline's run also carries the no-normalization
    mode, an extra "(w/o norm)" row is emitted right below its default row."""
    idx = {(r.get("set"), r.get("model_type")): r
           for r in metric_runs if not r.get("variant")}
    rows = []
    for s in SET_ORDER:
        for m in MODEL_ORDER:
            run = idx.get((s, m))
            if run is None:
                continue
            rows.append([s.capitalize(), MODEL_LABEL.get(m, m)]
                        + _row_cells(run, "prompt_generation"))
            if NO_NORM_MODE in run.get("metrics", {}):
                rows.append([s.capitalize(),
                             MODEL_LABEL.get(m, m) + " (w/o norm)"]
                            + _row_cells(run, NO_NORM_MODE))
    return md_table(["Set", "Methods"] + METRIC_COLS, rows)


def build_table2(metric_runs) -> str:
    idx = {r.get("set"): r for r in metric_runs
           if r.get("model_type") == "lens_craft" and not r.get("variant")}
    rows = []
    for s in SET_ORDER:
        run = idx.get(s)
        if run is None:
            continue
        metrics = run.get("metrics", {})
        for mode in MODE_ORDER:
            if mode in metrics:
                rows.append([s.capitalize(), MODE_TO_INPUT[mode]]
                            + _row_cells(run, mode))
    return md_table(["Set", "Input(s)"] + METRIC_COLS, rows)


def build_table3(metric_runs, mode: str) -> str:
    rows = []
    for r in metric_runs:
        if r.get("variant") and r.get("model_type") == "lens_craft":
            rows.append([str(r.get("variant"))] + _row_cells(r, mode))
    return md_table(["Variant"] + METRIC_COLS, rows)


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
    idx = {(r.get("set"), r.get("model_type")): r
           for r in metric_runs if not r.get("variant")}
    rows = []
    for s in SET_ORDER:
        for m in MODEL_ORDER:
            if m == "lens_craft":
                continue
            run = idx.get((s, m))
            if run is None:
                continue
            metrics = run.get("metrics", {})
            if "prompt_generation" in metrics:
                rows.append([s.capitalize(), MODEL_LABEL.get(m, m), "yes"]
                            + _row_cells(run, "prompt_generation"))
            if NO_NORM_MODE in metrics:
                rows.append([s.capitalize(), MODEL_LABEL.get(m, m), "no"]
                            + _row_cells(run, NO_NORM_MODE))
    return md_table(["Set", "Methods", "Sim. norm."] + METRIC_COLS, rows)


def write_flat_csv(metric_runs, path: str) -> None:
    header = ["model", "set", "variant", "mode"]
    for c in METRIC_COLS:
        header += [c, f"{c}_std"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in metric_runs:
            metrics = r.get("metrics", {})
            boot = r.get("bootstrap_std", {})
            for mode in metrics:
                row = [r.get("model_type"), r.get("set"), r.get("variant"), mode]
                for c in METRIC_COLS:
                    suffix = METRIC_SUFFIX[c]
                    row.append(_fmt(_get(metrics, mode, suffix)))
                    row.append(_fmt(_get_std(boot, mode, suffix)))
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

    write_flat_csv(metric_runs, os.path.join(out_dir, "all_metrics.csv"))

    tables = {
        "table1_sota.md": build_table1(metric_runs),
        "table2_multimodal.md": build_table2(metric_runs),
        "table3_ablation.md": build_table3(metric_runs, args.table3_mode),
        "table4_efficiency.md": build_table4(eff_runs),
        "table5_baseline_normalization.md": build_table5(metric_runs),
    }
    for name, content in tables.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(content + "\n")
        print(f"\n=== {name} ===\n{content}")

    print(f"\nFlat dump: {os.path.join(out_dir, 'all_metrics.csv')}")


if __name__ == "__main__":
    main()
