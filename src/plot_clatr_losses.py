"""Plot native CLaTr Lightning CSV logs without importing PyTorch or using a GPU."""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import math
import os
from pathlib import Path
import time


TERMS = (("Total loss", "loss"), ("Reconstruction", "recons"),
         ("Contrastive (InfoNCE)", "contrastive"), ("Latent alignment", "latent"),
         ("KL divergence", "kl"))
EPOCH_KEYS = tuple(key for _, term in TERMS for key in
                   (f"train/{term}_epoch", f"val/{term}"))
STEP_KEYS = tuple(f"train/{term}_step" for _, term in TERMS)


def _clatr_header(fields):
    return bool(fields and "epoch" in fields and
                {"train/loss_epoch", "train/loss_step", "val/loss"}.intersection(fields))


def find_metrics_csv(source: Path) -> Path:
    """Choose one native CLaTr logger version; never mix it with LensCraft logs."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    search_root = source / "clatr_native" if (source / "clatr_native").is_dir() else source
    candidates = [source] if source.is_file() else search_root.rglob("metrics.csv")
    matches = []
    for candidate in candidates:
        with candidate.open(newline="", encoding="utf-8-sig") as stream:
            if _clatr_header(next(csv.reader(stream), None)):
                matches.append(candidate)
    if not matches:
        raise ValueError(f"No native CLaTr CSV (epoch and train/loss or val/loss columns) found: {source}")
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _number(raw, key, line, *, integer=False):
    value = float(raw)
    if not math.isfinite(value) or (integer and (value < 0 or not value.is_integer())):
        raise ValueError(f"Invalid {key} at CSV line {line}: {raw!r}")
    return int(value) if integer else value


def read_training_log(path: Path) -> dict:
    """Merge sparse epoch rows, keeping step observations separate from averages."""
    before = path.stat()
    contents = path.read_text(encoding="utf-8-sig")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ValueError("CSV changed while being read; retry this snapshot")
    # A running logger may have written only part of its final row.
    if contents and not contents.endswith(("\n", "\r")):
        contents = contents.rsplit("\n", 1)[0] + "\n"
    reader = csv.DictReader(io.StringIO(contents))
    if not _clatr_header(reader.fieldnames):
        raise ValueError(f"Not a native CLaTr metrics CSV: {path}")
    epochs, rates, latest_step = {}, {}, None
    for line, row in enumerate(reader, start=2):
        if row.get("lr-AdamW") and row.get("step"):
            step = _number(row["step"], "step", line, integer=True)
            rates[step] = _number(row["lr-AdamW"], "lr-AdamW", line)
        if not row.get("epoch"):
            continue
        epoch = _number(row["epoch"], "epoch", line, integer=True)
        values = epochs.setdefault(epoch, {})
        for key in EPOCH_KEYS:
            if row.get(key):
                values[key] = _number(row[key], key, line)
        observations = {key: _number(row[key], key, line)
                        for key in STEP_KEYS if row.get(key)}
        if "train/loss_step" in observations and row.get("step"):
            latest_step = {"epoch": epoch,
                           "step": _number(row["step"], "step", line, integer=True),
                           **observations}
    return {
        "epochs": {epoch: values for epoch, values in sorted(epochs.items())
                   if "train/loss_epoch" in values and "val/loss" in values},
        "latest_step": latest_step,
        "learning_rates": [{"step": step, "lr": rates[step]} for step in sorted(rates)],
    }


def plot_training(data: dict, output: Path, title: str, captured_at: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = data["epochs"]
    numbers = list(epochs)
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#cbd5e1", "text.color": "#172b4d",
             "axes.labelcolor": "#475569", "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(14, 10.5))
        try:
            for ax, (label, term) in zip(axes.flat, TERMS):
                latest = []
                for name, key, color in (("Train", f"train/{term}_epoch", "#2563eb"),
                                         ("Validation", f"val/{term}", "#e76f24")):
                    values = [epochs[e].get(key, math.nan) for e in numbers]
                    if any(math.isfinite(value) for value in values):
                        ax.plot(numbers, values, color=color, linewidth=2, marker=".", label=name)
                    if values and math.isfinite(values[-1]):
                        latest.append(f"{name}: {values[-1]:.5g}")
                ax.set_title(label, loc="left", fontweight="bold")
                ax.set_xlabel("Epoch (zero-based)")
                ax.set_ylabel("Loss")
                if ax.lines:
                    ax.legend(frameon=False, fontsize=9)
                else:
                    message = "Waiting for a completed epoch" if not epochs else "Not logged"
                    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center")
                ax.text(0.98, 0.96, "  |  ".join(latest), transform=ax.transAxes,
                        ha="right", va="top", fontsize=9, color="#475569")
            rate_ax = axes[2, 1]
            rates = data["learning_rates"]
            if rates:
                rate_ax.plot([r["step"] for r in rates], [r["lr"] for r in rates],
                             color="#059669", linewidth=2)
                rate_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            else:
                rate_ax.text(0.5, 0.5, "Not logged", transform=rate_ax.transAxes, ha="center")
            rate_ax.set(title="Learning rate", xlabel="Global step", ylabel="AdamW learning rate")
            for ax in axes.flat:
                ax.set_facecolor("#f8fafc")
                ax.grid(axis="y", color="#dce3ed", linewidth=0.7)
                ax.set_axisbelow(True)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
                ax.margins(x=0.04, y=0.2)
            details = f"{len(epochs)} completed epochs"
            if epochs:
                best = min(epochs, key=lambda e: epochs[e]["val/loss"])
                value = epochs[best]["val/loss"]
                axes[0, 0].scatter([best], [value], marker="*", color="#e76f24",
                                   edgecolor="white", s=170, zorder=5)
                details += f" | Best val: {value:.5g} (epoch {best})"
            if data["latest_step"]:
                step = data["latest_step"]
                details += f" | Latest logged step: {step['step']} (epoch {step['epoch']})"
            fig.suptitle(title, x=0.07, y=0.98, ha="left", fontsize=19, fontweight="bold")
            fig.text(0.07, 0.946, details, fontsize=10, color="#475569")
            fig.text(0.07, 0.922, f"Snapshot: {captured_at}", fontsize=9, color="#64748b")
            fig.subplots_adjust(top=0.855, bottom=0.09, left=0.075, right=0.975,
                                hspace=0.55, wspace=0.22)
            fig.text(0.075, 0.028, "Loss panels: completed epoch averages, no smoothing; "
                     "components are unweighted. Partial epochs and step losses are excluded.",
                     fontsize=9, color="#64748b")
            for extension in ("png", "svg"):
                destination = output / f"clatr_loss_trends.{extension}"
                temporary = output / f".clatr_loss_trends.{os.getpid()}.tmp.{extension}"
                fig.savefig(temporary, dpi=160, facecolor="white")
                temporary.replace(destination)
        finally:
            plt.close(fig)


def _atomic_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def save_snapshot(metrics: Path, output: Path, title: str) -> dict:
    data = read_training_log(metrics)
    captured_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    output.mkdir(parents=True, exist_ok=True)
    plot_training(data, output, title, captured_at)
    epochs = data["epochs"]
    best = min(epochs, key=lambda e: epochs[e]["val/loss"]) if epochs else None
    summary = {"source": str(metrics), "captured_at": captured_at,
               "completed_epochs": len(epochs), "last_epoch": max(epochs) if epochs else None,
               "best_epoch": best, "best_val": epochs[best]["val/loss"] if best is not None else None,
               "latest_step": data["latest_step"],
               "last": epochs[max(epochs)] if epochs else None}
    _atomic_text(output / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _atomic_text(output / "epoch_losses.json", json.dumps(
        {"source": str(metrics), "captured_at": captured_at, **data}, indent=2, allow_nan=False) + "\n")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["epoch", *EPOCH_KEYS])
    writer.writeheader()
    for epoch, values in epochs.items():
        writer.writerow({"epoch": epoch, **values})
    _atomic_text(output / "epoch_losses.csv", stream.getvalue())
    print(f"[{captured_at}] Completed epochs: {len(epochs)}; "
          f"best val: {summary['best_val']} (epoch {best}); output: {output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Run directory, clatr_native directory, or metrics.csv")
    parser.add_argument("--output-dir", type=Path, help="Reuse/update this output directory")
    parser.add_argument("--title", default="Native CLaTr | Training loss trends")
    parser.add_argument("--watch", type=float, metavar="SECONDS",
                        help="Keep updating the same files when the CSV changes; Ctrl+C stops")
    args = parser.parse_args()
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch < 1):
        parser.error("--watch must be a finite interval of at least one second")
    try:
        # Pin one logger version for this process, including watch mode.
        metrics = find_metrics_csv(args.input)
        source = args.input.expanduser().resolve()
        parent = source.parent if source.is_file() else source
        output = (args.output_dir.expanduser().resolve() if args.output_dir else
                  parent / datetime.datetime.now().strftime("clatr_analysis_%Y%m%d_%H%M%S_%f"))
        print(f"Source (one logger version): {metrics}", flush=True)
        signature = None
        while True:
            try:
                stat = metrics.stat()
                current = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
                if current != signature:
                    save_snapshot(metrics, output, args.title)
                    signature = current
            except (OSError, ValueError) as error:
                if args.watch is None:
                    raise
                print(f"Snapshot not updated: {error}; retrying on next poll", flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nStopped watching; the last saved plots remain available.")
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
