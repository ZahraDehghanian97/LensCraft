"""Plot epoch-averaged LensCraft losses from Lightning CSV logs (CPU only)."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path


TERMS = (
    ("Total loss", "train_loss_epoch", "val_loss"),
    ("Cycle loss", "train_cycle_epoch", "val_cycle_epoch"),
    ("CLIP loss", "train_clip_epoch", "val_clip_epoch"),
    ("First-frame loss", "train_first_frame_epoch", "val_first_frame_epoch"),
    ("Relative-motion loss", "train_relative_epoch", "val_relative_epoch"),
    ("Speed loss", "train_speed_epoch", "val_speed_epoch"),
)
METRIC_KEYS = tuple(key for _, train, val in TERMS for key in (train, val))


def find_metrics_csv(source: Path) -> Path:
    """Select one CSV, preferring a run's train directory over evaluation logs."""
    source = source.expanduser().resolve()
    if source.is_file():
        return source
    if not source.is_dir():
        raise ValueError(f"Input does not exist: {source}")
    search_root = source / "train" if (source / "train").is_dir() else source
    candidates = list(search_root.rglob("metrics.csv"))
    if not candidates:
        raise ValueError(f"No metrics.csv found under {search_root}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def read_epoch_losses(metrics: Path) -> dict[int, dict[str, float]]:
    """Merge sparse train/validation rows, excluding step losses and partial epochs."""
    epochs: dict[int, dict[str, float]] = {}
    with metrics.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError(f"CSV has no epoch column: {metrics}")
        for line, row in enumerate(reader, start=2):
            if not row.get("epoch"):
                continue
            epoch_number = float(row["epoch"])
            if not math.isfinite(epoch_number) or not epoch_number.is_integer() or epoch_number < 0:
                raise ValueError(f"Invalid epoch at CSV line {line}: {row['epoch']}")
            values = epochs.setdefault(int(epoch_number), {})
            for key in METRIC_KEYS:
                raw = row.get(key)
                if key == "val_loss" and not raw:
                    raw = row.get("val_loss_epoch")
                if raw:
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite {key} at CSV line {line}: {raw}")
                    values[key] = value
    complete = {
        epoch: values for epoch, values in sorted(epochs.items())
        if "train_loss_epoch" in values and "val_loss" in values
    }
    if not complete:
        raise ValueError("No completed epochs with both train_loss_epoch and val_loss yet.")
    return complete


def plot_losses(
    epochs: dict[int, dict[str, float]], output: Path, title: str,
    captured_at: datetime.datetime,
) -> None:
    # Import lazily so --help and CSV parsing do not require the training stack.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    numbers = list(epochs)
    best_epoch = min(numbers, key=lambda e: epochs[e]["val_loss"])
    style = {
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 13, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1", "text.color": "#172b4d",
        "axes.labelcolor": "#475569", "xtick.color": "#64748b",
        "ytick.color": "#64748b", "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(14, 10.5))
        fig.patch.set_facecolor("white")
        try:
            for ax, (label, train_key, val_key) in zip(axes.flat, TERMS):
                ax.set_facecolor("#f8fafc")
                latest = []
                for name, key, color in (
                    ("Train", train_key, "#2563eb"),
                    ("Validation", val_key, "#e76f24"),
                ):
                    values = [epochs[e].get(key, math.nan) for e in numbers]
                    if any(math.isfinite(value) for value in values):
                        ax.plot(numbers, values, color=color, linewidth=2.1, label=name)
                    if math.isfinite(values[-1]):
                        ax.scatter([numbers[-1]], [values[-1]], color=color, s=27, zorder=4)
                        latest.append(f"{name.lower()} {values[-1]:.5g}")
                ax.set_title(label, loc="left", fontweight="bold", pad=10)
                ax.set_xlabel("Epoch (zero-based)")
                ax.set_ylabel("Loss")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(axis="y", color="#dce3ed", linewidth=0.7)
                ax.set_axisbelow(True)
                ax.margins(x=0.03, y=0.13)
                if ax.lines:
                    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
                else:
                    ax.text(0.5, 0.5, "Not logged", transform=ax.transAxes, ha="center")
                if latest:
                    ax.text(
                        0.98, 0.93, "Latest: " + "  |  ".join(latest),
                        transform=ax.transAxes, ha="right", va="top", fontsize=9,
                        color="#475569",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
                    )
            best_val = epochs[best_epoch]["val_loss"]
            axes[0, 0].scatter(
                [best_epoch], [best_val], marker="*", color="#e76f24",
                edgecolor="white", linewidth=0.7, s=145, zorder=5,
            )
            fig.suptitle(title, x=0.055, y=0.982, ha="left", fontsize=18, fontweight="bold")
            fig.text(
                0.055, 0.95,
                f"Completed epochs {numbers[0]}–{numbers[-1]}  •  "
                f"Updated {captured_at:%Y-%m-%d %H:%M %z}  •  "
                f"Best validation: {best_val:.3f} (epoch {best_epoch})",
                fontsize=10.5, color="#64748b",
            )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.955, 0.93),
                       frameon=False, ncol=2)
            fig.subplots_adjust(top=0.875, bottom=0.08, left=0.065, right=0.975,
                                hspace=0.48, wspace=0.20)
            fig.text(
                0.065, 0.024,
                "Epoch averages, no smoothing. Component panels show unweighted losses "
                "on linear scales. Incomplete epochs are excluded.",
                fontsize=9.5, color="#64748b",
            )
            for extension in ("png", "svg"):
                fig.savefig(output / f"loss_trends.{extension}", dpi=180, facecolor="white")
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Run/log directory or a metrics.csv file")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory; default: timestamped analysis folder beside the input")
    parser.add_argument("--title", default="LensCraft | Training loss trends")
    args = parser.parse_args()
    try:
        metrics = find_metrics_csv(args.input)
        epochs = read_epoch_losses(metrics)
        now = datetime.datetime.now().astimezone()
        source = args.input.expanduser().resolve()
        parent = source.parent if source.is_file() else source
        output = (args.output_dir.expanduser().resolve() if args.output_dir else
                  parent / now.strftime("analysis_%Y%m%d_%H%M%S_%f"))
        output.mkdir(parents=True, exist_ok=True)
        print(f"Source: {metrics}")
        plot_losses(epochs, output, args.title, now)
        snapshot = {"source": str(metrics), "captured_at": now.isoformat(), "epochs": epochs}
        (output / "epoch_losses.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        with (output / "epoch_losses.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", *METRIC_KEYS])
            writer.writeheader()
            for epoch, values in epochs.items():
                writer.writerow({"epoch": epoch, **values})
        best_epoch = min(epochs, key=lambda e: epochs[e]["val_loss"])
        summary = {
            "source": str(metrics), "directory": str(output), "captured_at": now.isoformat(),
            "completed_epochs": len(epochs), "last_epoch": max(epochs),
            "best_epoch": best_epoch, "best_val": epochs[best_epoch]["val_loss"],
            "first": epochs[min(epochs)], "last": epochs[max(epochs)],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved plots and epoch data: {output}")
        print(f"Completed epochs: {len(epochs)}; best validation: "
              f"{summary['best_val']:.5g} (epoch {best_epoch})")
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
