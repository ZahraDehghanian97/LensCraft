"""CLATR progress parsing uses native sparse Lightning logs without torch."""

import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_clatr_losses import find_metrics_csv, read_training_log


TERMS = ("loss", "recons", "contrastive", "latent", "kl")
FIELDS = ["epoch", "step", "lr-AdamW"] + [
    key
    for term in TERMS
    for key in (f"train/{term}_step", f"train/{term}_epoch", f"val/{term}")
]


class CLATRProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def write_csv(self, relative, rows=(), fields=None, modified=None):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        if modified is not None:
            os.utime(path, (modified, modified))
        return path

    def test_find_newest_clatr_csv_ignores_newer_lenscraft_csv(self):
        self.write_csv("clatr/version_0/metrics.csv", modified=100)
        newest = self.write_csv("clatr/version_1/metrics.csv", modified=200)
        self.write_csv(
            "lenscraft/version_2/metrics.csv",
            fields=["epoch", "step", "train_loss_epoch", "val_loss"],
            modified=300,
        )
        self.assertEqual(find_metrics_csv(self.root), newest)

    def test_find_prefers_native_clatr_subdirectory(self):
        native = self.write_csv("clatr_native/csv/version_0/metrics.csv", modified=100)
        self.write_csv("pilot128/clatr/version_1/metrics.csv", modified=200)
        self.assertEqual(find_metrics_csv(self.root), native)

    def test_find_accepts_explicit_clatr_csv(self):
        path = self.write_csv("saved-progress.csv")
        self.assertEqual(find_metrics_csv(path), path)

    def test_find_recognizes_step_only_and_validation_only_headers(self):
        for metric in ("train/loss_step", "val/loss", "train/loss_epoch"):
            with self.subTest(metric=metric):
                path = self.write_csv("metrics.csv", fields=["epoch", metric])
                self.assertEqual(find_metrics_csv(path), path)

    def test_find_rejects_other_logs_and_missing_input(self):
        for fields in (
            ["epoch", "train_loss_epoch", "val_loss"],
            ["step", "train/loss_step"],
            ["epoch", "step"],
        ):
            with self.subTest(fields=fields):
                path = self.write_csv("metrics.csv", fields=fields)
                with self.assertRaises(ValueError):
                    find_metrics_csv(path)
                with self.assertRaises(ValueError):
                    find_metrics_csv(self.root)
        with self.assertRaises(ValueError):
            find_metrics_csv(self.root / "missing")

    def test_sparse_rows_merge_without_step_losses_replacing_averages(self):
        train = {f"train/{term}_epoch": 2.0 + index for index, term in enumerate(TERMS)}
        validation = {f"val/{term}": 1.0 + index for index, term in enumerate(TERMS)}
        path = self.write_csv(
            "metrics.csv",
            [
                {"step": 0, "lr-AdamW": 0.0001},
                {"epoch": "0.0", "step": 49, "train/loss_step": 9.0},
                {"epoch": 0, "step": 99, **validation},
                {"epoch": 0, "step": 99, **train},
                {"epoch": 1, "step": 149, "train/loss_step": 0.8},
                {"epoch": 1, "step": 199, "val/loss": 0.7},
                {"epoch": 2, "step": 249, "train/loss_epoch": 0.6},
            ],
        )
        result = read_training_log(path)
        self.assertEqual(result["epochs"], {0: {**train, **validation}})
        self.assertEqual(result["latest_step"]["epoch"], 1)
        self.assertEqual(result["latest_step"]["step"], 149)
        self.assertEqual(result["latest_step"]["train/loss_step"], 0.8)
        self.assertEqual(result["learning_rates"], [{"step": 0, "lr": 0.0001}])

    def test_duplicate_sparse_rows_keep_latest_value_and_preserve_other_terms(self):
        path = self.write_csv(
            "metrics.csv",
            [
                {"epoch": 0, "step": 99, "train/loss_epoch": 2.0, "train/kl_epoch": 3.0},
                {"epoch": 0, "step": 99, "val/loss": 1.0, "val/kl": 4.0},
                {"epoch": 0, "step": 99, "val/loss": 0.9},
                {"epoch": 0, "step": 99, "train/loss_epoch": 1.5},
                {"epoch": 0, "step": 99, "train/loss_step": 0.01},
            ],
        )
        self.assertEqual(
            read_training_log(path)["epochs"][0],
            {"train/loss_epoch": 1.5, "train/kl_epoch": 3.0, "val/loss": 0.9, "val/kl": 4.0},
        )

    def test_header_only_log_is_valid_before_training_starts(self):
        result = read_training_log(self.write_csv("metrics.csv"))
        self.assertEqual(result["epochs"], {})
        self.assertFalse(result["latest_step"])
        self.assertEqual(result["learning_rates"], [])

    def test_partial_run_retains_latest_step_without_inventing_complete_epoch(self):
        path = self.write_csv(
            "metrics.csv",
            [
                {"epoch": 0, "step": 49, "train/loss_step": 2.0},
                {"epoch": 0, "step": 99, "train/loss_step": 1.0},
                {"epoch": 0, "step": 99, "val/loss": 0.8},
            ],
        )
        result = read_training_log(path)
        self.assertEqual(result["epochs"], {})
        self.assertEqual(result["latest_step"]["epoch"], 0)
        self.assertEqual(result["latest_step"]["step"], 99)
        self.assertEqual(result["latest_step"]["train/loss_step"], 1.0)

    def test_learning_rates_include_zero_and_rows_without_epochs(self):
        path = self.write_csv(
            "metrics.csv",
            [
                {"step": 0, "lr-AdamW": 0.0001},
                {"epoch": 0, "step": 49, "train/loss_step": 1.5},
                {"step": 100, "lr-AdamW": 0.0},
            ],
        )
        self.assertEqual(
            read_training_log(path)["learning_rates"],
            [{"step": 0, "lr": 0.0001}, {"step": 100, "lr": 0.0}],
        )

    def test_rejects_invalid_epoch_indices(self):
        for epoch in ("nan", "inf", "-inf", "-1", "0.5", "oops"):
            with self.subTest(epoch=epoch):
                path = self.write_csv("metrics.csv", [{"epoch": epoch, "step": 1, "train/loss_step": 1.0}])
                with self.assertRaises(ValueError):
                    read_training_log(path)

    def test_rejects_nonfinite_tracked_values_including_partial_epochs(self):
        for key in ("train/loss_epoch", "val/loss", "train/loss_step", "train/kl_epoch", "val/recons", "lr-AdamW"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(metric=key, value=value):
                    path = self.write_csv("metrics.csv", [{"epoch": 0, "step": 1, key: value}])
                    with self.assertRaises(ValueError):
                        read_training_log(path)


if __name__ == "__main__":
    unittest.main()
