"""Regression checks for manifold coverage of real and generated features."""

from math import dist
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from testing.metrics.modules.prdc import ManifoldMetrics


def reference_prdc(real, fake, k):
    """Small, independent oracle using pointwise distances and explicit balls."""
    def radii(points):
        return [
            sorted(dist(point, other) for j, other in enumerate(points) if j != i)[k - 1]
            for i, point in enumerate(points)
        ]

    real_radii, fake_radii = radii(real), radii(fake)
    generated_memberships = [
        sum(dist(point, center) < radius for center, radius in zip(real, real_radii))
        for point in fake
    ]
    precision = sum(count > 0 for count in generated_memberships) / len(fake)
    recall = sum(
        any(dist(point, center) < radius for center, radius in zip(fake, fake_radii))
        for point in real
    ) / len(real)
    density = sum(generated_memberships) / (k * len(fake))
    coverage = sum(
        any(dist(center, point) < radius for point in fake)
        for center, radius in zip(real, real_radii)
    ) / len(real)
    return precision, recall, density, coverage


class ManifoldMetricsTests(unittest.TestCase):
    REAL = [[value] for value in (7, 16, 19, 0, 17)]
    FAKE = [[value] for value in (6, 22, 20, 17, 13)]

    def assert_metrics_equal(self, actual, expected):
        for name, value, reference in zip(
            ("precision", "recall", "density", "coverage"), actual, expected
        ):
            with self.subTest(metric=name):
                self.assertAlmostEqual(float(value), reference, places=7)

    def compute_prdc(self, real, fake, k=3):
        return ManifoldMetrics(distance="euclidean").compute_prdc(
            torch.tensor(real, dtype=torch.float64),
            torch.tensor(fake, dtype=torch.float64),
            nearest_k=k,
        )

    def test_recall_uses_radius_of_each_generated_center(self):
        # The ball centered at fake=6 has radius 14 and covers real=0.
        # Broadcasting fake radii over real rows instead loses that point.
        self.assert_metrics_equal(
            self.compute_prdc(self.REAL, self.FAKE), (1.0, 1.0, 16 / 15, 1.0)
        )

    def test_rectangular_manifolds_match_pointwise_oracle(self):
        real = [[0, 0], [1, 2], [4, 1], [10, -2], [20, 4]]
        fake = [[-3, 1], [0, 0], [2, 1], [3, 5], [9, -1], [11, 0], [40, 8]]
        for reference, generated in ((real, fake), (fake, real)):
            for k in (1, 3):
                with self.subTest(real_count=len(reference), fake_count=len(generated), k=k):
                    self.assert_metrics_equal(
                        self.compute_prdc(reference, generated, k),
                        reference_prdc(reference, generated, k),
                    )

    def test_metrics_ignore_independent_sample_order(self):
        expected = reference_prdc(self.REAL, self.FAKE, 3)
        for real in (self.REAL, self.REAL[::-1]):
            for fake in (self.FAKE, self.FAKE[::-1], self.FAKE[2:] + self.FAKE[:2]):
                with self.subTest(real=real, fake=fake):
                    self.assert_metrics_equal(self.compute_prdc(real, fake), expected)

    def test_public_compute_accumulates_batches(self):
        metric = ManifoldMetrics(distance="euclidean", manifold_k=3)
        real = torch.tensor(self.REAL, dtype=torch.float64)
        fake = torch.tensor(self.FAKE, dtype=torch.float64)
        metric.update(real[:2], fake[:2])
        metric.update(real[2:], fake[2:])
        self.assert_metrics_equal(metric.compute(num_splits=1), (1.0, 1.0, 16 / 15, 1.0))

    def test_callback_constructs_corrected_metric(self):
        from testing.metrics import callback

        self.assertIs(callback.ManifoldMetrics, ManifoldMetrics)
        metrics = callback.MetricCallback(num_cams=1, device=torch.device("cpu"))
        metric = metrics._get_or_create_metric("regression")["clatr_prdc"]
        self.assertIsInstance(metric, ManifoldMetrics)
        metric.update(
            torch.tensor(self.REAL, dtype=torch.float32),
            torch.tensor(self.FAKE, dtype=torch.float32),
        )
        self.assert_metrics_equal(metric.compute(num_splits=1), (1.0, 1.0, 16 / 15, 1.0))


if __name__ == "__main__":
    unittest.main()
