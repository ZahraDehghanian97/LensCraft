from typing import Tuple, List
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class ManifoldMetrics(Metric):
    """Compute precision, recall, density, and coverage metrics on feature manifolds."""

    def __init__(
        self,
        reset_real_features: bool = True,
        manifold_k: int = 3,
        distance: str = "geodesic",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.manifold_k = manifold_k
        self.reset_real_features = reset_real_features
        self.distance = distance

        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")

    def _compute_pairwise_distance(self, data_x: Tensor, data_y: Tensor = None) -> Tensor:
        """Compute pairwise distances between two sets of features."""
        if data_y is None:
            data_y = data_x.clone()

        if self.distance == "euclidean":
            num_feats = data_x.shape[-1]
            X = data_x.reshape(-1, num_feats).unsqueeze(0)
            Y = data_y.reshape(-1, num_feats).unsqueeze(0)
            return torch.cdist(X, Y, 2).squeeze(0)

        raise ValueError(f"Unsupported distance metric: {self.distance}")

    def _get_kth_value(self, unsorted: Tensor, k: int, axis: int = -1) -> Tensor:
        """Get the k-th smallest value along specified axis."""
        k_smallests = torch.topk(unsorted, k, largest=False, dim=axis)
        return k_smallests.values.max(axis=axis).values

    def _compute_nn_distances(self, input_features: Tensor, nearest_k: int) -> Tensor:
        """Compute distances to k-nearest neighbors."""
        distances = self._compute_pairwise_distance(input_features)
        return self._get_kth_value(distances, k=nearest_k + 1, axis=-1)

    def _compute_metrics(
        self,
        real_features: Tensor,
        fake_features: Tensor,
        nearest_k: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute PRDC metrics for a single batch."""
        real_nn_distances = self._compute_nn_distances(
            real_features, nearest_k)
        fake_nn_distances = self._compute_nn_distances(
            fake_features, nearest_k)
        distance_real_fake = self._compute_pairwise_distance(
            real_features, fake_features)

        precision = (
            (distance_real_fake < real_nn_distances.unsqueeze(1))
            .any(axis=0)
            .to(float)
        ).mean()

        recall = (
            (distance_real_fake < fake_nn_distances.unsqueeze(1))
            .any(axis=1)
            .to(float)
        ).mean()

        density = (1.0 / float(nearest_k)) * (
            distance_real_fake < real_nn_distances.unsqueeze(1)
        ).sum(axis=0).to(float).mean()

        coverage = (
            (distance_real_fake.min(axis=1).values < real_nn_distances)
            .to(float)
            .mean()
        )

        return precision, recall, density, coverage

    def update(self, real_features: Tensor, fake_features: Tensor):
        """Update states with new batches of features."""
        self.real_features.append(real_features)
        self.fake_features.append(fake_features)

    def compute(self, num_splits: int = 5) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute final PRDC metrics by averaging over splits."""
        real_features = dim_zero_cat(
            self.real_features).chunk(num_splits, dim=0)
        fake_features = dim_zero_cat(
            self.fake_features).chunk(num_splits, dim=0)

        metrics: List[List[Tensor]] = [[], [], [], []]

        for real, fake in zip(real_features, fake_features):
            batch_metrics = self._compute_metrics(real, fake, self.manifold_k)
            for metric_list, metric_value in zip(metrics, batch_metrics):
                metric_list.append(metric_value)

        return tuple(torch.stack(metric_list).mean() for metric_list in metrics)
