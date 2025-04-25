from typing import Tuple
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class ManifoldMetrics(Metric):
    """Precision, Recall, Density & Coverage on feature manifolds."""

    def __init__(
        self,
        reset_real_features: bool = True,
        manifold_k: int = 3,
        distance: str = "euclidean",
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.manifold_k = manifold_k
        self.reset_real_features = reset_real_features
        self.distance = distance
        self._device = device

        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")

    def _compute_pairwise_distance(self, data_x: Tensor, data_y: Tensor = None) -> Tensor:
        """Compute pairwise distances between two sets of features."""
        data_x = data_x.to(dtype=torch.float16)
        data_y = data_x.clone() if data_y is None else data_y.to(dtype=torch.float16)

        if self.distance == "euclidean":
            num_feats = data_x.shape[-1]
            X = data_x.reshape(-1, num_feats).unsqueeze(0)
            Y = data_y.reshape(-1, num_feats).unsqueeze(0)
            return torch.cdist(X, Y, p=2).squeeze(0)

        raise ValueError(f"Unsupported distance metric: {self.distance}")

    def _get_kth_value(self, unsorted: Tensor, k: int, axis: int = -1) -> Tensor:
        """Get the k-th smallest value along specified axis."""
        k_smallests = torch.topk(unsorted, k, largest=False, dim=axis).values
        return k_smallests.max(axis=axis).values

    def update(self, real_features: Tensor, fake_features: Tensor):
        """Update states with new batches of features."""
        self.real_features.append(real_features.detach().to(self._device, dtype=torch.float16))
        self.fake_features.append(fake_features.detach().to(self._device, dtype=torch.float16))

    def compute(self, num_splits: int = 5) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute final PRDC metrics by averaging over splits."""
        if not self.real_features or not self.fake_features:
            zero = torch.tensor(0.0, dtype=torch.float16, device=self._device)
            return zero, zero, zero, zero

        with torch.no_grad():
            real_features = dim_zero_cat(self.real_features)
            fake_features = dim_zero_cat(self.fake_features)

            real_features_split = real_features.chunk(num_splits, dim=0)
            fake_features_split = fake_features.chunk(num_splits, dim=0)

            metrics = [[], [], [], []]

            for real_chunk, fake_chunk in zip(real_features_split, fake_features_split):
                if real_chunk.size(0) == 0 or fake_chunk.size(0) == 0:
                    continue

                real_distances = self._compute_pairwise_distance(real_chunk)
                real_distances.fill_diagonal_(float("inf"))
                real_nn = self._get_kth_value(real_distances, k=self.manifold_k + 1, axis=-1)

                fake_distances = self._compute_pairwise_distance(fake_chunk)
                fake_distances.fill_diagonal_(float("inf"))
                fake_nn = self._get_kth_value(fake_distances, k=self.manifold_k + 1, axis=-1)

                dist_real_fake = self._compute_pairwise_distance(real_chunk, fake_chunk)

                precision = (dist_real_fake < real_nn.unsqueeze(1)).any(dim=0).float().mean()
                recall = (dist_real_fake < fake_nn.unsqueeze(0)).any(dim=1).float().mean()
                density = (dist_real_fake < real_nn.unsqueeze(1)).sum(dim=0).float().mean() / float(
                    self.manifold_k
                )
                coverage = (dist_real_fake.min(dim=1).values < real_nn).float().mean()

                metrics[0].append(precision.to(dtype=torch.float16))
                metrics[1].append(recall.to(dtype=torch.float16))
                metrics[2].append(density.to(dtype=torch.float16))
                metrics[3].append(coverage.to(dtype=torch.float16))

                # Free memory between splits
                del real_distances, fake_distances, dist_real_fake, real_nn, fake_nn
                torch.cuda.empty_cache()

            results = [torch.stack(m).mean() if m else torch.tensor(0.0, dtype=torch.float16) for m in metrics]

            return tuple(results)
