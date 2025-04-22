from typing import Tuple
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
        distance: str = "euclidean",
        dtype=torch.float32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.manifold_k = manifold_k
        self.reset_real_features = reset_real_features
        self.distance = distance
        self.dtype = dtype

        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")

    def _compute_pairwise_distance(self, data_x: Tensor, data_y: Tensor = None) -> Tensor:
        """Compute pairwise distances between two sets of features."""
        if data_y is None:
            data_y = data_x.clone()

        with torch.no_grad():
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

    def update(self, real_features: Tensor, fake_features: Tensor):
        """Update states with new batches of features."""
        self.real_features.append(real_features.to(self.dtype).detach())
        self.fake_features.append(fake_features.to(self.dtype).detach())

    def compute(self, num_splits: int = 5) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute final PRDC metrics by averaging over splits."""
        if not self.real_features or not self.fake_features:
            zero = torch.tensor(0.0, dtype=self.dtype)
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
                real_distances.fill_diagonal_(float('inf'))
                real_nn_distances = self._get_kth_value(real_distances, k=self.manifold_k + 1, axis=-1)
                
                fake_distances = self._compute_pairwise_distance(fake_chunk)
                fake_distances.fill_diagonal_(float('inf'))
                fake_nn_distances = self._get_kth_value(fake_distances, k=self.manifold_k + 1, axis=-1)
                
                distance_real_fake = self._compute_pairwise_distance(real_chunk, fake_chunk)
                
                precision = (
                    (distance_real_fake < real_nn_distances.unsqueeze(1))
                    .any(dim=0)
                    .to(self.dtype)
                ).mean()
                
                recall = (
                    (distance_real_fake < fake_nn_distances.unsqueeze(0))
                    .any(dim=1)
                    .to(self.dtype)
                ).mean()
                
                density = (1.0 / float(self.manifold_k)) * (
                    distance_real_fake < real_nn_distances.unsqueeze(1)
                ).sum(dim=0).to(self.dtype).mean()
                
                coverage = (
                    (distance_real_fake.min(dim=1).values < real_nn_distances)
                    .to(self.dtype)
                    .mean()
                )
                
                metrics[0].append(precision)
                metrics[1].append(recall)
                metrics[2].append(density)
                metrics[3].append(coverage)
                
                del real_distances, fake_distances, distance_real_fake
                del real_nn_distances, fake_nn_distances
                torch.cuda.empty_cache()
            
            results = []
            for metric_list in metrics:
                if metric_list:
                    results.append(torch.stack(metric_list).mean())
                else:
                    results.append(torch.tensor(0.0, dtype=self.dtype))
            
            del real_features, fake_features
            del real_features_split, fake_features_split
            torch.cuda.empty_cache()
            
            return tuple(results)