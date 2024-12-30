from typing import Union, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric


class FrechetCLaTrDistance(Metric):
    """ Implementation of Frechet CLaTr Distance (FCD) metric. """

    def __init__(self, num_features: Union[int, Module] = 256, **kwargs):
        super().__init__(**kwargs)
        self._initialize_states(num_features)

    def _initialize_states(self, num_features: Union[int, Module]):
        """ Initialize metric states for tracking feature statistics. """
        mx_num_feats = (num_features, num_features)

        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum"
        )

    def _compute_mean_cov(
        self,
        feat_sum: Tensor,
        feat_cov_sum: Tensor,
        num_samples: int
    ) -> Tuple[Tensor, Tensor]:
        """ Compute mean and covariance matrix from accumulated statistics. """
        mean = feat_sum / num_samples
        mean = mean.unsqueeze(0)

        cov_num = feat_cov_sum - num_samples * mean.t().mm(mean)
        cov = cov_num / (num_samples - 1)

        return mean.squeeze(0), cov

    def _compute_fd(
        self,
        mu1: Tensor,
        sigma1: Tensor,
        mu2: Tensor,
        sigma2: Tensor
    ) -> Tensor:
        """ Compute Frechet Distance between two distributions. """

        diff_mean_sq = (mu1 - mu2).square().sum(dim=-1)

        trace_sum = sigma1.trace() + sigma2.trace()

        sqrt_prod_trace = torch.linalg.eigvals(
            sigma1 @ sigma2).sqrt().real.sum(dim=-1)

        return diff_mean_sq + trace_sum - 2 * sqrt_prod_trace

    def update(self, real_features: Tensor, fake_features: Tensor):
        """ Update states with new batches of features. """

        self.orig_dtype = real_features.dtype

        self.real_features_sum += real_features.sum(dim=0)
        self.real_features_cov_sum += real_features.t().mm(real_features)
        self.real_features_num_samples += real_features.shape[0]

        self.fake_features_sum += fake_features.sum(dim=0)
        self.fake_features_cov_sum += fake_features.t().mm(fake_features)
        self.fake_features_num_samples += fake_features.shape[0]

    def compute(self) -> Tensor:
        """ Compute final FCD score based on accumulated statistics. """

        mean_real, cov_real = self._compute_mean_cov(
            self.real_features_sum,
            self.real_features_cov_sum,
            self.real_features_num_samples
        )

        mean_fake, cov_fake = self._compute_mean_cov(
            self.fake_features_sum,
            self.fake_features_cov_sum,
            self.fake_features_num_samples
        )

        fd = self._compute_fd(mean_real, cov_real, mean_fake, cov_fake)

        return fd.to(self.orig_dtype)
