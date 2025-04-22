from typing import Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric

def _compute_fd(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    """Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)). # noqa

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    with torch.no_grad():
        # Compute the squared difference between means
        a = (mu1 - mu2).square().sum(dim=-1)
        
        # Compute the trace of sigma1 + sigma2
        b = sigma1.trace() + sigma2.trace()
        
        # Add small regularization to ensure numerical stability
        reg = 1e-6 * torch.eye(
            sigma1.size(0), device=sigma1.device, dtype=sigma1.dtype
        )
        sigma1_reg = sigma1 + reg
        sigma2_reg = sigma2 + reg
        
        # Compute eigenvalues
        c = torch.linalg.eigvals(sigma1_reg @ sigma2_reg).sqrt().real.sum(dim=-1)
        
        fd = a + b - 2 * c
        
        # Clean up tensors
        del a, b, c, sigma1_reg, sigma2_reg
        
        return fd

class FrechetCLaTrDistance(Metric):
    """ Implementation of Frechet CLaTr Distance (FCD) metric with enhanced debugging. """

    def __init__(self, num_features: Union[int, Module] = 5120, dtype=torch.float32, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.dtype = dtype
        self._initialize_states(num_features)

    def _initialize_states(self, num_features: Union[int, Module]):
        """ Initialize metric states for tracking feature statistics. """
        mx_num_feats = (num_features, num_features)

        self.add_state(
            "real_features_sum",
            torch.zeros(num_features),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feats),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feats),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum"
        )

    def update(self, real_features: Tensor, fake_features: Tensor):
        """ Update states with new batches of features. """
        real_features = real_features.to(self.dtype).detach()
        fake_features = fake_features.to(self.dtype).detach()
        
        with torch.no_grad():
            self.real_features_sum += real_features.sum(dim=0)
            self.real_features_cov_sum += real_features.t() @ real_features
            self.real_features_num_samples += real_features.shape[0]

            self.fake_features_sum += fake_features.sum(dim=0)
            self.fake_features_cov_sum += fake_features.t() @ fake_features
            self.fake_features_num_samples += fake_features.shape[0]

    def compute(self) -> Tensor:
        """
        Calculate FD_CLaTr score based on accumulated extracted features from the two
        distributions.
        """
        if self.real_features_num_samples == 0 or self.fake_features_num_samples == 0:
            return torch.tensor(0.0, dtype=self.dtype)
            
        with torch.no_grad():
            mean_real = self.real_features_sum / self.real_features_num_samples
            mean_fake = self.fake_features_sum / self.fake_features_num_samples
            
            cov_real_num = (
                self.real_features_cov_sum
                - self.real_features_num_samples * torch.outer(mean_real, mean_real)
            )
            cov_real = cov_real_num / (self.real_features_num_samples - 1)
            
            cov_fake_num = (
                self.fake_features_cov_sum
                - self.fake_features_num_samples * torch.outer(mean_fake, mean_fake)
            )
            cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
            
            fd = _compute_fd(mean_real, cov_real, mean_fake, cov_fake)
            
            del mean_real, mean_fake, cov_real, cov_fake, cov_real_num, cov_fake_num
            torch.cuda.empty_cache()
            
            return fd.to(self.dtype)