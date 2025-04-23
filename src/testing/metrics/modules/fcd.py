from typing import Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric

def _compute_fd(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    """Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.
    """
    mu1 = mu1.to(torch.float32)
    mu2 = mu2.to(torch.float32)
    sigma1 = sigma1.to(torch.float32)
    sigma2 = sigma2.to(torch.float32)

    with torch.no_grad():
        a = (mu1 - mu2).square().sum(dim=-1)
        b = sigma1.trace() + sigma2.trace()
        
        reg = 1e-3 * torch.eye(
            sigma1.size(0), device=sigma1.device, dtype=torch.float32
        )
        sigma1_reg = sigma1 + reg
        sigma2_reg = sigma2 + reg
        
        try:
            sigma_product = sigma1_reg @ sigma2_reg
            
            if torch.isnan(sigma_product).any() or torch.isinf(sigma_product).any():
                raise RuntimeError("NaN or Inf values in sigma_product")
                
            eigvals = torch.linalg.eigvals(sigma_product)
            
            c = eigvals.abs().sqrt().sum(dim=-1)
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            print(f"Warning: Using fallback method for eigenvalue calculation: {e}")
            
            sqrt_sigma1 = torch.linalg.cholesky(sigma1_reg)
            sqrt_sigma1_sigma2_sqrt_sigma1 = sqrt_sigma1 @ sigma2_reg @ sqrt_sigma1.t()
            
            eigvals = torch.linalg.eigvals(sqrt_sigma1_sigma2_sqrt_sigma1)
            c = eigvals.abs().sum(dim=-1)
            
            if torch.isnan(c) or torch.isinf(c):
                print("Warning: Using trace approximation")
                c = torch.sqrt(torch.trace(sigma1_reg @ sigma2_reg))
        
        fd = a + b - 2 * c
        
        fd = torch.max(fd, torch.tensor(0.0, device=fd.device, dtype=torch.float32))
        
        del a, b, c, sigma1_reg, sigma2_reg
        if 'sigma_product' in locals():
            del sigma_product
        if 'eigvals' in locals():
            del eigvals
        torch.cuda.empty_cache()
        
        return fd.to(torch.float16)


class FrechetCLaTrDistance(Metric):
    """ Implementation of Frechet CLaTr Distance (FCD) metric with enhanced debugging. """

    def __init__(self, num_features: Union[int, Module] = 5120, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self._initialize_states(num_features)

    def _initialize_states(self, num_features: Union[int, Module]):
        """ Initialize metric states for tracking feature statistics. """
        mx_num_feats = (num_features, num_features)

        self.add_state(
            "real_features_sum",
            torch.zeros(num_features, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feats, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feats, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum"
        )

    def update(self, real_features: Tensor, fake_features: Tensor):
        """ Update states with new batches of features. """
        real_features = real_features.detach().to(torch.float32)
        fake_features = fake_features.detach().to(torch.float32)
        
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
            return torch.tensor(0.0, dtype=torch.float16)
            
        with torch.no_grad():
            real_num = self.real_features_num_samples.to(torch.float32)
            fake_num = self.fake_features_num_samples.to(torch.float32)
            
            mean_real = self.real_features_sum / real_num
            mean_fake = self.fake_features_sum / fake_num
            
            cov_real_num = (
                self.real_features_cov_sum
                - real_num * torch.outer(mean_real, mean_real)
            )
            cov_real = cov_real_num / (real_num - 1.0)
            
            cov_fake_num = (
                self.fake_features_cov_sum
                - fake_num * torch.outer(mean_fake, mean_fake)
            )
            cov_fake = cov_fake_num / (fake_num - 1.0)
            
            if torch.isnan(cov_real).any() or torch.isinf(cov_real).any():
                print("Warning: NaN or Inf values in real covariance matrix")
                cov_real = cov_real.clone()
                cov_real[torch.isnan(cov_real) | torch.isinf(cov_real)] = 0.0
                
            if torch.isnan(cov_fake).any() or torch.isinf(cov_fake).any():
                print("Warning: NaN or Inf values in fake covariance matrix")
                cov_fake = cov_fake.clone()
                cov_fake[torch.isnan(cov_fake) | torch.isinf(cov_fake)] = 0.0
            
            fd = _compute_fd(mean_real, cov_real, mean_fake, cov_fake)
            
            del mean_real, mean_fake, cov_real, cov_fake, cov_real_num, cov_fake_num
            torch.cuda.empty_cache()
            
            return fd
