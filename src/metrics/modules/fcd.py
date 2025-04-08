from typing import Union, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric


class FrechetCLaTrDistance(Metric):
    """ Implementation of Frechet CLaTr Distance (FCD) metric with enhanced debugging. """

    def __init__(self, num_features: Union[int, Module] = 5120, **kwargs):
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

        # Debug: Check properties of the covariance matrix
        print(f"Covariance matrix shape: {cov.shape}")
        print(f"Covariance matrix trace: {cov.trace().item()}")
        print(f"Covariance matrix min eigenvalue: {torch.linalg.eigvals(cov).real.min().item()}")
        print(f"Covariance matrix max eigenvalue: {torch.linalg.eigvals(cov).real.max().item()}")
        
        # Check if covariance matrix is positive semi-definite by looking at eigenvalues
        min_eig = torch.linalg.eigvals(cov).real.min().item()
        if min_eig < 0 and abs(min_eig) > 1e-6:  # Allow for small numerical errors
            print(f"WARNING: Covariance matrix has negative eigenvalues: {min_eig}")

        return mean.squeeze(0), cov

    def _compute_fd(
        self,
        mu1: Tensor,
        sigma1: Tensor,
        mu2: Tensor,
        sigma2: Tensor
    ) -> Tensor:
        """ Compute Frechet Distance between two distributions with detailed debugging. """
        # 1. Mean difference component
        diff_mean_sq = (mu1 - mu2).square().sum(dim=-1)
        
        # 2. Trace component
        trace_sum = sigma1.trace() + sigma2.trace()
        
        # 3. Cross-covariance component
        # Check numerical stability of sigma1 @ sigma2
        sigma_prod = sigma1 @ sigma2
        
        # Compute eigenvalues and check for complex values
        eigvals = torch.linalg.eigvals(sigma_prod)
        eigvals_real = eigvals.real
        eigvals_imag = eigvals.imag
        
        # Debug numerical issues with eigenvalues
        has_complex = torch.any(torch.abs(eigvals_imag) > 1e-6)
        has_negative = torch.any(eigvals_real < 0)
        
        # Detailed debugging of eigenvalues
        print("\n=== Frechet Distance Component Breakdown ===")
        print(f"Mean difference squared: {diff_mean_sq.item()}")
        print(f"Trace sum: {trace_sum.item()}")
        
        if has_complex:
            print(f"WARNING: Found complex eigenvalues! Max imaginary part: {torch.abs(eigvals_imag).max().item()}")
        
        if has_negative:
            print(f"WARNING: Found negative eigenvalues! Min real part: {eigvals_real.min().item()}")
            # Count small negative values that could be numerical errors
            small_neg = torch.sum((eigvals_real < 0) & (eigvals_real > -1e-6)).item()
            if small_neg > 0:
                print(f"  {small_neg} negative eigenvalues might be due to numerical errors (> -1e-6)")
        
        # Debug: Print statistics about eigenvalues
        print(f"Eigenvalue stats - Min: {eigvals_real.min().item()}, Max: {eigvals_real.max().item()}, Mean: {eigvals_real.mean().item()}")
        
        # Handle negative or complex eigenvalues more gracefully
        sqrt_eigvals = eigvals_real.clamp(min=0).sqrt()
        sqrt_prod_trace = sqrt_eigvals.sum(dim=-1)
        
        print(f"2*sqrt(prod_trace): {2 * sqrt_prod_trace.item()}")
        
        # Calculate final FD and show components
        fd = diff_mean_sq + trace_sum - 2 * sqrt_prod_trace
        
        print(f"Final FD: {fd.item()}")
        print(f"Proportion from mean diff: {diff_mean_sq.item() / fd.item():.4f}")
        print(f"Proportion from trace sum: {trace_sum.item() / fd.item():.4f}")
        print(f"Proportion from sqrt_prod_trace: {(2 * sqrt_prod_trace.item()) / fd.item():.4f}")
        print("==========================================\n")

        return fd

    def update(self, real_features: Tensor, fake_features: Tensor):
        """ Update states with new batches of features. """
        # Debug: Check if features are normalized
        real_norms = torch.norm(real_features, p=2, dim=1)
        fake_norms = torch.norm(fake_features, p=2, dim=1)
        
        print("\n=== Feature Normalization Check ===")
        print(f"Real Features - Mean norm: {real_norms.mean().item():.4f}, Min: {real_norms.min().item():.4f}, Max: {real_norms.max().item():.4f}")
        print(f"Fake Features - Mean norm: {fake_norms.mean().item():.4f}, Min: {fake_norms.min().item():.4f}, Max: {fake_norms.max().item():.4f}")
        
        # Check for any NaN or Inf values
        if torch.isnan(real_features).any() or torch.isinf(real_features).any():
            print("WARNING: Real features contain NaN or Inf values!")
        
        if torch.isnan(fake_features).any() or torch.isinf(fake_features).any():
            print("WARNING: Fake features contain NaN or Inf values!")
        
        print(f"Real Features shape: {real_features.shape}")
        print(f"Fake Features shape: {fake_features.shape}")
        self.orig_dtype = real_features.dtype

        # Debug: Print basic statistics about features
        print(f"Real Features - Mean: {real_features.mean().item():.4f}, Std: {real_features.std().item():.4f}")
        print(f"Fake Features - Mean: {fake_features.mean().item():.4f}, Std: {fake_features.std().item():.4f}")
        
        self.real_features_sum += real_features.sum(dim=0)
        self.real_features_cov_sum += real_features.t().mm(real_features)
        self.real_features_num_samples += real_features.shape[0]

        self.fake_features_sum += fake_features.sum(dim=0)
        self.fake_features_cov_sum += fake_features.t().mm(fake_features)
        self.fake_features_num_samples += fake_features.shape[0]

    def compute(self) -> Tensor:
        """ Compute final FCD score based on accumulated statistics. """
        print("\n=== Computing Frechet Distance ===")
        print(f"Number of real samples: {self.real_features_num_samples.item()}")
        print(f"Number of fake samples: {self.fake_features_num_samples.item()}")
        
        mean_real, cov_real = self._compute_mean_cov(
            self.real_features_sum,
            self.real_features_cov_sum,
            self.real_features_num_samples
        )
        
        print("\n--- Real Features Statistics ---")
        print(f"Mean norm: {torch.norm(mean_real).item():.6f}")
        print(f"Mean min: {mean_real.min().item():.6f}, max: {mean_real.max().item():.6f}")
        print(f"Covariance trace: {cov_real.trace().item():.6f}")

        mean_fake, cov_fake = self._compute_mean_cov(
            self.fake_features_sum,
            self.fake_features_cov_sum,
            self.fake_features_num_samples
        )
        
        print("\n--- Fake Features Statistics ---")
        print(f"Mean norm: {torch.norm(mean_fake).item():.6f}")
        print(f"Mean min: {mean_fake.min().item():.6f}, max: {mean_fake.max().item():.6f}")
        print(f"Covariance trace: {cov_fake.trace().item():.6f}")
        
        # Compare means directly
        mean_diff = mean_real - mean_fake
        mean_diff_norm = torch.norm(mean_diff).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            mean_real.unsqueeze(0), 
            mean_fake.unsqueeze(0)
        ).item()
        
        print("\n--- Mean Comparison ---")
        print(f"Mean difference norm: {mean_diff_norm:.6f}")
        print(f"Cosine similarity between means: {cosine_sim:.6f}")
        
        # Now compute the actual FD
        fd = self._compute_fd(mean_real, cov_real, mean_fake, cov_fake)

        return fd.to(self.orig_dtype)