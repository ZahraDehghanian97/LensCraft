from typing import Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric


def _compute_fd(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Fréchet distance between N(μ₁, Σ₁) and N(μ₂, Σ₂).

    d² = ‖μ₁ – μ₂‖² + Tr(Σ₁ + Σ₂ − 2·√(Σ₁ Σ₂))
    """
    mu1, mu2 = mu1.float(), mu2.float()
    sigma1, sigma2 = sigma1.float(), sigma2.float()

    a = (mu1 - mu2).pow(2).sum()
    b = torch.trace(sigma1) + torch.trace(sigma2)
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum()
    return a + b - 2.0 * c


class FrechetCLaTrDistance(Metric):
    """Fréchet distance on camera‑trajectory embeddings."""

    def __init__(self, num_features: Union[int, Module] = 5120, device: torch.device = torch.device("cpu"), **kwargs):
        super().__init__(**kwargs)
        self._device = device
        self._init_states(int(num_features))

    def _init_states(self, d: int):
        shape = (d, d)

        # real distribution
        self.add_state("real_sum", torch.zeros(d, dtype=torch.float32, device=self._device), dist_reduce_fx="sum")
        self.add_state("real_cov", torch.zeros(shape, dtype=torch.float32, device=self._device), dist_reduce_fx="sum")
        self.add_state("real_n", torch.zeros((), dtype=torch.long, device=self._device), dist_reduce_fx="sum")

        # fake distribution
        self.add_state("fake_sum", torch.zeros_like(self.real_sum), dist_reduce_fx="sum")
        self.add_state("fake_cov", torch.zeros_like(self.real_cov), dist_reduce_fx="sum")
        self.add_state("fake_n", torch.zeros_like(self.real_n), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, real_features: Tensor, fake_features: Tensor):
        """Accumulate batch statistics."""
        real = real_features.detach().to(self._device, dtype=torch.float32)
        fake = fake_features.detach().to(self._device, dtype=torch.float32)

        self.real_sum += real.sum(0)
        self.real_cov += real.t() @ real
        self.real_n += real.shape[0]

        self.fake_sum += fake.sum(0)
        self.fake_cov += fake.t() @ fake
        self.fake_n += fake.shape[0]

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.real_n < 2 or self.fake_n < 2:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)

        # means
        mu_r = self.real_sum / self.real_n.float()
        mu_f = self.fake_sum / self.fake_n.float()

        # unbiased covariances
        cov_r = (self.real_cov - self.real_n * torch.outer(mu_r, mu_r)) / (self.real_n - 1)
        cov_f = (self.fake_cov - self.fake_n * torch.outer(mu_f, mu_f)) / (self.fake_n - 1)

        return _compute_fd(mu_r, cov_r, mu_f, cov_f)
