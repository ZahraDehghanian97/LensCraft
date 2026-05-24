from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class KLLoss:
    def __call__(
        self,
        q: Tuple[Tensor, Tensor],
        p: Tuple[Tensor, Tensor],
    ) -> Tensor:
        mu_q, logvar_q = q
        mu_p, logvar_p = p
        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self) -> str:
        return "KLLoss()"


class InfoNCEWithFiltering:
    def __init__(self, temperature: float = 0.1, threshold_selfsim: Optional[float] = 0.995) -> None:
        self.temperature = float(temperature)
        self.threshold_selfsim = threshold_selfsim

    @staticmethod
    def _sim_matrix(x: Tensor, y: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        sent_token: Optional[Tensor] = None,
    ) -> Tensor:
        bs = x.shape[0]
        device = x.device
        sim_matrix = self._sim_matrix(x, y) / self.temperature

        if sent_token is not None and self.threshold_selfsim is not None:
            # threshold_selfsim is in [0, 1]; convert to cosine space [-1, 1]
            real_threshold = 2.0 * self.threshold_selfsim - 1.0
            sent_token = F.normalize(sent_token, dim=-1)
            selfsim = sent_token @ sent_token.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            sim_matrix = sim_matrix.masked_fill(
                selfsim_nodiag > real_threshold, float("-inf")
            )

        labels = torch.arange(bs, device=device)
        loss = 0.5 * (
            F.cross_entropy(sim_matrix, labels)
            + F.cross_entropy(sim_matrix.T, labels)
        )
        return loss

    def __repr__(self) -> str:
        return f"InfoNCEWithFiltering(temp={self.temperature}, thr={self.threshold_selfsim})"
