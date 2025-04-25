import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class CLaTrScore(Metric):
    """Compute cosine similarity between trajectory and text features."""

    def __init__(self, device: torch.device = torch.device("cpu"), **kwargs):
        super().__init__(**kwargs)
        self._device = device
        self.add_state("traj_feat", default=[], dist_reduce_fx="cat")
        self.add_state("text_feats", default=[], dist_reduce_fx="cat")

    @staticmethod
    def _normalize_features(features: torch.Tensor) -> torch.Tensor:
        """Normalize features using L2 norm."""
        features = features.to(dtype=torch.float16)
        return features / (features.norm(p=2, dim=-1, keepdim=True) + 1e-4)

    def update(self, traj_feat: torch.Tensor, text_feats: torch.Tensor):
        self.traj_feat.append(self._normalize_features(traj_feat.to(self._device)).detach())
        self.text_feats.append(self._normalize_features(text_feats.to(self._device)).detach())

    def compute(self) -> torch.Tensor:
        if len(self.traj_feat) == 0 or len(self.text_feats) == 0:
            return torch.tensor(0.0, dtype=torch.float16, device=self._device)

        traj_feat = dim_zero_cat(self.traj_feat).to(self._device, dtype=torch.float16)
        text_feats = dim_zero_cat(self.text_feats).to(self._device, dtype=torch.float16)
        score = (100 * (traj_feat * text_feats).sum(dim=-1)).mean()
        result = torch.max(score, torch.zeros_like(score))

        # Explicitly free memory
        del traj_feat, text_feats, score
        torch.cuda.empty_cache()
        return result
