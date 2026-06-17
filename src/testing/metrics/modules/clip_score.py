import torch
from torch.nn.functional import cosine_similarity
from torchmetrics import Metric


class ClipScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("per_sample", default=[], dist_reduce_fx="cat")

    def update(self, gen_embedding, prompt_embedding, prompt_none_mask=None):
        sims = cosine_similarity(gen_embedding, prompt_embedding, dim=-1)  # [n_high, batch]
        sims = sims.transpose(0, 1)                                        # [batch, n_high]

        if prompt_none_mask is not None:
            mask = prompt_none_mask.to(sims.device).bool()                 # True = valid token
            counts = mask.sum(dim=1)
            summed = (sims * mask).sum(dim=1)
            valid = counts > 0
            per_sample = summed[valid] / counts[valid].clamp(min=1)        # ignore all-none rows
        else:
            per_sample = sims.mean(dim=1)

        self.per_sample.append(per_sample.detach())

    def per_sample_scores(self) -> torch.Tensor:
        if not self.per_sample:
            return torch.empty(0)
        return torch.cat([p.flatten() for p in self.per_sample])

    def compute(self) -> torch.Tensor:
        scores = self.per_sample_scores()
        if scores.numel() == 0:
            return torch.tensor(0.0)
        return scores.mean()
