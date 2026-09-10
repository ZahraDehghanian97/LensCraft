import logging
from typing import Any, Dict, List, Optional

import torch

from testing.metrics.modules.caption_top1 import CaptionTop1
from testing.metrics.modules.clip_score import ClipScore
from testing.metrics.modules.prdc import ManifoldMetrics
from utils.importing import ModuleImporter
from utils.paths import third_party

logger = logging.getLogger(__name__)

# Import by file so LensCraft's own `src` package cannot shadow DIRECTOR's.
FrechetCLaTrDistance = ModuleImporter.import_module(
    "_lenscraft_director_fcd",
    third_party("DIRECTOR", "src", "metrics", "modules", "fcd.py"),
).FrechetCLaTrDistance
_CLaTrScoreBase = ModuleImporter.import_module(
    "_lenscraft_director_clatr_score",
    third_party("DIRECTOR", "src", "metrics", "modules", "clatr_score.py"),
).CLaTrScore


class CLaTrScore(_CLaTrScoreBase):
    def compute(self):
        if len(self.traj_feat) == 0 or len(self.text_feats) == 0:
            return torch.tensor(0.0)
        return super().compute()


class MetricCallback:
    CLATR_FEAT_DIM: int = 256

    def __init__(
        self,
        num_cams: int,
        device: torch.device,
        clip_embeddings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.num_cams = num_cams
        self._device = device
        self.clip_embeddings = clip_embeddings

        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.active_metrics: set[str] = set()

    def _get_or_create_metric(self, run_type: str) -> Dict[str, Any]:
        if run_type not in self.metrics:
            self.metrics[run_type] = {
                "clatr_fd": FrechetCLaTrDistance(
                    num_features=self.CLATR_FEAT_DIM
                ).to(self._device),
                "clatr_prdc": ManifoldMetrics(distance="euclidean").to(self._device),
                "clatr_score": CLaTrScore().to(self._device),
                "clip_score": ClipScore().to(self._device),
            }
            if self.clip_embeddings is not None:
                self.metrics[run_type]["caption_top1"] = CaptionTop1(
                    clip_embeddings=self.clip_embeddings,
                ).to(self._device)
        self.active_metrics.add(run_type)
        return self.metrics[run_type]

    def update_clatr_metrics(
        self,
        run_type: str,
        gen_features: torch.Tensor,
        ref_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
    ) -> None:
        m = self._get_or_create_metric(run_type)

        gen = gen_features.to(self._device, dtype=torch.float32)
        ref = ref_features.to(self._device, dtype=torch.float32)

        m["clatr_prdc"].update(ref, gen)
        m["clatr_fd"].update(ref, gen)

        if text_features is not None:
            txt = text_features.to(self._device, dtype=torch.float32)
            if txt.shape[-1] != gen.shape[-1]:
                raise ValueError(
                    "CLaTr-Score requires trajectory and text latents to live "
                    f"in the same space; got traj={tuple(gen.shape)}, "
                    f"text={tuple(txt.shape)}. Both must be 256-d CLaTr latents."
                )
            m["clatr_score"].update(gen, txt)

    def update_caption_top1(
        self,
        run_type: str,
        encoder_features: torch.Tensor,
        params: List[Any],
    ) -> None:
        if self.clip_embeddings is None:
            return
        m = self._get_or_create_metric(run_type)
        if "caption_top1" in m:
            m["caption_top1"].update(encoder_features.to(self._device), params)

    def update_clip_score(
        self,
        run_type: str,
        gen_embedding: torch.Tensor,
        prompt_embedding: torch.Tensor,
        prompt_none_mask: Optional[torch.Tensor] = None,
    ) -> None:
        m = self._get_or_create_metric(run_type)
        m["clip_score"].update(
            gen_embedding.to(self._device),
            prompt_embedding.to(self._device),
            prompt_none_mask,
        )

    def bootstrap_metrics(self, run_type, n_boot=500, seed=0, max_samples=None):
        import numpy as np
        m = self.metrics[run_type]

        def _cat(x):
            return (torch.cat(x) if isinstance(x, (list, tuple)) else x).detach().float()

        ref = _cat(m["clatr_prdc"].real_features)     # (N, d) reference traj feats
        gen = _cat(m["clatr_prdc"].fake_features)      # (N, d) generated traj feats
        tg = _cat(m["clatr_score"].traj_feat)          # generated feats paired with text
        tx = _cat(m["clatr_score"].text_feats)         # text feats

        device, n, d = ref.device, ref.shape[0], ref.shape[-1]
        boot_n = n if max_samples is None else min(n, max_samples)
        if boot_n <= d:
            logger.warning(
                "FCD covariance is %d-d but only %d samples are resampled for "
                "'%s'; its +/- will be unstable (needs well over %d samples for "
                "a non-singular covariance).",
                d, boot_n, run_type, d,
            )
        rng = np.random.default_rng(seed)
        keys = ("fcd", "precision", "recall", "density", "coverage",
                "clatr_score", "caption_overall_top1", "clip_score")
        acc = {k: [] for k in keys}
        has_text = tg.shape[0] > 0 and tx.shape[0] > 0

        caption = m.get("caption_top1")
        cs_correct = cs_total = cs_n = None
        if caption is not None and self.clip_embeddings is not None:
            outcomes = caption.per_sample_outcomes()
            cs_correct = np.array(
                [sum(1 for _, ok in s if ok) for s in outcomes], dtype=np.float64
            )
            cs_total = np.array([len(s) for s in outcomes], dtype=np.float64)
            cs_n = cs_correct.shape[0]

        clip_metric = m.get("clip_score")
        clip_scores = clip_n = None
        if clip_metric is not None and clip_metric.per_sample:
            clip_scores = clip_metric.per_sample_scores().detach().cpu().numpy()
            clip_n = clip_scores.shape[0]

        for _ in range(n_boot):
            idx = torch.as_tensor(rng.integers(0, n, boot_n), device=device, dtype=torch.long)

            fcd = FrechetCLaTrDistance(num_features=d).to(device)
            fcd.update(ref[idx], gen[idx])
            acc["fcd"].append(float(fcd.compute()))

            prdc = ManifoldMetrics(distance="euclidean").to(device)
            prdc.update(ref[idx], gen[idx])
            p, r, dn, c = prdc.compute()
            acc["precision"].append(float(p)); acc["recall"].append(float(r))
            acc["density"].append(float(dn)); acc["coverage"].append(float(c))

            if has_text:
                tn = tg.shape[0] if max_samples is None else min(tg.shape[0], max_samples)
                tidx = torch.as_tensor(
                    rng.integers(0, tg.shape[0], tn),
                    device=device, dtype=torch.long,
                )
                cs = CLaTrScore().to(device)
                cs.update(tg[tidx], tx[tidx])
                acc["clatr_score"].append(float(cs.compute()))

            if cs_n:
                cidx = rng.integers(0, cs_n, cs_n)
                tot = cs_total[cidx].sum()
                acc["caption_overall_top1"].append(
                    float(cs_correct[cidx].sum() / tot) if tot else 0.0
                )

            if clip_n:
                clidx = rng.integers(0, clip_n, clip_n)
                acc["clip_score"].append(float(clip_scores[clidx].mean()))

        return {
            f"{run_type}/{k}": (float(np.mean(v)), float(np.std(v, ddof=1)))
            for k, v in acc.items() if v
        }

    def compute_clatr_metrics(self, run_type: str) -> Dict[str, float]:
        if run_type not in self.active_metrics:
            return {
                f"{run_type}/clatr_score": 0.0,
                f"{run_type}/precision": 0.0,
                f"{run_type}/recall": 0.0,
                f"{run_type}/density": 0.0,
                f"{run_type}/coverage": 0.0,
                f"{run_type}/fcd": 0.0,
            }

        m = self.metrics[run_type]

        has_clatr_samples = len(m["clatr_prdc"].real_features) > 0

        if has_clatr_samples:
            clatr_score = m["clatr_score"].compute()
            precision, recall, density, coverage = m["clatr_prdc"].compute()
            fcd = m["clatr_fd"].compute()
        else:
            zero = torch.tensor(0.0)
            clatr_score = zero
            precision = recall = density = coverage = zero
            fcd = zero

        m["clatr_score"].reset()
        m["clatr_prdc"].reset()
        m["clatr_fd"].reset()

        caption_top1_metrics: Dict[str, float] = {}
        if "caption_top1" in m and self.clip_embeddings is not None:
            caption_top1_metrics = m["caption_top1"].compute()
            m["caption_top1"].reset()

        clip_score: Optional[float] = None
        if "clip_score" in m:
            if m["clip_score"].per_sample:
                clip_score = float(m["clip_score"].compute().item())
            m["clip_score"].reset()

        self.active_metrics.remove(run_type)

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result: Dict[str, float] = {
            f"{run_type}/clatr_score": float(clatr_score.item()),
            f"{run_type}/precision": float(precision.item()),
            f"{run_type}/recall": float(recall.item()),
            f"{run_type}/density": float(density.item()),
            f"{run_type}/coverage": float(coverage.item()),
            f"{run_type}/fcd": float(fcd.item()),
        }
        for key, value in caption_top1_metrics.items():
            result[f"{run_type}/caption_{key}_top1"] = float(value)
        if clip_score is not None:
            result[f"{run_type}/clip_score"] = clip_score
        return result
