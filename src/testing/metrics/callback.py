import os
from typing import Any, Dict, List, Optional

import torch

from testing.metrics.modules.caption_top1 import CaptionTop1
from utils.importing import ModuleImporter

_ET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_parties", "DIRECTOR")
)

with ModuleImporter.temporary_module(_ET_ROOT, replace_modules=["utils.rotation_utils"]):
    from src.metrics.modules.prdc import ManifoldMetrics
with ModuleImporter.temporary_module(_ET_ROOT):
    from src.metrics.modules.fcd import FrechetCLaTrDistance
    from src.metrics.modules.clatr_score import CLaTrScore as _CLaTrScoreBase


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

        clatr_score = m["clatr_score"].compute()
        m["clatr_score"].reset()

        precision, recall, density, coverage = m["clatr_prdc"].compute()
        m["clatr_prdc"].reset()

        fcd = m["clatr_fd"].compute()
        m["clatr_fd"].reset()

        caption_top1_metrics: Dict[str, float] = {}
        if "caption_top1" in m and self.clip_embeddings is not None:
            caption_top1_metrics = m["caption_top1"].compute()
            m["caption_top1"].reset()

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
        return result
