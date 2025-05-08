from typing import Any, Dict
import torch

from testing.metrics.modules.fcd import FrechetCLaTrDistance
from testing.metrics.modules.prdc import ManifoldMetrics
from testing.metrics.modules.clatr_score import CLaTrScore


class MetricCallback:
    def __init__(self, num_cams: int, device: torch.device):
        self.num_cams = num_cams
        self._device = device
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.active_metrics = set()

    def _get_or_create_metric(self, run_type: str):
        if run_type not in self.metrics:
            self.metrics[run_type] = {
                "clatr_fd": FrechetCLaTrDistance().to(self._device),
                "clatr_prdc": ManifoldMetrics(distance="euclidean").to(self._device),
                "clatr_score": CLaTrScore().to(self._device),
            }
        self.active_metrics.add(run_type)
        return self.metrics[run_type]

    def update_clatr_metrics(self, run_type: str, pred, ref, text):
        metrics = self._get_or_create_metric(run_type)

        pred = pred.to(self._device, dtype=torch.float16)
        ref = ref.to(self._device, dtype=torch.float16)

        if text is not None:
            text = text.to(self._device, dtype=torch.float16)
            metrics["clatr_score"].update(pred, text)

        metrics["clatr_prdc"].update(pred, ref)
        metrics["clatr_fd"].update(pred, ref)

    def compute_clatr_metrics(self, run_type: str) -> Dict[str, Any]:
        if run_type not in self.active_metrics:
            return {
                f"{run_type}/clatr_score": 0.0,
                f"{run_type}/precision": 0.0,
                f"{run_type}/recall": 0.0,
                f"{run_type}/density": 0.0,
                f"{run_type}/coverage": 0.0,
                f"{run_type}/fcd": 0.0,
            }

        metrics_dict = self.metrics[run_type]

        clatr_score = metrics_dict["clatr_score"].compute()
        metrics_dict["clatr_score"].reset()

        clatr_p, clatr_r, clatr_d, clatr_c = metrics_dict["clatr_prdc"].compute()
        metrics_dict["clatr_prdc"].reset()

        fcd = metrics_dict["clatr_fd"].compute()
        metrics_dict["clatr_fd"].reset()

        self.active_metrics.remove(run_type)

        with torch.no_grad():
            torch.cuda.empty_cache()

        return {
            f"{run_type}/clatr_score": float(clatr_score.item()),
            f"{run_type}/precision": float(clatr_p.item()),
            f"{run_type}/recall": float(clatr_r.item()),
            f"{run_type}/density": float(clatr_d.item()),
            f"{run_type}/coverage": float(clatr_c.item()),
            f"{run_type}/fcd": float(fcd.item()),
        }
