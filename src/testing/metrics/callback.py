from typing import Any, Dict
import torch

from testing.metrics.modules.fcd import FrechetCLaTrDistance
from testing.metrics.modules.prdc import ManifoldMetrics
from testing.metrics.modules.clatr_score import CLaTrScore


class MetricCallback:
    def __init__(self, num_cams: int, device: str):
        self.num_cams = num_cams
        self.device = device
        self.metrics = {}
        self.active_metrics = set()
        
    def _get_or_create_metric(self, run_type: str):
        if run_type not in self.metrics:
            self.metrics[run_type] = {
                "clatr_fd": FrechetCLaTrDistance(),
                "clatr_prdc": ManifoldMetrics(distance="euclidean"),
                "clatr_score": CLaTrScore()
            }
            for metric in self.metrics[run_type].values():
                metric.to(self.device)
                
        self.active_metrics.add(run_type)
        return self.metrics[run_type]

    def update_clatr_metrics(self, run_type: str, pred, ref, text):
        metrics = self._get_or_create_metric(run_type)
        
        pred = pred.to(torch.float16)
        ref = ref.to(torch.float16)
        
        if text is not None:
            text = text.to(torch.float16)
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
            
        metrics = self.metrics[run_type]
        
        clatr_score = metrics["clatr_score"].compute()
        metrics["clatr_score"].reset()

        clatr_p, clatr_r, clatr_d, clatr_c = metrics["clatr_prdc"].compute()
        metrics["clatr_prdc"].reset()

        fcd = metrics["clatr_fd"].compute()
        metrics["clatr_fd"].reset()
        
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
