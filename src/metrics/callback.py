from typing import Any, Dict

from metrics.modules.fcd import FrechetCLaTrDistance
from metrics.modules.prdc import ManifoldMetrics
from metrics.modules.clatr_score import CLaTrScore


class MetricCallback:
    def __init__(self, num_cams: int, device: str):
        self.num_cams = num_cams

        # Initialize metrics for each run type
        self.clatr_fd = {
            "reconstruction": FrechetCLaTrDistance(),
            "prompt_generation": FrechetCLaTrDistance(), 
            "hybrid_generation": FrechetCLaTrDistance(),
        }
        self.clatr_prdc = {
            "reconstruction": ManifoldMetrics(distance="euclidean"),
            "prompt_generation": ManifoldMetrics(distance="euclidean"),
            "hybrid_generation": ManifoldMetrics(distance="euclidean"),
        }
        self.clatr_score = {
            "reconstruction": CLaTrScore(),
            "prompt_generation": CLaTrScore(),
            "hybrid_generation": CLaTrScore(),
        }

        self.device = device
        self._move_to_device(device)

    def _move_to_device(self, device: str):
        for run_type in ["reconstruction", "prompt_generation", "hybrid_generation"]:
            self.clatr_fd[run_type].to(device)
            self.clatr_prdc[run_type].to(device)
            self.clatr_score[run_type].to(device)

    def update_clatr_metrics(self, run_type: str, pred, ref, text):
        self.clatr_score[run_type].update(pred, text)
        self.clatr_prdc[run_type].update(pred, ref)
        self.clatr_fd[run_type].update(pred, ref)

    def compute_clatr_metrics(self, run_type: str) -> Dict[str, Any]:
        clatr_score = self.clatr_score[run_type].compute()
        self.clatr_score[run_type].reset()

        clatr_p, clatr_r, clatr_d, clatr_c = self.clatr_prdc[run_type].compute()
        self.clatr_prdc[run_type].reset()

        fcd = self.clatr_fd[run_type].compute()
        self.clatr_fd[run_type].reset()

        return {
            f"{run_type}/clatr_score": clatr_score.item(),
            f"{run_type}/precision": clatr_p.item(),
            f"{run_type}/recall": clatr_r.item(),
            f"{run_type}/density": clatr_d.item(),
            f"{run_type}/coverage": clatr_c.item(),
            f"{run_type}/fcd": fcd.item(),
        }
