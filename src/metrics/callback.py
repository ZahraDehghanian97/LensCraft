from typing import Any, Dict, List

from src.metrics.modules.fcd import FrechetCLaTrDistance
from src.metrics.modules.prdc import ManifoldMetrics
from src.metrics.modules.text_trajectory_alignment_score import TextTrajectoryAlignmentScore


class MetricCallback:
    def __init__(
        self,
        num_cams: int,
        num_classes: int,
        device: str,
    ):
        self.num_cams = num_cams

        self.clip_fd = {
            "train": FrechetCLaTrDistance(),
            "val": FrechetCLaTrDistance(),
            "test": FrechetCLaTrDistance(),
        }
        self.clip_prdc = {
            "train": ManifoldMetrics(distance="euclidean"),
            "val": ManifoldMetrics(distance="euclidean"),
            "test": ManifoldMetrics(distance="euclidean"),
        }
        self.text_trajectory_alignment_score = {
            "train": TextTrajectoryAlignmentScore(),
            "val": TextTrajectoryAlignmentScore(),
            "test": TextTrajectoryAlignmentScore(),
        }

        self.device = device
        self._move_to_device(device)

    def _move_to_device(self, device: str):
        for stage in ["train", "val", "test"]:
            self.clip_fd[stage].to(device)
            self.clip_prdc[stage].to(device)
            self.text_trajectory_alignment_score[stage].to(device)

    def update_metrics(self, stage, pred, ref, text):
        self.text_trajectory_alignment_score[stage].update(pred, text)
        self.clip_prdc[stage].update(pred, ref)
        self.clip_fd[stage].update(pred, ref)

    def compute_metrics(self, stage: str) -> Dict[str, Any]:
        text_trajectory_alignment_score = self.text_trajectory_alignment_score[stage].compute()
        self.text_trajectory_alignment_score[stage].reset()

        clip_p, clip_r, clip_d, clip_c = self.clip_prdc[stage].compute()
        self.clip_prdc[stage].reset()

        fcd = self.clip_fd[stage].compute()
        self.clip_fd[stage].reset()

        return {
            "clip/alignment": text_trajectory_alignment_score,
            "clip/precision": clip_p,
            "clip/recall": clip_r,
            "clip/density": clip_d,
            "clip/coverage": clip_c,
            "clip/fcd": fcd,
        }
