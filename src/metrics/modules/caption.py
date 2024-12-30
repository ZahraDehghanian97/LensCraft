from typing import Tuple

from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import torchmetrics.functional as F


class CaptionMetrics(Metric):
    """Metrics for caption evaluation including precision, recall, and F1 score."""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.metric_kwargs = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "weighted",
            "zero_division": 0,
        }
        
        self.add_state("pred", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds, labels):
        self.target.append(labels)
        self.pred.append(preds)

    def compute(self) -> Tuple[float, float, float]:
        target = dim_zero_cat(self.target)
        pred = dim_zero_cat(self.pred)

        return (
            F.precision(pred, target, **self.metric_kwargs),
            F.recall(pred, target, **self.metric_kwargs),
            F.f1_score(pred, target, **self.metric_kwargs)
        )
