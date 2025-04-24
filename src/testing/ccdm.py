from typing import Dict, Any
import torch

from testing.metrics.callback import MetricCallback


def process_ccdm_batch(
    ccdm_adapter: Any,
    batch: Dict[str, torch.Tensor],
    metric_callback: MetricCallback,
    device: torch.device
) -> None:
    batch_data = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }
    
    generation = ccdm_adapter.process_batch(batch_data)
    
    if generation is None:
        return
    
    ground_truth = batch_data["camera_trajectory"]
    
    metric_callback.update(
        ground_truth=ground_truth,
        reconstruction=generation["reconstructed"]
    )
    
    batch_size = generation["reconstructed"].shape[0]
    
    real_features = ground_truth.reshape(batch_size, -1).detach().cpu().numpy()
    fake_features = generation["reconstructed"].reshape(batch_size, -1).detach().cpu().numpy()
    
    metric_callback.update_clatr_metrics(
        "reconstruction",
        fake_features,
        real_features,
        None
    )


