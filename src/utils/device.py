from typing import Any, Dict

import torch


def move_batch_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
