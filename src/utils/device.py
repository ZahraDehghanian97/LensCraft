from typing import Any, Dict

import torch


def resolve_device(cfg) -> torch.device:
    """Honor an explicit device, otherwise use CUDA when available."""
    if cfg.get("device"):
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Move nested tensors without changing the caller's batch containers.

    Physical geometry references and high-resolution views are nested mappings;
    their masks must share the device used by the corresponding model view.
    Non-tensor metadata remains unchanged.
    """
    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            items = [move(item) for item in value]
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return tuple(items)
        return value

    return move(batch)
