import math
from collections.abc import Mapping
from typing import Any


_MISSING = object()


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, _MISSING)
    return getattr(config, key, _MISSING)


def finite_validation_loss(value: Any) -> float:
    """Convert a callback/validate result to the finite sweep objective."""

    if value is None:
        raise RuntimeError("Training produced no validation loss for sweep objective")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()

    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Validation loss is not finite: {result}")
    return result


def checkpoint_monitor(requested: str, training_config: Any) -> str:
    """Select the same objective for checkpointing, stopping and sweep return."""
    if requested != 'auto':
        if not isinstance(requested, str) or not requested:
            raise ValueError('checkpoint_monitor must be auto or a logged metric name')
        return requested
    validation = _config_value(training_config, 'validation_conditioning')
    enabled = _config_value(validation, 'enabled') if validation is not _MISSING and validation is not None else False
    # A present partial configuration uses ConditioningValidation's defaults.
    return 'val_conditioning_score' if enabled is _MISSING or enabled else 'val_loss'
