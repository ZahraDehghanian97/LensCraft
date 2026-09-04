import inspect
import math
from collections.abc import Mapping
from typing import Any


_MISSING = object()

# Training behavior that is not encoded by the model itself must be restored
# from the current Hydra configuration when loading a Lightning checkpoint.
_OPTIONAL_TRAINING_FIELDS = (
    "use_merged_memory",
    "decode_mode",
    "use_cycle_consistency",
    "sim_weight",
    "ccdm_weight",
)


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, _MISSING)
    return getattr(config, key, _MISSING)


def build_lightning_init_kwargs(
    trainer_class: type,
    training_config: Any,
    *,
    model: Any,
    optimizer: Any,
    lr_scheduler: Any,
    loss_module: Any,
    compile_mode: str,
    compile_enabled: bool,
    dataset_mode: str,
) -> dict[str, Any]:
    """Build complete, signature-checked kwargs for checkpoint restoration."""

    parameters = inspect.signature(trainer_class.__init__).parameters
    accepts_extra = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    candidates = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "loss_module": loss_module,
        "noise": _config_value(training_config, "noise"),
        "mask": _config_value(training_config, "mask"),
        "teacher_forcing_schedule": _config_value(
            training_config, "teacher_forcing_schedule"
        ),
        "compile_mode": compile_mode,
        "compile_enabled": compile_enabled,
        "dataset_mode": dataset_mode,
    }
    for name in _OPTIONAL_TRAINING_FIELDS:
        value = _config_value(training_config, name)
        if value is not _MISSING:
            candidates[name] = value

    kwargs = {
        name: value
        for name, value in candidates.items()
        if value is not _MISSING and (accepts_extra or name in parameters)
    }
    missing = [
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
        and name not in kwargs
    ]
    if missing:
        raise TypeError(
            f"Missing required {trainer_class.__name__} constructor values: "
            f"{', '.join(missing)}"
        )
    return kwargs


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
