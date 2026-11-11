"""Load the categorical generator for semantic trajectory editing."""
from collections.abc import Mapping

import torch

from utils.load_lens_craft import load_categorical_generator


def read_edit_checkpoint(path):
    """Read once on CPU; callers may reuse the payload during dataset setup."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Edit checkpoint must contain a checkpoint mapping")
    return payload


def _normalization_vector(parameters, group, statistic, source):
    try:
        value = torch.as_tensor(parameters[group][statistic], dtype=torch.float64).detach().cpu()
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{source} requires normalization {group}.{statistic}") from error
    if (value.shape != (3,) or not torch.isfinite(value).all()
            or (statistic == "std" and (value <= 0).any())):
        raise ValueError(f"{source} normalization {group}.{statistic} must be finite [3] with positive std")
    return value


def load_edit_generator(model_module, model_inference, device, *, normalization_parameters,
                        checkpoint_payload=None):
    payload = (read_edit_checkpoint(model_inference.checkpoint_path)
               if checkpoint_payload is None else checkpoint_payload)
    saved = payload.get("normalization") if isinstance(payload, Mapping) else None
    if saved is not None:
        if not isinstance(saved, Mapping) or not isinstance(normalization_parameters, Mapping):
            raise ValueError("Saved and input normalization must be mappings")
        for group in ("camera_position", "subject_position", "subject_dimensions"):
            for statistic in ("mean", "std"):
                actual = _normalization_vector(saved, group, statistic, "Checkpoint")
                supplied = _normalization_vector(normalization_parameters, group, statistic, "Input")
                if not torch.allclose(actual, supplied, rtol=1e-6, atol=1e-8):
                    raise ValueError(f"Checkpoint/input normalization mismatch: {group}.{statistic}")
    model, provenance = load_categorical_generator(
        model_module, model_inference, device, checkpoint_payload=payload,
    )
    provenance.update({
        "normalization_verified": saved is not None,
        "normalization_tolerance": {"rtol": 1e-6, "atol": 1e-8} if saved is not None else None,
        "editing_implementation": "categorical caption_edit_mask field substitution",
    })
    return model, provenance
