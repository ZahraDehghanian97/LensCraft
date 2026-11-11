"""Bounded cinematography probabilities, separate from attention field codes.

The shared semantic representation is ``[10, batch, 21]``. Each field uses
only its declared vocabulary; the remaining 117 entries are fixed zeros.
Missing annotations use an all-zero field and a separate presence mask.
This module returns that mask without choosing how a decoder attends to it.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn

from data.simulation.constants import (
    CameraMovementType,
    CameraVerticalAngle,
    MovementSpeed,
    ShotSize,
    SubjectInFramePosition,
    SubjectView,
)


FIELD_NAMES = (
    "initial_cameraAngle", "initial_shotSize", "initial_subjectView",
    "initial_subjectFraming", "movement_type", "movement_speed",
    "final_cameraAngle", "final_shotSize", "final_subjectView",
    "final_subjectFraming",
)
FIELD_WIDTHS = (5, 8, 8, 13, 21, 4, 5, 8, 8, 13)
FIELD_COUNT = len(FIELD_NAMES)
TOKEN_WIDTH = max(FIELD_WIDTHS)
SEMANTIC_DIM = sum(FIELD_WIDTHS)
NATIVE_MOVEMENT_VALUES = ("nativePan", "nativeOrbit")
MOVEMENT_VALUES = tuple(value.value for value in CameraMovementType) + NATIVE_MOVEMENT_VALUES
_FIELD_ENUMS = (
    CameraVerticalAngle, ShotSize, SubjectView, SubjectInFramePosition,
    CameraMovementType, MovementSpeed,
    CameraVerticalAngle, ShotSize, SubjectView, SubjectInFramePosition,
)
CATEGORY_VALUES = tuple(
    MOVEMENT_VALUES if field == "movement_type" else tuple(value.value for value in kind)
    for field, kind in zip(FIELD_NAMES, _FIELD_ENUMS)
)
FIELD_VALUES = CATEGORY_VALUES
if tuple(map(len, CATEGORY_VALUES)) != FIELD_WIDTHS:
    raise RuntimeError("The source cinematography vocabulary no longer matches the 93-value schema")

_SIGNED_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def category_specification() -> dict:
    """Return a serializable declaration of the categorical coordinate system."""
    return {
        "field_names": list(FIELD_NAMES),
        "field_widths": list(FIELD_WIDTHS),
        "category_values": [list(values) for values in CATEGORY_VALUES],
        "field_count": FIELD_COUNT,
        "token_width": TOKEN_WIDTH,
        "semantic_value_count": SEMANTIC_DIM,
        "fixed_padding_value_count": FIELD_COUNT * TOKEN_WIDTH - SEMANTIC_DIM,
        "output_shape": "[10,batch,21]",
        "presence_shape": "[batch,10]",
        "missing_id": -1,
        "missing_fill": "zero field with presence false",
        "clip_vectors_used": False,
        "trainable_parameter_count": 0,
    }


def valid_category_mask(device=None) -> torch.Tensor:
    """True for the 93 valid semantic entries, shaped ``[10, 21]``."""
    widths = torch.tensor(FIELD_WIDTHS, device=device)
    return torch.arange(TOKEN_WIDTH, device=device)[None, :] < widths[:, None]


def _check_semantic_shape(values: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(values):
        raise TypeError(f"{name} must be a tensor")
    if values.ndim != 3 or values.shape[0] != FIELD_COUNT or values.shape[2] != TOKEN_WIDTH:
        raise ValueError(f"{name} must have shape [10,batch,21]")
    if not values.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def masked_category_logits(logits: torch.Tensor) -> torch.Tensor:
    """Exclude padding before activation, leaving valid logits unmodified.

    This operation does not infer annotation presence. All ten predicted
    fields retain their own categorical distribution. Full value validation
    is available separately through :func:`validate_semantic`.
    """
    _check_semantic_shape(logits, "logits")
    valid = valid_category_mask(logits.device)[:, None, :]
    return logits.masked_fill(~valid, float("-inf"))


def masked_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Apply independent field softmax with exact-zero padding and gradients."""
    return torch.softmax(masked_category_logits(logits), dim=-1)


def _validate_ids(ids: torch.Tensor) -> None:
    if not torch.is_tensor(ids):
        raise TypeError("Category IDs must be a tensor")
    if ids.ndim != 2 or ids.shape[1] != FIELD_COUNT:
        raise ValueError("Category IDs must have shape [batch,10]")
    if ids.dtype not in _SIGNED_INTEGER_DTYPES:
        raise TypeError("Category IDs must have a signed integer dtype")
    widths = torch.tensor(FIELD_WIDTHS, device=ids.device)
    if ((ids < -1) | (ids >= widths[None, :])).any():
        raise ValueError("Category ID is outside its field vocabulary; only -1 denotes missing")


def _floating_dtype(dtype):
    dtype = torch.float32 if dtype is None else dtype
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("Semantic values and fixed codes require a floating-point dtype")
    return dtype


def from_ids(ids: torch.Tensor, *, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode signed field IDs as one-hot semantics and annotation presence.

    Values are local to each field. Native movement IDs 19 and 20 are valid
    only in the movement field. No CLIP vector or trainable table is used.
    """
    _validate_ids(ids)
    dtype = _floating_dtype(dtype)
    presence = ids >= 0
    values = torch.zeros(
        (ids.shape[0], FIELD_COUNT, TOKEN_WIDTH), device=ids.device, dtype=dtype,
    )
    values.scatter_(
        2, ids.clamp_min(0).long().unsqueeze(-1), presence.unsqueeze(-1).to(dtype),
    )
    return values.transpose(0, 1).contiguous(), presence


def categorical_presence(values: torch.Tensor) -> torch.Tensor:
    """Infer ``[batch,10]`` presence from nonzero valid semantic entries."""
    _check_semantic_shape(values, "Semantic values")
    valid = valid_category_mask(values.device)[:, None, :]
    return ((values != 0) & valid).any(dim=-1).transpose(0, 1)


def validate_semantic(
    values: torch.Tensor, presence: torch.Tensor | None = None, *, atol: float = 1e-5,
) -> None:
    """Check bounds, exact padding/absence zeros, and unit present-field sums.

    This explicit boundary check may synchronize tensor devices. It is not
    called by softmax in the encoder's forward path. ``atol`` applies only
    to unit sums: values must remain within [0,1], and zeros must be exact.
    """
    _check_semantic_shape(values, "Semantic values")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if not torch.isfinite(values).all():
        raise ValueError("Semantic probabilities must be finite")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Semantic probabilities must lie in [0,1]")
    valid = valid_category_mask(values.device)[:, None, :]
    if (values.masked_select(~valid) != 0).any():
        raise ValueError("Padded category entries must be exactly zero")
    if presence is None:
        presence = categorical_presence(values)
    if not torch.is_tensor(presence) or presence.dtype != torch.bool:
        raise TypeError("Semantic presence must be a boolean tensor")
    if presence.shape != (values.shape[1], FIELD_COUNT):
        raise ValueError("Semantic presence must have shape [batch,10]")
    if presence.device != values.device:
        raise ValueError("Semantic values and presence must share a device")
    if (values.transpose(0, 1).masked_select(~presence[..., None]) != 0).any():
        raise ValueError("Absent semantic fields must be exactly zero")
    sums = values.sum(dim=-1).transpose(0, 1)
    expected = presence.to(values.dtype)
    if not torch.isclose(sums, expected, atol=atol, rtol=0).all():
        raise ValueError("Every present semantic field must sum to one")


def ids_from_batch_parameters(batch_params, *, device=None) -> torch.Tensor:
    """Read first-ten source parameter tuples; ignore raw values and CLIP data.

    Each sample supplies at least ten ``(name, raw_value, index, ...)`` entries
    in source field order. Additional simulation entries never enter this
    representation. Native movement IDs are accepted directly; raw values
    are intentionally not reinterpreted through the legacy movement enum.
    """
    if not isinstance(batch_params, (list, tuple)):
        raise TypeError("Batch parameters must be a sequence of samples")
    rows = []
    for sample in batch_params:
        if not isinstance(sample, (list, tuple)) or len(sample) < FIELD_COUNT:
            raise ValueError("Each sample must contain at least ten field parameter tuples")
        row = []
        for field, entry in zip(FIELD_NAMES, sample[:FIELD_COUNT]):
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                raise ValueError("Each field parameter must contain name, raw value, and category ID")
            if entry[0] != field:
                raise ValueError("Cinematography parameter names must follow the declared field order")
            index = entry[2]
            if torch.is_tensor(index):
                if index.ndim != 0 or index.dtype not in _SIGNED_INTEGER_DTYPES:
                    raise TypeError("Parameter category IDs must be signed integer scalars")
                index = index.item()
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError("Parameter category IDs must be signed integer scalars")
            row.append(int(index))
        rows.append(row)
    ids = torch.tensor(rows, device=device, dtype=torch.long).reshape(-1, FIELD_COUNT)
    _validate_ids(ids)
    return ids


def get_targets(batch_params, *, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-hot categorical targets and presence from source metadata."""
    return from_ids(ids_from_batch_parameters(batch_params, device=device), dtype=dtype)


def fixed_sinusoidal_codes(count: int, width: int, *, device=None, dtype=None) -> torch.Tensor:
    """Return deterministic ``[count,width]`` sine/cosine identifiers.

    Codes are constants for queries or attention preparation, never semantic
    probabilities. Odd widths keep the final sine channel. No RNG is used.
    """
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("Code count must be a nonnegative integer")
    if isinstance(width, bool) or not isinstance(width, Integral) or width <= 0:
        raise ValueError("Code width must be a positive integer")
    dtype = _floating_dtype(dtype)
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    positions = torch.arange(count, device=device, dtype=compute_dtype)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=compute_dtype)
        * (-math.log(10000.0) / width)
    )
    codes = torch.zeros((count, width), device=device, dtype=compute_dtype)
    codes[:, 0::2] = torch.sin(positions * frequencies)
    codes[:, 1::2] = torch.cos(positions * frequencies[:width // 2])
    return codes.to(dtype=dtype)


def fixed_field_codes(width: int = TOKEN_WIDTH, *, device=None, dtype=None) -> torch.Tensor:
    """Fixed field identities for attention consumers, shaped ``[10,width]``."""
    return fixed_sinusoidal_codes(FIELD_COUNT, width, device=device, dtype=dtype)


def fixed_query_codes(width: int = 256, *, device=None, dtype=None) -> torch.Tensor:
    """Ten fixed semantic encoder queries, shaped ``[10,width]``."""
    return fixed_sinusoidal_codes(FIELD_COUNT, width, device=device, dtype=dtype)


class CategoricalLatentCodec(nn.Module):
    """Parameter-free module facade for the canonical semantic representation."""

    def __init__(self):
        super().__init__()
        self.register_buffer("valid_mask", valid_category_mask(), persistent=False)
        self.category_specification = category_specification()

    def forward(self, ids: torch.Tensor, *, dtype=None):
        return from_ids(ids, dtype=dtype)

    from_ids = staticmethod(from_ids)
    masked_category_logits = staticmethod(masked_category_logits)
    masked_softmax = staticmethod(masked_softmax)
    validate_semantic = staticmethod(validate_semantic)
    categorical_presence = staticmethod(categorical_presence)
    ids_from_batch_parameters = staticmethod(ids_from_batch_parameters)
    get_targets = staticmethod(get_targets)
