"""Raw structured annotation metadata and categorical model conditioning.

Simulator fields remain available for geometry/domain bookkeeping. Only the
first ten cinematography fields form the model's categorical semantics.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import math
from typing import Any, Dict, List

import torch

from .constants import CameraMovementType, NumericFeature, cinematography_struct, simulation_struct


def get_enum_index(enum_class, value) -> int:
    if isinstance(enum_class, type) and issubclass(enum_class, Enum):
        if enum_class is CameraMovementType and value in ("nativePan", "nativeOrbit"):
            return len(enum_class) + (value == "nativeOrbit")
        try:
            return list(enum_class).index(enum_class(value))
        except (TypeError, ValueError, KeyError):
            pass
    return -1


def _coerce_bool(value: Any):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "false"):
            return normalized == "true"
    return None


def _coerce_finite_number(value: Any):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_cinematography_parameters(data: Dict, struct: List, prefix: str = ""):
    """Read ordered ``(name, raw_value, local_id, None)`` metadata tuples.

    Unknown enums use -1; booleans use 0/1; finite numeric metadata uses 0.
    Numeric values and simulator fields never become model semantic tokens.
    """
    data = data if isinstance(data, Mapping) else {}
    parameters = []
    for key, kind in struct:
        field = f"{prefix}_{key}" if prefix else key
        raw = data.get(key)
        if isinstance(kind, list):
            parameters.extend(extract_cinematography_parameters(raw, kind, field))
            continue
        if kind is bool:
            raw = _coerce_bool(raw)
            index = -1 if raw is None else int(raw)
        elif isinstance(kind, NumericFeature):
            raw = _coerce_finite_number(raw)
            index = -1 if raw is None else 0
        elif isinstance(kind, type) and issubclass(kind, Enum):
            index = get_enum_index(kind, raw)
        else:
            raise TypeError("Unsupported annotation field: " + field)
        parameters.append((field, raw, index, None))
    return parameters


def count_total_parameters_in_struct(struct: List) -> int:
    return sum(count_total_parameters_in_struct(kind) if isinstance(kind, list) else 1
               for _, kind in struct)


def flatten_struct_parameters(struct: list, prefix: str = "") -> list:
    parameters = []
    for name, kind in struct:
        field = f"{prefix}_{name}" if prefix else name
        parameters.extend(flatten_struct_parameters(kind, field) if isinstance(kind, list)
                          else [(field, kind)])
    return parameters


def create_structured_presence_mask(cinematography_parameters, simulation_parameters):
    """Presence for all 46 raw fields, separate from the model's ten-field mask."""
    return torch.tensor([entry[2] >= 0 for entry in
                         list(cinematography_parameters) + list(simulation_parameters)],
                        dtype=torch.bool)


def structured_conditioning_from_batch(batch):
    """Return categorical semantics ``[10,batch,21]`` from integer metadata."""
    from data.categorical_conditioning import conditioning_from_batch

    return conditioning_from_batch(batch)


cinematography_struct_size = count_total_parameters_in_struct(cinematography_struct)
simulation_struct_size = count_total_parameters_in_struct(simulation_struct)
cinematography_struct_parameters = flatten_struct_parameters(cinematography_struct)
simulation_struct_parameters = flatten_struct_parameters(simulation_struct)
STRUCTURED_PARAMETERS = cinematography_struct_parameters + simulation_struct_parameters
STRUCTURED_PARAMETERS_DICT = dict(STRUCTURED_PARAMETERS)
