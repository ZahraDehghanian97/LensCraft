"""The model's ten categorical cinematography fields, with 93 valid values."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import torch

from data.simulation.constants import CameraMovementType, cinematography_struct, simulation_struct
from data.simulation.utils import (
    cinematography_struct_parameters, create_structured_presence_mask,
    extract_cinematography_parameters,
)


FIELD_NAMES = tuple(name for name, _ in cinematography_struct_parameters)
FIELD_VALUES = tuple(
    tuple(value.value for value in kind) + (("nativePan", "nativeOrbit") if kind is CameraMovementType else ())
    for _, kind in cinematography_struct_parameters
)
FIELD_WIDTHS = tuple(map(len, FIELD_VALUES))
_SIGNED_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def _validate_ids(ids):
    if not torch.is_tensor(ids):
        raise TypeError("cinematography_ids must be a signed integer tensor [batch,10]")
    if ids.ndim != 2 or ids.shape[1] != len(FIELD_NAMES):
        raise ValueError("cinematography_ids must have shape [batch,10]")
    if ids.dtype not in _SIGNED_DTYPES:
        raise TypeError("cinematography_ids must have a signed integer dtype")
    widths = torch.tensor(FIELD_WIDTHS, device=ids.device)
    if ((ids < -1) | (ids >= widths)).any():
        raise ValueError("A cinematography ID is outside its field vocabulary; only -1 denotes unknown")
    return ids


def film_parameters_from_prompt(prompt):
    """Return ten ordered metadata tuples, including explicit native movements.

    Missing or unrecognized values use ID -1. Simulator instructions and numeric
    fields are excluded, and no text model, embedding cache, or fill is loaded.
    """
    if prompt is None:
        prompt = {}
    if not isinstance(prompt, Mapping):
        raise TypeError("The cinematography prompt must be a mapping or None")
    entries = []
    for section, fields in cinematography_struct:
        values = prompt.get(section)
        values = values if isinstance(values, Mapping) else {}
        for name, _ in fields:
            field = section + "_" + name
            vocabulary = FIELD_VALUES[len(entries)]
            raw = values.get(name)
            local_id = vocabulary.index(raw) if isinstance(raw, str) and raw in vocabulary else -1
            entries.append((field, raw, local_id, None))
    return entries


def film_ids_from_prompt(prompt, *, device=None):
    """Return the ten signed category IDs for one structured prompt."""
    return torch.tensor([entry[2] for entry in film_parameters_from_prompt(prompt)],
                        dtype=torch.long, device=device)


def metadata_from_prompt(prompt, instruction=None):
    """Return model IDs plus complete, embedding-free raw annotation metadata."""
    parameters = film_parameters_from_prompt(prompt)
    simulation = extract_cinematography_parameters(instruction, simulation_struct)
    ids = torch.tensor([entry[2] for entry in parameters], dtype=torch.long)
    return {"cinematography_ids": ids, "prompt_none_mask": ids >= 0,
            "cinematography_prompt_parameters": parameters,
            "simulation_instruction_parameters": simulation,
            "structured_presence_mask": create_structured_presence_mask(parameters, simulation)}


def cinematography_ids_from_batch(batch, *, device=None):
    """Prefer explicit IDs; otherwise read ordered source parameter tuples."""
    if batch.get("cinematography_ids") is not None:
        ids = _validate_ids(batch["cinematography_ids"])
        return ids.to(device=device) if device is not None else ids
    rows = batch.get("cinematography_prompt_parameters")
    if rows is None:
        raise ValueError(
            "Categorical conditioning requires cinematography_ids [batch,10] or "
            "cinematography_prompt_parameters; CLIP vectors cannot supply category IDs"
        )
    if not isinstance(rows, (list, tuple)):
        raise TypeError("cinematography_prompt_parameters must be a sequence of samples")
    ids = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < len(FIELD_NAMES):
            raise ValueError("Each categorical sample requires ten ordered cinematography parameters")
        indices = []
        for name, entry in zip(FIELD_NAMES, row):
            if not isinstance(entry, (list, tuple)) or len(entry) < 3 or entry[0] != name:
                raise ValueError("Cinematography parameter names must follow the declared field order")
            index = entry[2]
            if torch.is_tensor(index):
                if index.ndim != 0 or index.dtype not in _SIGNED_DTYPES:
                    raise TypeError("Category metadata IDs must be signed integer scalars")
                index = index.item()
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError("Category metadata IDs must be signed integer scalars")
            indices.append(int(index))
        ids.append(indices)
    return _validate_ids(torch.tensor(ids, dtype=torch.long, device=device).reshape(-1, len(FIELD_NAMES)))


def _conditioning_device(batch, model=None):
    for key in ("camera_trajectory", "subject_trajectory", "cinematography_ids"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value.device
    return getattr(model, "device", None)


def _categorical_inputs(batch, model=None):
    ids = cinematography_ids_from_batch(batch, device=_conditioning_device(batch, model))
    presence = ids >= 0
    declared = batch.get("prompt_none_mask")
    if declared is not None:
        if (not torch.is_tensor(declared) or declared.dtype != torch.bool
                or declared.shape != ids.shape):
            raise ValueError("Categorical prompt_none_mask must be boolean [batch,10]")
        presence = presence & declared.to(device=ids.device)
    return ids, presence


def conditioning_from_batch(batch, model=None):
    """Return categorical one-hots [10,batch,21]; missing fields stay zero."""
    from models.categorical_latent import from_ids

    ids, presence = _categorical_inputs(batch, model)
    values, _ = from_ids(ids)
    return values.masked_fill(~presence.T[..., None], 0)


def conditioning_presence_from_batch(batch, model=None):
    """Return the model's ten-field boolean presence mask [batch,10]."""
    return _categorical_inputs(batch, model)[1]


def conditioning_from_prompt(prompt, *, device=None):
    """Return (tokens[10,1,21], presence[1,10], ordered film metadata)."""
    from models.categorical_latent import from_ids

    parameters = film_parameters_from_prompt(prompt)
    ids = torch.tensor([[entry[2] for entry in parameters]], dtype=torch.long, device=device)
    values, presence = from_ids(ids)
    return values, presence, parameters


ids_from_batch = cinematography_ids_from_batch
