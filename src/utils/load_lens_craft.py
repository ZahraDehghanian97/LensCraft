"""Strict loading of the sole ten-token categorical LensCraft generator."""
from collections.abc import Mapping
from copy import deepcopy
import os

import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict

from models.camera_trajectory_model import LensCraft
from models.categorical_latent import TOKEN_WIDTH

_MODEL_TARGET = "models.camera_trajectory_model.LensCraft"
_REMOVED_SETTINGS = (
    "conditioning_representation", "use_merged_memory", "denormalize_memory",
    "camera_memory_norms", "caption_presence_conditioning",
)
_OLD_CHECKPOINT_ERROR = (
    "This checkpoint is not a ten-token categorical LensCraft generator. "
    "CLIP, old ID adapters, and robust-wrapper generator checkpoints are no longer supported; "
    "select the categorical 100-epoch checkpoint or another categorical checkpoint."
)


def categorical_checkpoint_state(payload):
    """Extract current standalone or Lightning weights without legacy remapping."""
    if not isinstance(payload, Mapping):
        raise ValueError("LensCraft checkpoint must contain a mapping")
    layouts = [key for key in ("state_dict", "model") if key in payload]
    if len(layouts) != 1:
        raise ValueError(_OLD_CHECKPOINT_ERROR)
    source = payload[layouts[0]]
    if not isinstance(source, Mapping) or not source:
        raise ValueError("Categorical checkpoint weights must be a nonempty mapping")
    state = {}
    for key, value in source.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("Categorical checkpoint weights must map parameter names to tensors")
        name = key.removeprefix("model.") if layouts[0] == "state_dict" else key
        name = name.removeprefix("_orig_mod.")
        if name in state:
            raise ValueError(f"Duplicate categorical checkpoint key: {name}")
        state[name] = value
    head = state.get("encoder.semantic_projection.weight")
    keys = state.get("decoder.transformer_decoder.layers.0.multihead_attn.k_proj_weight")
    values = state.get("decoder.transformer_decoder.layers.0.multihead_attn.v_proj_weight")
    if (head is None or head.ndim != 2 or head.shape[0] != TOKEN_WIDTH
            or keys is None or keys.ndim != 2 or keys.shape[1] != TOKEN_WIDTH
            or values is None or values.shape != keys.shape):
        raise ValueError(_OLD_CHECKPOINT_ERROR)
    return state, layouts[0]


def categorical_model_configuration(model_module, model_inference):
    """Migrate redundant saved categorical settings without resolving CLIP nodes."""
    config_path = model_inference.get("config")
    if config_path not in (None, "", "None"):
        path = to_absolute_path(os.path.expanduser(str(config_path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Saved LensCraft configuration does not exist: {path}")
        saved = OmegaConf.load(path)
        candidate = OmegaConf.select(saved, "training.model.module")
        if candidate is None:
            candidate = OmegaConf.select(saved, "model.module")
        if candidate is None:
            raise ValueError("Saved configuration must contain training.model.module")
        model_module = candidate
    module = deepcopy(model_module)
    if not OmegaConf.is_config(module):
        module = OmegaConf.create(module)
    if module.get("_target_") != _MODEL_TARGET:
        raise ValueError(f"Categorical generation requires {_MODEL_TARGET}")
    representation = module.get("conditioning_representation")
    if representation not in (None, "categorical"):
        raise ValueError(_OLD_CHECKPOINT_ERROR)
    # Weight schema is checked first. Dropping obsolete options changes no
    # learned computation and cannot turn an old generator into this model.
    with open_dict(module):
        for key in _REMOVED_SETTINGS:
            if key in module:
                del module[key]
    return module


def load_categorical_generator(model_module, model_inference, device, *, checkpoint_payload=None):
    """Return a strictly restored categorical generator and load provenance."""
    payload = (torch.load(model_inference.checkpoint_path, map_location="cpu", weights_only=False)
               if checkpoint_payload is None else checkpoint_payload)
    state, source_format = categorical_checkpoint_state(payload)
    module = categorical_model_configuration(model_module, model_inference)
    model = instantiate(module)
    if type(model) is not LensCraft:
        raise ValueError(f"Categorical generation requires {_MODEL_TARGET}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {
        "source_format": source_format,
        "source_protocol": payload.get("protocol"),
        "runtime_implementation": _MODEL_TARGET,
        "semantic_shape": [10, "batch", 21],
        "semantic_coordinate_count": 93,
        "config_schema_migration": "remove redundant categorical constructor settings",
    }


def load_lens_craft_model(model_module: DictConfig, model_inference: DictConfig, device: torch.device) -> LensCraft:
    return load_categorical_generator(model_module, model_inference, device)[0]
