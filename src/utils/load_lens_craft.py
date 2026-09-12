from copy import deepcopy
import os
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate, to_absolute_path

from models.camera_trajectory_model import LensCraft
from utils.checkpoint import load_checkpoint


def _memory_normalization_override(model_inference: DictConfig):
    if "camera_memory_normalization" not in model_inference:
        return None
    mode = model_inference.camera_memory_normalization
    if mode not in ("vocabulary", "none"):
        raise ValueError(
            "inference.camera_memory_normalization must be 'vocabulary' or 'none'; "
            "omit it to preserve the saved model configuration"
        )
    if mode == "vocabulary":
        for key in ("clip_model_name", "clip_embeddings_cache"):
            value = model_inference.get(key)
            if not isinstance(value, str) or not value.strip() or value == "None":
                raise ValueError(
                    f"inference.{key} is required for vocabulary camera memory normalization"
                )
    return mode


def load_lens_craft_model(model_module: DictConfig, model_inference: DictConfig, device: torch.device) -> LensCraft:
    mode = _memory_normalization_override(model_inference)
    config_path = model_inference.get("config")
    if config_path not in (None, "", "None"):
        checkpoint_cfg_path = to_absolute_path(os.path.expanduser(str(config_path)))
        if os.path.isfile(checkpoint_cfg_path):
            loaded_config = OmegaConf.load(checkpoint_cfg_path)
            if OmegaConf.select(loaded_config, "training.model.module") is not None:
                model_module = loaded_config.training.model.module
            elif OmegaConf.select(loaded_config, "ref_model.module") is not None:
                model_module = loaded_config.ref_model.module

    model_module = deepcopy(model_module)
    if mode is not None:
        with open_dict(model_module):
            if "camera_memory_norms" in model_module.keys():
                del model_module["camera_memory_norms"]
    model: LensCraft = instantiate(model_module)
    model = load_checkpoint(model_inference.checkpoint_path, model, device)
    if mode == "none":
        model.set_camera_memory_norms(None)
    elif mode == "vocabulary":
        if model.use_merged_memory or model.denormalize_memory:
            raise ValueError(
                "Vocabulary camera memory normalization requires use_merged_memory=false "
                "and denormalize_memory=false; select inference.camera_memory_normalization=none "
                "for this saved architecture"
            )
        from data.simulation.init_embeddings import initialize_all_clip_embeddings
        from data.simulation.memory_norms import vocabulary_memory_norms

        clip_model_name = model_inference.clip_model_name
        cache_file = to_absolute_path(os.path.expanduser(model_inference.clip_embeddings_cache))
        if os.path.isdir(cache_file):
            raise ValueError(f"inference.clip_embeddings_cache must be a cache file, not a directory: {cache_file}")
        embedding_dim = model.subject_trajectory_projection.out_features
        embeddings = initialize_all_clip_embeddings(
            clip_model_name=clip_model_name,
            cache_file=cache_file,
            embedding_dimension=embedding_dim,
        )
        model.set_camera_memory_norms(vocabulary_memory_norms(embeddings, embedding_dim))
    model.to(device)
    model.eval()

    return model
