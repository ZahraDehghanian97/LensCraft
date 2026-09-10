"""Select generation models without importing unrelated baseline dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig


def model_type_from_cfg(cfg: DictConfig) -> str:
    data_format_type = cfg.training.model.data_format.get("type", "simulation")
    return "lens_craft" if data_format_type == "simulation" else data_format_type


def load_model(cfg: DictConfig, model_type: str, device: torch.device):
    """Load the configured model, importing only its implementation."""
    if model_type == "lens_craft":
        from utils.load_lens_craft import load_lens_craft_model

        return load_lens_craft_model(
            model_module=cfg.training.model.module,
            model_inference=cfg.training.model.inference,
            device=device,
        )
    if model_type == "ccdm":
        from models.baselines.ccdm_adapter import CCDMAdapter

        return CCDMAdapter(cfg.training.model.inference, device)
    if model_type == "et":
        from models.baselines.et_adapter import ETAdapter

        return ETAdapter(cfg.training.model.inference, device)
    if model_type == "gendop":
        from models.baselines.gendop_adapter import GenDoPAdapter

        return GenDoPAdapter(cfg.training.model.inference, device)
    raise ValueError(f"Unsupported model type: {model_type}")
