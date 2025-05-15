import os
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path

from models.camera_trajectory_model import LensCraft
from utils.checkpoint import load_checkpoint


def load_lens_craft_model(model_module: DictConfig, model_inference: DictConfig, device: torch.device) -> LensCraft:
    if model_inference.config and model_inference.config != 'None':
        checkpoint_cfg_path = to_absolute_path(model_inference.config)
        if os.path.exists(checkpoint_cfg_path):
            loaded_config = OmegaConf.load(checkpoint_cfg_path)
            if 'ref_model' in loaded_config and 'module' in loaded_config.ref_model:
                model_module = loaded_config.ref_model.module
            elif 'training' in loaded_config and 'model' in loaded_config.training and 'module' in loaded_config.training.model:
                model_module = loaded_config.training.model.module
    
    model: LensCraft = instantiate(model_module)
    model = load_checkpoint(model_inference.checkpoint_path, model, device)
    model.to(device)
    model.eval()
    
    return model
