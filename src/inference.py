import logging
import os
import json
import numpy as np
from typing import Literal, Any

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from utils.load_lens_craft import load_lens_craft_model
from inferencing.process import inference_batch
from models.ccdm_adapter import CCDMAdapter
from models.et_adapter import ETAdapter

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

DatasetType = Literal["ccdm", "et", "simulation"]

class TensorEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle tensors and other non-serializable types."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        try:
            return super().default(obj)
        except:
            return str(obj)

def tensor_to_serializable(obj: Any) -> Any:
    """Recursively convert tensors and other objects to JSON-serializable types."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return [tensor_to_serializable(i) for i in obj]
    elif hasattr(obj, '__dict__'):
        try:
            return tensor_to_serializable(obj.__dict__)
        except:
            return str(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig) -> None:
    if cfg.get("device"):
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    data_format_type = cfg.training.model.data_format.get("type", "simulation")
    model_type = "lens_craft" if data_format_type == "simulation" else data_format_type
    
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()
    
    target = cfg.data.dataset.config["_target_"]
    dataset_type = "ccdm" if "CCDMDataset" in target else "et" if "ETDataset" in target else "simulation"
    
    model = None
    
    if model_type == "lens_craft":
        model = load_lens_craft_model(model_module=cfg.training.model.module, model_inference=cfg.training.model.inference, device=device)
    else:
        if model_type == "ccdm":
            model = CCDMAdapter(cfg.training.model.inference, device)
        elif model_type == "et":
            model = ETAdapter(cfg.training.model.inference, device)

    test_dataloader = data_module.test_dataloader()

    with torch.no_grad():
        try:
            first_batch = next(iter(test_dataloader))
            batch_size = len(first_batch['text_prompts'])
            
            first_batch['random_prompt_index'] = np.random.randint(0, batch_size, size=batch_size).tolist()
            
            trajectories =\
                inference_batch(model, first_batch, device, dataset_type, model_type, seq_length=cfg.training.model.data_format.seq_length)
            
            result = {
                "trajectories": trajectories,
                "batch_data": {
                    "camera_trajectory": first_batch.get("camera_trajectory"),
                    "subject_trajectory": first_batch.get("subject_trajectory"),
                    "subject_volume": first_batch.get("subject_volume"),
                    "padding_mask": first_batch.get("padding_mask"),
                    "text_prompts": first_batch.get("text_prompts"),
                    "raw_prompt": first_batch.get("raw_prompt"),
                    "raw_instruction": first_batch.get("raw_instruction"),
                    "random_prompt_index": first_batch.get("random_prompt_index"),
                },
                "dataset_type": dataset_type,
                "model_type": model_type,
            }
            
            json_serializable_result = tensor_to_serializable(result)
            
            output_file = os.path.join(os.getcwd(), "inference_result.json")
            with open(output_file, 'w') as f:
                json.dump(json_serializable_result, f, cls=TensorEncoder, indent=2)
            
            logger.info(f"Inference result saved to {output_file}")
            print(f"Inference result saved to {output_file}")
            
        except StopIteration:
            logger.error("No batches available in the test dataloader")
            print("No batches available in the test dataloader")

if __name__ == "__main__":
    main()
