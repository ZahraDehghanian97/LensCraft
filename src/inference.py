import logging
import os
import json
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from data.dataset_type import resolve_dataset_type
from inferencing.process import inference_batch
from models.factory import (
    load_model as _load_model,
    model_type_from_cfg as _model_type_from_cfg,
)
from utils.device import resolve_device as _resolve_device

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def tensor_to_serializable(obj: Any) -> Any:
    """Recursively convert tensors and numpy types into JSON-serializable values."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tensor_to_serializable(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "__dict__"):
        try:
            return tensor_to_serializable(obj.__dict__)
        except Exception:
            return str(obj)
    return obj


def _build_inference_result(
    first_batch,
    trajectories,
    sim_camera_trajectory,
    sim_subject_trajectory,
    sim_subject_volume,
    sim_padding_mask,
    key_framing_padding_mask,
    dataset_type,
    model_type,
):
    trajectories["GT"] = sim_camera_trajectory
    return {
        "trajectories": trajectories,
        "batch_data": {
            "subject_trajectory": sim_subject_trajectory,
            "subject_volume": sim_subject_volume,
            "padding_mask": sim_padding_mask,
            "text_prompts": first_batch.get("text_prompts"),
            "raw_prompt": first_batch.get("raw_prompt"),
            "raw_instruction": first_batch.get("raw_instruction"),
            "random_prompt_index": first_batch.get("random_prompt_index"),
            "key_framing_padding_mask": key_framing_padding_mask,
        },
        "dataset_type": dataset_type,
        "model_type": model_type,
    }


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig) -> None:
    device = _resolve_device(cfg)

    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    model_type = _model_type_from_cfg(cfg)

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()
    dataset_type = resolve_dataset_type(cfg.data.dataset.config["_target_"])

    model = _load_model(cfg, model_type, device)
    test_dataloader = data_module.test_dataloader()

    try:
        first_batch = next(iter(test_dataloader))
    except StopIteration:
        logger.error("No batches available in the test dataloader")
        print("No batches available in the test dataloader")
        return

    batch_size = len(first_batch["text_prompts"])
    first_batch["random_prompt_index"] = (
        np.random.randint(0, batch_size, size=batch_size).tolist()
    )

    with torch.no_grad():
        (
            trajectories,
            sim_camera_trajectory,
            sim_subject_trajectory,
            sim_subject_volume,
            sim_padding_mask,
            key_framing_padding_mask,
        ) = inference_batch(
            model,
            first_batch,
            device,
            dataset_type,
            model_type,
            seq_length=cfg.training.model.data_format.seq_length,
            num_keyframes=cfg.get("num_keyframes", 4),
            keyframe_sample_seeds=[
                int(cfg.get("seed", 42)) + i for i in range(batch_size)
            ],
        )

    result = _build_inference_result(
        first_batch,
        trajectories,
        sim_camera_trajectory,
        sim_subject_trajectory,
        sim_subject_volume,
        sim_padding_mask,
        key_framing_padding_mask,
        dataset_type,
        model_type,
    )

    output_file = os.path.join(os.getcwd(), "inference_result.json")
    with open(output_file, "w") as f:
        json.dump(tensor_to_serializable(result), f, indent=2)

    logger.info("Inference result saved to %s", output_file)
    print(f"Inference result saved to {output_file}")


if __name__ == "__main__":
    main()
