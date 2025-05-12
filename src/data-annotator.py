import logging
import json
from pathlib import Path
from typing import Literal, List, Tuple
import math

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from dotenv import load_dotenv

from data.datamodule import CameraTrajectoryDataModule
from annotator.constants import system_prompt, CINEMATOGRAPHY_JSON_SCHEMA

load_dotenv()
logger = logging.getLogger(__name__)

DatasetType = Literal["ccdm", "et", "simulation"]


def create_openai_batch_file(prompt_id_pairs: List[Tuple[str, str]], output_file: str) -> None:
    tasks = []
    for prompt, item_id in prompt_id_pairs:
        task = {
            "custom_id": item_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-nano",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": CINEMATOGRAPHY_JSON_SCHEMA,
                },
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Convert this camera movement description into a "
                            "cinematographyPrompts JSON object that matches the provided schema:\n"
                            f"\"{prompt}\""
                        ),
                    },
                ],
            },
        }
        tasks.append(task)

    with open(output_file, "w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")

    logger.info("Created batch file with %s tasks at %s", len(tasks), output_file)


def split_into_batches(items, num_batches=5):
    """Split a list into specified number of roughly equal batches."""
    batch_size = math.ceil(len(items) / num_batches)
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


@hydra.main(version_base=None, config_path="../config", config_name="annotator")
def main(cfg: DictConfig) -> None:
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    batch_dir = Path(cfg.output.batch_dir)
    results_dir = Path(cfg.output.results_dir)
    output_dir = Path(cfg.output.annotations_dir)

    for p in (batch_dir, results_dir, output_dir):
        p.mkdir(exist_ok=True)

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=2,
        val_size=0,
        test_size=1,
    )
    data_module.setup()
    test_dl = data_module.test_dataloader()

    logger.info("Dataset size: %d", len(test_dl.dataset))

    prompt_id_pairs: List[Tuple[str, str]] = []
    
    target = cfg.data.dataset.config["_target_"]
    dataset_type = (
        "ccdm" if "CCDMDataset" in target else "et" if "ETDataset" in target else "simulation"
    )
    
    for batch in tqdm(test_dl, desc="Collecting text prompts and IDs"):
        ids = [f"{dataset_type}-{id}" for id in
               (batch["item_ids"] if dataset_type == "et" else range(len(batch["text_prompts"])))]
        prompt_id_pairs.extend(list(zip(batch["text_prompts"], ids)))
    
    batches = split_into_batches(prompt_id_pairs, num_batches=5)
    
    for i, batch in enumerate(batches):
        batch_file = batch_dir / f"{dataset_type}_dataset_batch_{i+1}.jsonl"
        create_openai_batch_file(batch, str(batch_file))
    
    logger.info(f"Created 5 batch files in {batch_dir}")

if __name__ == "__main__":
    main()
