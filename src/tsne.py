import torch
import os
import logging
from omegaconf import DictConfig
import hydra
from dotenv import load_dotenv

from visualization.utils import tSNE_visualize_embeddings


load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="tsne")
def main(cfg: DictConfig) -> None:

    features_path = cfg.feature_path
    save_path = os.path.join(features_path, "Embedding Visualization using t-SNE.png")

    features_lens_craft_path = os.path.join(features_path, "dataset_simulation_model_lens_craft.pth")
    features_et_path = os.path.join(features_path, "dataset_simulation_model_et.pth")
    features_ccdm_path = os.path.join(features_path, "dataset_simulation_model_ccdm.pth")
    
    prompt_generation_lens_craft = torch.load(features_lens_craft_path)["prompt_generation"]
    prompt_generation_et = torch.load(features_et_path)["prompt_generation"]
    prompt_generation_ccdm = torch.load(features_ccdm_path)["prompt_generation"]
    
    all_prompt_generations = dict()

    all_prompt_generations["LensCraft Features"] = prompt_generation_lens_craft["GEN"]
    all_prompt_generations["ET Features"] = prompt_generation_et["GEN"]
    all_prompt_generations["CCDM Features"] = prompt_generation_ccdm["GEN"]
    all_prompt_generations["GT"] = prompt_generation_lens_craft["GT"]


    tSNE_visualize_embeddings(
        all_prompt_generations,
        title=f"Embedding Visualization using t-SNE",
        save_path=save_path,
    )


if "__main__" == __name__:
    main()