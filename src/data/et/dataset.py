import os
import sys
from typing import Any, Dict

import hydra
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from hydra.core.global_hydra import GlobalHydra
from dotenv import load_dotenv

try:
    PROJECT_PATH = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../..'))
    if PROJECT_PATH not in sys.path:
        sys.path.insert(0, PROJECT_PATH)

    from utils.importing import ModuleImporter
except ImportError:
    raise


class ETDataset(Dataset):
    def __init__(self, director_project_config_path: str, data_dir: str, set_name: str, split: str):
        self._initialize_dataset(
            director_project_config_path, data_dir, set_name, split
        )

    def _initialize_dataset(self, config_path: str, data_dir: str, set_name: str, split: str) -> Any:
        self.config_path = config_path
        config_rel_path = os.path.dirname(
            os.path.relpath(self.config_path, os.path.dirname(__file__)))
        with initialize(version_base=None, config_path=config_rel_path):
            director_cfg = compose(config_name="config.yaml", overrides=[
                f"dataset.trajectory.set_name={set_name}",
                f"data_dir={data_dir}"
            ])

        with ModuleImporter.temporary_module(os.path.dirname(os.path.dirname(self.config_path)), ['utils.file_utils', 'utils.rotation_utils']):
            self.original_dataset = instantiate(
                director_cfg.dataset).set_split(split)

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        original_item = self.original_dataset[index]
        return self.process_item(original_item)

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return item


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    GlobalHydra.instance().clear()

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    load_dotenv()

    dataset = instantiate(cfg.data.dataset)
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")


if __name__ == "__main__":
    main()
