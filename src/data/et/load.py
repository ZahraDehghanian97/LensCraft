import os

from hydra import initialize, compose
from hydra.utils import instantiate
from torch.utils.data import Dataset

from utils.importing import ModuleImporter


def load_et_dataset(project_config_dir: str, dataset_dir: str, set_name: str, split: str) -> Dataset:
    config_rel_path = os.path.dirname(
        os.path.relpath(project_config_dir, os.path.dirname(__file__)))
    with initialize(version_base=None, config_path=config_rel_path):
        director_cfg = compose(config_name="config.yaml", overrides=[
            f"dataset.trajectory.set_name={set_name}",
            f"data_dir={dataset_dir}"
        ])

    with ModuleImporter.temporary_module(os.path.dirname(os.path.dirname(project_config_dir)), ['utils.file_utils', 'utils.rotation_utils']):
        return instantiate(director_cfg.dataset).set_split(split)
