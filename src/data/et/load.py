import os

from hydra import initialize, compose
from hydra.utils import instantiate
from torch.utils.data import Dataset

from utils.importing import ModuleImporter


def load_et_config(project_config_dir: str, config_name: str="config.yaml" , dataset_dir: str = None, set_name: str = None):
    config_rel_path = os.path.dirname(
        os.path.relpath(project_config_dir, os.path.dirname(__file__)))
    
    overrides = []
    if set_name is not None:
        overrides.append(f"dataset.trajectory.set_name={set_name}")
    if dataset_dir is not None:
        overrides.append(f"data_dir={dataset_dir}")
    
    with initialize(version_base=None, config_path=config_rel_path):
        return compose(config_name=config_name, overrides=overrides)


def load_et_dataset(project_config_dir: str, dataset_dir: str, set_name: str, split: str) -> Dataset:
    director_cfg = load_et_config(project_config_dir, "config.yaml", dataset_dir, set_name)

    with ModuleImporter.temporary_module(os.path.dirname(os.path.dirname(project_config_dir)), ['utils.file_utils', 'utils.rotation_utils']):
        return instantiate(director_cfg.dataset).set_split(split)
