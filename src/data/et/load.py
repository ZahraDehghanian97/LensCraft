import os
from pathlib import Path

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from utils.importing import ModuleImporter


def load_et_config(project_config_dir: str, config_name: str="config.yaml" , dataset_dir: str=None, set_name: str=None, et_type=None):
    config_rel_path = os.path.dirname(
        os.path.relpath(project_config_dir, os.path.dirname(__file__)))
    
    overrides = []
    if set_name is not None:
        overrides.append(f"dataset.trajectory.set_name={set_name}")
    if dataset_dir is not None:
        overrides.append(f"data_dir={dataset_dir}")
    if et_type is not None:
        overrides.append(f"diffuser/network/module={et_type}_director")
        overrides.append(f"checkpoint_path=checkpoints/director/{et_type}-mixed-e449.ckpt")
    
    with initialize(version_base=None, config_path=config_rel_path):
        return compose(config_name=config_name, overrides=overrides)


class _RawCaptionDataset(Dataset):
    """Use DIRECTOR geometry and splits without its cached text features."""

    def __init__(self, native, caption_dir):
        self.native = native
        self.caption_dir = Path(caption_dir)
        self.root_filenames = native.root_filenames

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        item = dict(self.native[index])
        name = Path(item["traj_filename"]).stem
        item["caption_raw"] = {"caption": (self.caption_dir / (name + ".txt")).read_text()}
        return item


def load_et_dataset(project_config_dir: str, dataset_dir: str, set_name: str, split: str) -> Dataset:
    director_cfg = load_et_config(project_config_dir, "config.yaml", dataset_dir, set_name)
    # Resolve native geometry options before removing the caption modality:
    # upstream's descriptive dataset name refers to its caption configuration.
    configuration = OmegaConf.to_container(director_cfg.dataset, resolve=True)
    caption = configuration.pop("caption")
    configuration["num_cond_feats"] = [configuration["char"]["num_feats"]]

    with ModuleImporter.temporary_module(os.path.dirname(os.path.dirname(project_config_dir)), ['utils.file_utils', 'utils.rotation_utils']):
        native = instantiate(configuration).set_split(split)
    return _RawCaptionDataset(native, caption["raw_caption_dir"])
