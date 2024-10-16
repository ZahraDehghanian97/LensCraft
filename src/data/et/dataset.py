import os

from torch.utils.data import Dataset
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.path import temporary_sys_path


class ETDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.original_dataset = instantiate(cfg).set_split(split)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        original_item = self.original_dataset[index]
        processed_item = self.process_item(original_item)
        return processed_item

    def process_item(self, item):
        return item


def main():
    overrides = [
        "dataset.trajectory.set_name=mixed",
        "data_dir=/Users/reza/Projects/ET/et-data"
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    director_path = os.path.abspath(os.path.join(
        current_dir, "..", "..", "..", "third_parties", "DIRECTOR"))
    config_dir = os.path.join(director_path, "configs")

    with temporary_sys_path(director_path):
        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = compose(config_name="config.yaml", overrides=overrides)

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        dataset = ETDataset(cfg.dataset)

    print(dataset[0])


if __name__ == "__main__":
    main()
