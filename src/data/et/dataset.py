import sys
from torch.utils.data import Dataset
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.append("third_parties/DIRECTOR/")


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

    with initialize(version_base="1.3", config_path="../../../third_parties/DIRECTOR/configs"):
        cfg = compose(config_name="config.yaml", overrides=overrides)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    dataset = ETDataset(cfg.dataset)

    print(dataset[0])


if __name__ == "__main__":
    main()
