import sys
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from hydra.core.global_hydra import GlobalHydra

sys.path.append("third_parties/DIRECTOR/")


class CustomETDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.original_dataset = hydra.utils.instantiate(
            cfg.dataset).set_split(split)

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

    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    hydra.initialize(version_base="1.3",
                     config_path="../../third_parties/DIRECTOR/configs")
    cfg = hydra.compose(config_name="config.yaml", overrides=overrides)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    dataset = CustomETDataset(cfg, split="train")

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.compnode.num_workers,
        pin_memory=True,
    )

    first_batch = next(iter(train_dataloader))
    print(first_batch)


if __name__ == "__main__":
    main()
