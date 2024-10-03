import sys
from copy import deepcopy
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from hydra.core.global_hydra import GlobalHydra

sys.path.append("third_parties/DIRECTOR/")


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

    dataset = hydra.utils.instantiate(cfg.dataset)

    train_dataset = deepcopy(dataset).set_split("train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.compnode.num_workers,
        pin_memory=True,
    )

    first_batch = next(iter(train_dataloader))
    print(first_batch)


if __name__ == "__main__":
    main()
