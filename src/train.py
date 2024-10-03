import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import lightning as L
from torch.utils.data import random_split, DataLoader
from data.simulation.dataset import batch_collate


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(42)

    full_dataset = instantiate(cfg.dataset)

    train_size = int((1 - cfg.data.val_size) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=batch_collate
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=batch_collate
    )

    model = instantiate(cfg.model)

    lightning_model = instantiate(cfg.lightning_model, model=model)

    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(lightning_model, train_dataloader, val_dataloader)

    print("Training completed!")
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
