import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import lightning as L
from data.datamodule import CameraTrajectoryDataModule
from dotenv import load_dotenv
load_dotenv()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    L.seed_everything(cfg.seed)

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size
    )

    model = instantiate(cfg.training.model)

    lightning_model = instantiate(
        cfg.training, model=model, compile_mode=cfg.compile.mode, compile_enabled=cfg.compile.enabled, dataset_mode=data_module.dataset_mode)

    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(lightning_model, datamodule=data_module)

    print("Training completed!")
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    trainer.test(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    main()
