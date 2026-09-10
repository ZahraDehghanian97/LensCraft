import os
import logging

from dotenv import load_dotenv
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
import lightning as L
import torch
from data.datamodule import CameraTrajectoryDataModule
from data.multi_dataset_module import MultiDatasetModule
from training.module_factory import (
    build_lightning_init_kwargs,
    finite_validation_loss,
)

load_dotenv()

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    L.seed_everything(cfg.seed)

    use_multi_dataset = cfg.data.use_multi_dataset if hasattr(cfg.data, 'use_multi_dataset') else False

    if use_multi_dataset:
        data_module = MultiDatasetModule(
            simulation_config=cfg.data.dataset.simulation_config,
            ccdm_config=cfg.data.dataset.ccdm_config,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            val_size=cfg.data.val_size,
            test_size=cfg.data.test_size,
            sim_ratio=getattr(cfg.data, 'sim_ratio', 0.5)
        )
    else:
        data_module = CameraTrajectoryDataModule(
            dataset_config=cfg.data.dataset.config,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            val_size=cfg.data.val_size,
            test_size=cfg.data.test_size
        )

    model = instantiate(cfg.training.model.module)

    optimizer = instantiate(cfg.training.optimizer)
    lr_scheduler = instantiate(cfg.training.lr_scheduler)

    LightningModuleClass = get_class(cfg.training._target_)

    resume_checkpoint = getattr(cfg, "resume_checkpoint", None)
    checkpoint_path = None
    if resume_checkpoint not in (None, "", "None", "null"):
        checkpoint_path = str(resume_checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        loss_module = instantiate(cfg.training.loss_module)
        init_kwargs = build_lightning_init_kwargs(
            LightningModuleClass,
            cfg.training,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_module=loss_module,
            compile_mode=cfg.compile.mode,
            compile_enabled=cfg.compile.enabled,
            dataset_mode=getattr(data_module, 'dataset_mode', 'simulation'),
        )
        lightning_model = LightningModuleClass.load_from_checkpoint(
            checkpoint_path, map_location="cpu", **init_kwargs
        )
        logger.info("Checkpoint loaded successfully")
    else:
        lightning_model = instantiate(
            cfg.training,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            compile_mode=cfg.compile.mode,
            compile_enabled=cfg.compile.enabled,
            dataset_mode=getattr(data_module, 'dataset_mode', 'simulation'),
        )

    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    try:
        # Restore loop, optimizer, scheduler, and callback state as well as weights.
        trainer.fit(lightning_model, datamodule=data_module, ckpt_path=checkpoint_path)
    except KeyboardInterrupt:
        logger.info(
            "Training was interrupted by user; evaluating the current model "
            "to obtain a validation objective."
        )
    except Exception as e:
        logger.exception("Error during training: %s", e)
        raise

    validation_loss = trainer.callback_metrics.get("val_loss")
    if validation_loss is None:
        validation_loss = trainer.callback_metrics.get("val_loss_epoch")
    if validation_loss is None:
        validation_results = trainer.validate(
            lightning_model, datamodule=data_module, verbose=False
        )
        if validation_results:
            validation_loss = validation_results[0].get("val_loss")
            if validation_loss is None:
                validation_loss = validation_results[0].get("val_loss_epoch")

    validation_loss = finite_validation_loss(validation_loss)

    logger.info("Sweep objective val_loss: %.8f", validation_loss)
    return validation_loss


if __name__ == "__main__":
    main()
