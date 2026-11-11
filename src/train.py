import os
import logging

from dotenv import load_dotenv
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import lightning as L
import torch
from data.datamodule import CameraTrajectoryDataModule
from data.multi_dataset_module import MultiDatasetModule
from training.module_factory import (
    finite_validation_loss,
    checkpoint_monitor,
)
from utils.checkpoint import load_checkpoint
from utils.normalization import prepare_evaluation_normalization

load_dotenv()

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def _compact_batches_enabled(cfg: DictConfig) -> bool:
    return bool(getattr(cfg.data, "compact_batches", False))


def _validate_multi_config(cfg: DictConfig) -> None:
    if not OmegaConf.select(cfg, "data.use_multi_dataset", default=False):
        return
    if (OmegaConf.select(cfg, "training._target_", default="")
            != "training.multi_dataset_trainer.MultiDatasetTrainer"):
        raise ValueError("Mixed-dataset training requires MultiDatasetTrainer")
    annotations_path = OmegaConf.select(cfg, "data.dataset.ccdm_config.annotations_path")
    if annotations_path is None or not str(annotations_path).strip():
        raise ValueError(
            "Mixed categorical training requires structured CCDM caption annotations. "
            "Set CCDM_ANNOTATIONS_PATH or "
            "data.dataset.ccdm_config.annotations_path=/path/to/annotations.json. "
            "Rows must contain custom_id, source_prompt, cinematographyPrompts and "
            "simulationInstructions for every selected CCDM sample; native caption "
            "text alone does not condition the categorical generator."
        )
    annotations_path = hydra.utils.to_absolute_path(os.path.expanduser(str(annotations_path)))
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(
            f"CCDM training annotations not found: {annotations_path}. "
            "Set CCDM_ANNOTATIONS_PATH or data.dataset.ccdm_config.annotations_path "
            "to an existing annotations.json file."
        )
    # Hydra changes into the output directory; keep configured relative paths
    # anchored to the original project directory for the subsequent loader.
    with open_dict(cfg.data.dataset.ccdm_config):
        cfg.data.dataset.ccdm_config.annotations_path = annotations_path


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    L.seed_everything(cfg.seed)
    _validate_multi_config(cfg)

    monitor = checkpoint_monitor(cfg.get('checkpoint_monitor', 'auto'), cfg.training)
    with open_dict(cfg):
        cfg.checkpoint_monitor = monitor
    initial_checkpoint = cfg.get('initialize_from_checkpoint')
    resume_checkpoint = cfg.get('resume_checkpoint')
    empty = (None, '', 'None', 'null')
    if initial_checkpoint not in empty and resume_checkpoint not in empty:
        raise ValueError('initialize_from_checkpoint and resume_checkpoint are mutually exclusive')

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
            test_size=cfg.data.test_size,
            compact_batches=_compact_batches_enabled(cfg),
        )

    model = instantiate(cfg.training.model.module)
    if initial_checkpoint not in empty:
        initial_path = hydra.utils.to_absolute_path(os.path.expanduser(str(initial_checkpoint)))
        saved_checkpoint = torch.load(initial_path, map_location='cpu', weights_only=False)
        if saved_checkpoint.get('normalization') is not None:
            data_module.load_state_dict({'normalization': saved_checkpoint['normalization']})
        model = load_checkpoint(initial_path, model, torch.device('cpu'), checkpoint=saved_checkpoint)
        if saved_checkpoint.get('normalization') is None:
            prepare_evaluation_normalization(data_module, model, 'lens_craft')
        del saved_checkpoint
        logger.info('Initialized weights for a new run from %s; optimizer and epoch start fresh', initial_checkpoint)

    optimizer = instantiate(cfg.training.optimizer)
    lr_scheduler = instantiate(cfg.training.lr_scheduler)

    checkpoint_path = None
    if resume_checkpoint not in empty:
        checkpoint_path = hydra.utils.to_absolute_path(os.path.expanduser(str(resume_checkpoint)))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        saved_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved_checkpoint.get('normalization') is not None:
            data_module.load_state_dict({'normalization': saved_checkpoint['normalization']})
        else:
            prepare_evaluation_normalization(data_module, model, 'lens_craft')
        del saved_checkpoint
        logger.info("Resuming training from checkpoint: %s", checkpoint_path)

    # Keep current training options in both cases; fit restores all checkpoint
    # state after setup, including any torch.compile wrapping of the model.
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

    validation_loss = trainer.callback_metrics.get(monitor)
    if validation_loss is None:
        validation_loss = trainer.callback_metrics.get(f'{monitor}_epoch')
    if validation_loss is None:
        validation_results = trainer.validate(
            lightning_model, datamodule=data_module, verbose=False
        )
        if validation_results:
            validation_loss = validation_results[0].get(monitor)
            if validation_loss is None:
                validation_loss = validation_results[0].get(f'{monitor}_epoch')

    validation_loss = finite_validation_loss(validation_loss)

    logger.info("Sweep objective %s: %.8f", monitor, validation_loss)
    return validation_loss


if __name__ == "__main__":
    main()
