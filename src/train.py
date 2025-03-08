import os
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate, get_class
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
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size
    )

    model = instantiate(cfg.training.model)

    optimizer = instantiate(cfg.training.optimizer)
    lr_scheduler = instantiate(cfg.training.lr_scheduler)

    LightningModuleClass = get_class(cfg.training._target_)

    if hasattr(cfg, 'resume_checkpoint') and cfg.resume_checkpoint != "None":
        checkpoint_path = cfg.resume_checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        lightning_model = LightningModuleClass.load_from_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            noise=cfg.training.noise,
            mask=cfg.training.mask,
            teacher_forcing_schedule=cfg.training.teacher_forcing_schedule,
            compile_mode=cfg.compile.mode,
            compile_enabled=cfg.compile.enabled,
            dataset_mode=data_module.dataset_mode,
        )
        print("Checkpoint loaded successfully")
    else:
        lightning_model = instantiate(
            cfg.training,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            compile_mode=cfg.compile.mode,
            compile_enabled=cfg.compile.enabled,
            dataset_mode=data_module.dataset_mode
        )

    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    is_sweep_run = os.environ.get('IS_SWEEP_RUN', 'false').lower() == 'true'
    
    if is_sweep_run:
        sweep_max_epochs = int(os.environ.get('SWEEP_MAX_EPOCHS', '20'))
        
        trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
        if isinstance(trainer_config, dict):
            trainer_config.pop('_target_', None)
            trainer_config.pop('_partial_', None)
            trainer_config['max_epochs'] = sweep_max_epochs
            trainer = L.Trainer(**trainer_config, callbacks=callbacks)
        else:
            trainer = instantiate(cfg.trainer, callbacks=callbacks, max_epochs=sweep_max_epochs)
    else:
        trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(lightning_model, datamodule=data_module)

    val_loss = None
    for callback in callbacks:
        if hasattr(callback, 'best_model_score'):
            val_loss = callback.best_model_score.item()
            break
    
    if val_loss is None:
        test_results = trainer.test(lightning_model, datamodule=data_module)
        if len(test_results) > 0:
            val_loss = test_results[0].get('test_loss', float('inf'))
    
    return val_loss


if __name__ == "__main__":
    main()
