import os
import sys
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
import lightning as L
import torch
from data.datamodule import CameraTrajectoryDataModule
from data.multi_dataset_module import MultiDatasetModule
from testing.metrics.callback import MetricCallback
from testing.process import test_batch
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()


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

    if hasattr(cfg, 'resume_checkpoint') and cfg.resume_checkpoint != "None":
        checkpoint_path = cfg.resume_checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
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
            dataset_mode=getattr(data_module, 'dataset_mode', 'simulation'),
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
    trainer.fit(lightning_model, datamodule=data_module)
    if trainer.global_rank != 0:
        sys.exit(0)
    
    if use_multi_dataset:
        dataset_type = "simulation" if getattr(cfg.data, 'sim_ratio', 0) > 0 else "ccdm"
    else:
        target = cfg.data.dataset.config["_target_"]
        dataset_type = (
            "ccdm" if "CCDMDataset" in target else "et" if "ETDataset" in target else "simulation"
        )
    
    prdc_sum = 0
    if dataset_type == "simulation":
        model = lightning_model.model
        model.eval()
        
        device = model.device
        model.to(device)

        metric_callback = MetricCallback(num_cams=1, device=device)
        
        val_dataloader = data_module.val_dataloader()
        
        metric_items = ["prompt_generation", "hybrid_generation"]
        
        with torch.no_grad():
            for batch in val_dataloader:
                test_batch(model, model, batch, metric_callback, device, metric_items, dataset_type='simulation', model_type='lens_craft')
        
        for metric_item in metric_items:
            metrics = metric_callback.compute_clatr_metrics(metric_item)
            logger.info(f"{metric_item} Metrics: {metrics}")
            # Sum PRDC metrics
            type_prdc_sum = (
                max(0, min(1, metrics[f"{metric_item}/precision"])) + 
                max(0, min(1, metrics[f"{metric_item}/recall"])) + 
                max(0, min(1, metrics[f"{metric_item}/density"])) + 
                max(0, min(1, metrics[f"{metric_item}/coverage"]))
            )
            prdc_sum += type_prdc_sum
            logger.info(f"{metric_item} PRDC sum: {type_prdc_sum}")
        
        logger.info(f"Total PRDC sum: {prdc_sum}")

    return -float(prdc_sum)


if __name__ == "__main__":
    main()
