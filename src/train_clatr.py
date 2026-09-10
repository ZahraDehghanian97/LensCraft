from __future__ import annotations

import logging
import os
from typing import Optional

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from training.clatr import LightningCLaTr

load_dotenv()
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")


def _infer_traj_feat_dim(cfg: DictConfig) -> int:
    """Pick the trajectory feature dim from the dataset target."""
    if "clatr_model" in cfg and cfg.clatr_model.get("traj_num_feats", None):
        return int(cfg.clatr_model.traj_num_feats)
    target = cfg.data.dataset.config["_target_"]
    if "SimulationDataset" in target:
        return 6  # 3 pos + 3 euler
    if "ETDataset" in target:
        return 9  # rot6d (6) + translation (3)
    if "CCDMDataset" in target:
        return 6
    return 6


@hydra.main(version_base=None, config_path="../config", config_name="clatr/train")
def main(cfg: DictConfig) -> Optional[float]:
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    L.seed_everything(cfg.seed)

    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )

    traj_num_feats = _infer_traj_feat_dim(cfg)
    logger.info("Training CLaTr with traj_num_feats=%d", traj_num_feats)

    clatr_lit = LightningCLaTr(
        traj_num_feats=traj_num_feats,
        text_num_feats=cfg.clatr_model.text_num_feats,
        latent_dim=cfg.clatr_model.latent_dim,
        ff_size=cfg.clatr_model.ff_size,
        num_layers=cfg.clatr_model.num_layers,
        num_heads=cfg.clatr_model.num_heads,
        dropout=cfg.clatr_model.dropout,
        vae=cfg.clatr_model.vae,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        lmd_recons=cfg.loss.lmd_recons,
        lmd_latent=cfg.loss.lmd_latent,
        lmd_kl=cfg.loss.lmd_kl,
        lmd_nce=cfg.loss.lmd_nce,
        temperature=cfg.loss.temperature,
        threshold_selfsim=cfg.loss.threshold_selfsim,
        max_text_tokens=cfg.clatr_model.max_text_tokens,
        clip_model_name=cfg.clip.model_name
        if cfg.clip.model_name.startswith("openai/")
        else f"openai/{cfg.clip.model_name}",
    )

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="clatr-best-epoch={epoch:03d}-val_loss={val/loss:.4f}",
            auto_insert_metric_name=False,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=output_dir,
            every_n_epochs=cfg.checkpoint.every_n_epochs,
            save_top_k=-1,
            filename="clatr-{epoch:03d}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.early_stopping.enable:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=cfg.early_stopping.patience,
            )
        )

    tb_logger = CSVLogger(save_dir=output_dir, name="csv_logs")

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    resume_ckpt = cfg.get("resume_checkpoint", None)
    if resume_ckpt in (None, "None", ""):
        resume_ckpt = None

    trainer.fit(clatr_lit, datamodule=data_module, ckpt_path=resume_ckpt)

    best_path = callbacks[0].best_model_path
    last_path = os.path.join(output_dir, "last.ckpt")
    logger.info("Best CLaTr checkpoint: %s", best_path)
    logger.info("Last CLaTr checkpoint: %s", last_path)
    logger.info(
        "Set CLATR_NATIVE_CHECKPOINT_PATH=%s (or pass it via test config) "
        "to use this checkpoint at evaluation time.",
        best_path or last_path,
    )

    return trainer.callback_metrics.get("val/loss", torch.tensor(0.0)).item()


if __name__ == "__main__":
    main()
