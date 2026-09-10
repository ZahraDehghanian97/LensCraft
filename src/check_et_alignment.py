from __future__ import annotations

import itertools
import logging

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from data.convertor.alignment import recenter_rescale_sim
from data.convertor.convertor import convert_to_target
from data.datamodule import CameraTrajectoryDataModule
from data.et.config import STANDARDIZATION_CONFIG_TORCH as ET_CFG

load_dotenv()
logger = logging.getLogger(__name__)

ET_SEQ_LENGTH = 300


def _axis_candidates():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            yield perm, signs


def _candidate_name(perm, signs):
    axes = "xyz"
    return ", ".join(
        f"{axes[i]}'={'-' if s < 0 else ''}{axes[p]}"
        for i, (p, s) in enumerate(zip(perm, signs))
    )


def check_batch(batch) -> None:
    cam, subj, vol, _origin, _scale = recenter_rescale_sim(
        batch["camera_trajectory"],
        batch["subject_trajectory"],
        batch["subject_volume"],
    )

    subj0 = subj[:, 0, :3]
    print(f"\n1. subject frame 0 after recentering, abs max: {subj0.abs().max():.2e} (expect ~0)")

    traj_et, subj_et, _, _ = convert_to_target(
        "simulation", "et", cam, subj, vol, batch["padding_mask"],
        ET_SEQ_LENGTH, need_denormal=False,
    )

    subj_vel_z = subj_et[:, 1:, :].abs()
    print(
        f"2. subject velocity z-scores (frames 1+): mean {subj_vel_z.mean():.2f}, "
        f"p95 {subj_vel_z.reshape(-1).quantile(0.95):.2f} (expect O(1-3))"
    )

    cam_vel_z = traj_et[:, 1:, 6:].abs()
    print(
        f"3. camera velocity z-scores (frames 1+):  mean {cam_vel_z.mean():.2f}, "
        f"p95 {cam_vel_z.reshape(-1).quantile(0.95):.2f} (expect O(1))"
    )

    # Camera frame 0 in the recentered/rescaled scene, raw (un-normalized):
    # z-score it under shift_mean/shift_std for every axis mapping candidate.
    traj_raw, _, _, _ = convert_to_target(
        "simulation", "et", cam, subj, vol, batch["padding_mask"],
        ET_SEQ_LENGTH, need_denormal=False, need_normal=False,
    )
    # Already in E.T.'s native Y-down world after ETConvertor.from_standard;
    # candidates below are additional axis changes, so identity is still correct.
    cam0 = traj_raw[:, 0, 6:]  # [B, 3] absolute first position

    shift_mean = ET_CFG["shift_mean"].to(cam0)
    shift_std = ET_CFG["shift_std"].to(cam0)

    scored = []
    for perm, signs in _axis_candidates():
        mapped = torch.stack(
            [cam0[:, p] * s for p, s in zip(perm, signs)], dim=-1
        )
        z = (mapped - shift_mean) / shift_std
        scored.append((z.abs().mean().item(), perm, signs))
    scored.sort()

    print("\n4. native E.T. camera frame-0 mean |z| under shift stats, per additional axis mapping (best 5):")
    identity_rank = next(
        i for i, (_, p, s) in enumerate(scored)
        if p == (0, 1, 2) and s == (1.0, 1.0, 1.0)
    )
    for score, perm, signs in scored[:5]:
        tag = "  <-- identity (current conversion)" if (perm, signs) == ((0, 1, 2), (1.0, 1.0, 1.0)) else ""
        print(f"   |z|={score:6.2f}   {_candidate_name(perm, signs)}{tag}")
    if identity_rank >= 5:
        score = scored[identity_rank][0]
        print(f"   ... identity mapping ranks #{identity_rank + 1} with |z|={score:.2f} -- axis convention likely mismatched")
    else:
        print(f"   identity mapping ranks #{identity_rank + 1} -- axis convention looks consistent")


@hydra.main(version_base=None, config_path="../config", config_name="test")
def main(cfg: DictConfig) -> None:
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    data_module.setup()
    batch = next(iter(data_module.test_dataloader()))
    check_batch(batch)


if __name__ == "__main__":
    main()
