import shutil
from pathlib import Path

import os
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from inference.inferencer import ModelInference


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    
    inference = ModelInference(cfg=cfg)
    
    dataset = hydra.utils.instantiate(cfg.data.dataset.module)

    sample_indices = [dataset.original_dataset.root_filenames.index(cfg.sample_id)] if cfg.sample_id else range(10)

    for idx in sample_indices:
        process_sample(cfg, inference, dataset, idx)


def process_sample(cfg: DictConfig, inference: ModelInference, dataset, idx: int):
    sample = dataset[idx]
    sample_id = dataset.original_dataset.root_filenames[idx]

    subject_trajectory = sample['subject_trajectory'].unsqueeze(0)
    camera_trajectory = sample['camera_trajectory'].unsqueeze(0).transpose(1, 2)

    sample_output_dir = os.path.join(cfg.output_dir, sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    dataset_dir = Path(str(cfg.data.dataset.module.dataset_dir))
    output_dir = Path(sample_output_dir)
    
    shutil.copy2(dataset_dir / 'char' / f"{sample_id}.npy", output_dir / "char.npy")
    shutil.copy2(dataset_dir / 'traj' / f"{sample_id}.txt", output_dir / "traj.txt")
    
    inference.reconstruct_trajectory(
        camera_trajectory,
        subject_trajectory,
        output_path=sample_output_dir
    )
    
    if 'caption_feat' in sample:
        inference.generate_from_caption_feat(
            sample['caption_feat'].unsqueeze(0),
            subject_trajectory,
            output_path=sample_output_dir
        )


if __name__ == "__main__":
    main()