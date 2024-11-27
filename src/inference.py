import shutil
from pathlib import Path

import os
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from inference.inferencer import ModelInference


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    
    inference = ModelInference(cfg=cfg)
    
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.module,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size
    )
    
    data_module.setup()
    dataset = data_module.train_dataset.dataset
    
    sample_indices = []
    if cfg.sample_id:
        for idx in range(len(dataset)):
            if dataset.original_dataset.root_filenames[idx] == cfg.sample_id:
                sample_indices.append(idx)
                break
        if not sample_indices:
            raise ValueError(f"Sample ID {cfg.sample_id} not found in dataset")
    else:
        sample_indices = range(10)

    dataset_dir = Path(str(cfg.data.dataset.module.dataset_dir))
    
    for idx in sample_indices:
        sample = dataset[idx]
        sample_id = dataset.original_dataset.root_filenames[idx]
        process_sample(
            cfg=cfg,
            inference=inference,
            sample_data={
                'subject_trajectory': sample['subject_trajectory'],
                'camera_trajectory': sample['camera_trajectory'],
                'padding_mask': sample.get('padding_mask', None),
                'caption_feat': sample.get('caption_feat', None)
            },
            sample_id=sample_id,
            dataset_dir=dataset_dir
        )


def process_sample(
    cfg: DictConfig,
    inference: ModelInference,
    sample_data: dict,
    sample_id: str,
    dataset_dir: Path
):
    subject_trajectory = sample_data['subject_trajectory'].unsqueeze(0)
    camera_trajectory = sample_data['camera_trajectory'].unsqueeze(0).transpose(1, 2)
    padding_mask = sample_data['padding_mask']
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(0)

    sample_output_dir = os.path.join(cfg.output_dir, sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    output_dir = Path(sample_output_dir)
    
    shutil.copy2(dataset_dir / 'char' / f"{sample_id}.npy", output_dir / "char.npy")
    shutil.copy2(dataset_dir / 'traj' / f"{sample_id}.txt", output_dir / "traj.txt")
    
    inference.reconstruct_trajectory(
        camera_trajectory,
        subject_trajectory,
        padding_mask,
        output_path=sample_output_dir
    )
    
    if 'caption_feat' in sample_data and sample_data['caption_feat'] is not None:
        inference.generate_from_caption_feat(
            sample_data['caption_feat'].unsqueeze(0),
            subject_trajectory,
            padding_mask,
            output_path=sample_output_dir
        )


if __name__ == "__main__":
    main()