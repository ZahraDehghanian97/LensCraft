import torch
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from data.datamodule import CameraTrajectoryDataModule
from inference.inferencer import ModelInference
from inference.trajectory_processor import TrajectoryData, TrajectoryProcessor

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    
    inference = ModelInference(cfg=cfg)
    data_module = setup_data_module(cfg)
    dataset = data_module.test_dataset.dataset
    
    sample_indices = get_sample_indices(cfg, dataset)
    
    processor = TrajectoryProcessor(
        output_dir=cfg.output_dir,
        dataset_dir=Path(str(cfg.data.dataset.module.dataset_dir)) if 'ETDataset' in cfg.data.dataset.module['_target_'] else None
    )
    
    process_samples(cfg, inference, dataset, sample_indices, processor)

def setup_data_module(cfg: DictConfig) -> CameraTrajectoryDataModule:
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.module,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size
    )
    data_module.setup()
    return data_module

def get_sample_indices(cfg: DictConfig, dataset) -> list:
    if cfg.sample_id:
        for idx in range(len(dataset)):
            if dataset.original_dataset.root_filenames[idx] == cfg.sample_id:
                return [idx]
        raise ValueError(f"Sample ID {cfg.sample_id} not found in dataset")
    return list(range(10))

def process_samples(
    cfg: DictConfig,
    inference: ModelInference,
    dataset,
    sample_indices: list,
    processor: TrajectoryProcessor
):
    if 'ETDataset' in cfg.data.dataset.module['_target_']:
        for idx in sample_indices:
            sample = dataset[idx]
            data = TrajectoryData(
                subject_trajectory=sample['subject_trajectory'].unsqueeze(0),
                camera_trajectory=sample['camera_trajectory'].unsqueeze(0),
                padding_mask=sample.get('padding_mask', None),
                caption_feat=sample.get('caption_feat', None).unsqueeze(0)
            )
            sample_id = dataset.original_dataset.root_filenames[idx]
            output_dir = processor.prepare_output_directory(sample_id)
            processor.copy_dataset_files(sample_id, Path(output_dir))
            inference.reconstruct_trajectory(data, output_dir)
            
            if data.caption_feat is not None:
                inference.generate_from_caption_feat(data, output_dir)
    else:
        simulations = []
        for idx in sample_indices:
            sample = dataset[idx]
            data = TrajectoryData(
                subject_trajectory=sample['subject_trajectory'].unsqueeze(0),
                camera_trajectory=sample['camera_trajectory'].unsqueeze(0),
                padding_mask=sample.get('padding_mask', None),
                caption_feat=torch.stack([
                    sample['movement_clip'],
                    sample['easing_clip'],
                    sample['angle_clip'],
                    sample['shot_clip']
                ]).unsqueeze(1)
            )
            data.teacher_forcing_ratio = 0.0
            rec = inference.reconstruct_trajectory(data)
            data.teacher_forcing_ratio = 0.4
            full_key_gen = inference.reconstruct_trajectory(data)
            data.teacher_forcing_ratio = 1.0
            prompt_gen = inference.reconstruct_trajectory(data)
            simulations.append({
                "subject": dataset[idx]['subject_trajectory'],
                "camera": dataset[idx]['camera_trajectory'],
                "rec": rec,
                "full_key_gen": full_key_gen,
                "prompt_gen": prompt_gen,
                "instruction": dataset[idx]['instruction'],
            })
        output_dir = processor.prepare_output_directory()
        processor.save_simulation_format(simulations, output_dir)


if __name__ == "__main__":
    main()
