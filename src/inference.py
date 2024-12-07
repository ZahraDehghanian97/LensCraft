import os
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
    dataset = data_module.train_dataset.dataset
    
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
                caption_feat=sample.get('caption_feat', None)
            )
            process_et_sample(
                inference=inference,
                data=data,
                sample_id=dataset.original_dataset.root_filenames[idx],
                processor=processor
            )
    else:
        data = TrajectoryData(
            subject_trajectory=dataset[0]['subject_trajectory'].unsqueeze(0),
            camera_trajectory=dataset[0]['camera_trajectory'].unsqueeze(0),
            padding_mask=dataset[0].get('padding_mask', None),
            caption_feat=dataset[0].get('caption_feat', None)
        )
        process_simulation_sample(
            inference=inference,
            data=data,
            processor=processor
        )

def process_et_sample(
    inference: ModelInference,
    data: TrajectoryData,
    sample_id: str,
    processor: TrajectoryProcessor
):
    output_dir = processor.prepare_output_directory(sample_id)
    processor.copy_dataset_files(sample_id, Path(output_dir))
    
    data.camera_trajectory = data.camera_trajectory.transpose(1, 2)
    inference.reconstruct_trajectory(data, output_dir)
    
    if data.caption_feat is not None:
        inference.generate_from_caption_feat(data, output_dir)

def process_simulation_sample(
    inference: ModelInference,
    data: TrajectoryData,
    processor: TrajectoryProcessor
):
    output_dir = processor.prepare_output_directory()
    output = inference.reconstruct_trajectory(data, output_dir, is_simulation=True)
    
    output_path = os.path.join(output_dir, 'simulation-out.json')
    processor.save_simulation_format(output, data.subject_trajectory, output_path)

if __name__ == "__main__":
    main()
