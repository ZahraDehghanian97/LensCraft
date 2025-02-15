import torch
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from data.datamodule import CameraTrajectoryDataModule
from inference.trajectory_processor import TrajectoryData, TrajectoryProcessor
from inference.checkpoint_utils import load_checkpoint
from data.simulation.dataset import collate_fn
from models.camera_trajectory_model import MultiTaskAutoencoder


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
        
    device = torch.device(cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")
    
    model = instantiate(cfg.training.model)
    model = load_checkpoint(cfg.checkpoint_path, model, device)
    model.eval()
    
    data_module = setup_data_module(cfg)
    dataset = data_module.train_dataset.dataset
    
    sample_indices = get_sample_indices(cfg, dataset)
    
    processor = TrajectoryProcessor(
        output_dir=cfg.output_dir,
        dataset_dir=Path(str(cfg.data.dataset.module.dataset_dir)) if 'ETDataset' in cfg.data.dataset.module['_target_'] else None
    )
    
    process_samples(cfg, model, dataset, sample_indices, processor)

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
    model: MultiTaskAutoencoder,
    dataset,
    sample_indices: list,
    processor: TrajectoryProcessor
):
    if 'ETDataset' in cfg.data.dataset.module['_target_']:
        for idx in sample_indices:
            sample_id = dataset.original_dataset.root_filenames[idx]
            output_dir = processor.prepare_output_directory(sample_id)
            processor.copy_dataset_files(sample_id, Path(output_dir))
            model.inference(
                subject_trajectory=dataset[idx]['subject_trajectory'].unsqueeze(0),
                camera_trajectory=dataset[idx]['camera_trajectory'].unsqueeze(0),
                padding_mask=dataset[idx].get('padding_mask', None),
                caption_embedding=dataset[idx].get('caption_feat', None).unsqueeze(0)
            )
    else:
        simulations = []
        for idx in range(7):
            batch = collate_fn([dataset[idx]])
            
            rec = model.inference(
                subject_trajectory=batch['subject_trajectory'],
                camera_trajectory=batch['camera_trajectory']
            )
            prompt_gen = model.inference(
                subject_trajectory=batch['subject_trajectory'],
                camera_trajectory=batch['camera_trajectory'],
                caption_embedding=batch['cinematography_prompt'],
                teacher_forcing_ratio=1.0
            )
            hybrid_gen = model.inference(
                subject_trajectory=batch['subject_trajectory'],
                camera_trajectory=batch['camera_trajectory'],
                caption_embedding=batch['cinematography_prompt'],
                teacher_forcing_ratio=0.4
            )
            
            simulations.append({
                "subject": batch['subject_trajectory'][0],
                "camera": batch['camera_trajectory'][0],
                "rec": rec,
                "prompt_gen": prompt_gen,
                "hybrid_gen": hybrid_gen,
            })
            
        output_dir = processor.prepare_output_directory()
        processor.save_simulation_format(simulations, output_dir)


if __name__ == "__main__":
    main()
