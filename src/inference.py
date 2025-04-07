import torch
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from inference.checkpoint_utils import load_checkpoint
from data.simulation.dataset import collate_fn
from data.et.dataset import collate_fn as et_collate_fn
from data.ccdm.dataset import collate_fn as ccdm_collate_fn
from inference.export_et import export_et_trajectories
from inference.export_simulation import export_simulation, prepare_output_directory


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    
    device = torch.device(cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")
    
    model = instantiate(cfg.training.model)
    model = load_checkpoint(cfg.checkpoint_path, model, device)
    model.eval()
    
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.module,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size
    )
    data_module.setup()
    dataset = data_module.test_dataset.dataset

    if 'CCDMDataset' in cfg.data.dataset.module['_target_']:
        simulations = []
        num_samples = min(7, len(dataset)) if cfg.sample_id is None else 1
        sample_indices = [cfg.sample_id] if cfg.sample_id is not None else range(num_samples)
        
        for idx in sample_indices:
            batch = ccdm_collate_fn([dataset[idx]])
            
            subject_traj_loc_rot = batch['subject_trajectory_loc_rot'].to(device)
            subject_vol = batch['subject_volume'].to(device)
            
            with torch.no_grad():
                rec = model.generate_camera_trajectory(
                    subject_trajectory_loc_rot=subject_traj_loc_rot,
                    subject_volume=subject_vol,
                    camera_trajectory=batch['camera_trajectory'].to(device),
                    padding_mask=batch.get('padding_mask', None),
                )
            
            original_item = dataset[idx]
            caption = original_item.get('raw_text', None) if isinstance(original_item, dict) else None
            
            sim_data = {
                "subject": batch['subject_trajectory'][0] if 'subject_trajectory' in batch else None,
                "subject_loc_rot": subject_traj_loc_rot[0].cpu(),
                "subject_vol": subject_vol[0].cpu(),
                "camera": batch['camera_trajectory'][0],
                "rec": rec,
                "padding_mask": batch.get('padding_mask', None),
            }
            
            simulations.append(sim_data)
            
        output_dir = prepare_output_directory(cfg.output_dir)
        export_simulation(simulations, output_dir)
    
    elif 'ETDataset' in cfg.data.dataset.module['_target_']:
        simulations = []
        num_samples = min(7, len(dataset)) if cfg.sample_id is None else 1
        sample_indices = [cfg.sample_id] if cfg.sample_id is not None else range(num_samples)
        
        for idx in sample_indices:
            batch = et_collate_fn([dataset[idx]])
            
            subject_traj = batch['subject_trajectory'].to(device)
            subject_loc_rot = torch.cat([subject_traj[:, :, :3], subject_traj[:, :, 6:]], dim=2)
            subject_vol = subject_traj[:, 0:1, 3:6].permute(0, 2, 1)
            
            with torch.no_grad():
                rec = model.generate_camera_trajectory(
                    subject_trajectory_loc_rot=subject_loc_rot,
                    subject_volume=subject_vol,
                    camera_trajectory=batch['camera_trajectory'].to(device),
                    padding_mask=batch.get('padding_mask', None),
                )
                
                prompt_gen = model.generate_camera_trajectory(
                    subject_trajectory_loc_rot=subject_loc_rot,
                    subject_volume=subject_vol,
                    caption_embedding=batch['caption_feat'].to(device).unsqueeze(0),
                    padding_mask=batch.get('padding_mask', None),
                    teacher_forcing_ratio=1.0
                )
                
                hybrid_gen = None
                if 'camera_trajectory' in batch and 'caption_feat' in batch:
                    hybrid_gen = model.generate_camera_trajectory(
                        subject_trajectory_loc_rot=subject_loc_rot,
                        subject_volume=subject_vol,
                        camera_trajectory=batch['camera_trajectory'].to(device),
                        caption_embedding=batch['caption_feat'].to(device).unsqueeze(0),
                        padding_mask=batch.get('padding_mask', None),
                        teacher_forcing_ratio=0.4
                    )
            
            caption = None
            original_item = dataset[idx]
            if isinstance(original_item, dict) and 'caption_raw' in original_item and 'caption' in original_item['caption_raw']:
                caption = original_item['caption_raw']['caption']
            
            subject_trajectory = batch['subject_trajectory'] if 'subject_trajectory' in batch else None
            
            sim_data = {
                "subject": subject_trajectory[0] if subject_trajectory is not None else None,
                "subject_loc_rot": subject_loc_rot[0].cpu(),
                "subject_vol": subject_vol[0].cpu(),
                "camera": batch['camera_trajectory'][0],
                "rec": rec,
                "padding_mask": batch.get('padding_mask', None),
            }
            
            if prompt_gen is not None:
                sim_data["prompt_gen"] = prompt_gen
            
            if hybrid_gen is not None:
                sim_data["hybrid_gen"] = hybrid_gen
                
            if caption is not None:
                sim_data["caption"] = caption
                
            simulations.append(sim_data)
            
        output_dir = prepare_output_directory(cfg.output_dir)
        export_et_trajectories(simulations, output_dir)
        
    else:
        simulations = []
        for idx in range(7):
            batch = collate_fn([dataset[idx]])
            
            subject_trajectory_loc_rot = batch['subject_trajectory_loc_rot'].to(device)
            subject_volume = batch['subject_volume'].to(device)
            
            rec = model.generate_camera_trajectory(
                subject_trajectory_loc_rot=subject_trajectory_loc_rot,
                subject_volume=subject_volume,
                camera_trajectory=batch['camera_trajectory'].to(device)
            )
            
            prompt_gen = model.generate_camera_trajectory(
                subject_trajectory_loc_rot=subject_trajectory_loc_rot,
                subject_volume=subject_volume,
                camera_trajectory=batch['camera_trajectory'].to(device),
                caption_embedding=batch['cinematography_prompt'].to(device),
                teacher_forcing_ratio=1.0
            )
            
            hybrid_gen = model.generate_camera_trajectory(
                subject_trajectory_loc_rot=subject_trajectory_loc_rot,
                subject_volume=subject_volume,
                camera_trajectory=batch['camera_trajectory'].to(device),
                caption_embedding=batch['cinematography_prompt'].to(device),
                teacher_forcing_ratio=0.4
            )
            
            simulations.append({
                "subject_loc_rot": subject_trajectory_loc_rot[0].cpu(),
                "subject_vol": subject_volume[0].cpu(),
                "camera": batch['camera_trajectory'][0],
                "rec": rec,
                "prompt_gen": prompt_gen,
                "hybrid_gen": hybrid_gen,
            })
            
        output_dir = prepare_output_directory(cfg.output_dir)
        export_simulation(simulations, output_dir)

if __name__ == "__main__":
    main()
