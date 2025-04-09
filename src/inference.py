import torch
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from models.camera_trajectory_model import MultiTaskAutoencoder
from data.datamodule import CameraTrajectoryDataModule
from inference.checkpoint_utils import load_checkpoint
from data.simulation.dataset import collate_fn
from data.et.dataset import collate_fn as et_collate_fn
from data.ccdm.dataset import collate_fn as ccdm_collate_fn
from inference.export_et import export_et_trajectories
from inference.export_simulation import export_simulation, prepare_output_directory
from metrics.callback import MetricCallback
from dotenv import load_dotenv
load_dotenv()


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    
    device = torch.device(cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")
    
    model : MultiTaskAutoencoder = instantiate(cfg.training.model)
    model = load_checkpoint(cfg.checkpoint_path, model, device)
    model.to(device)
    model.eval()
    
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.config,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size
    )
    data_module.setup()
    dataset = data_module.test_dataset.dataset

    # Initialize metrics callback
    metric_callback = MetricCallback(num_cams=1, device=device)
    metrics = {}
    simulations = []

    if 'CCDMDataset' in cfg.data.dataset.config['_target_']:
        num_samples = min(7, len(dataset)) if cfg.sample_id is None else 1
        sample_indices = [cfg.sample_id] if cfg.sample_id is not None else range(num_samples)
        
        for idx in sample_indices:
            batch = ccdm_collate_fn([dataset[idx]])
            
            # Move all tensors to device
            subject_traj_loc_rot = batch['subject_trajectory_loc_rot'].to(device)
            subject_vol = batch['subject_volume'].to(device)
            camera_trajectory = batch['camera_trajectory'].to(device)
            padding_mask = batch.get('padding_mask', None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            
            with torch.no_grad():
                # Get reference embedding - encode the ground truth camera trajectory
                subject_embedding_loc_rot = model.subject_projection_loc_rot(subject_traj_loc_rot)
                subject_embedding_vol = model.subject_projection_vol(subject_vol)
                subject_embedding = torch.cat([subject_embedding_loc_rot, subject_embedding_vol], 1)
                ref_embedding = model.encoder(camera_trajectory, subject_embedding)[:model.memory_tokens_count]
                
                rec = model.generate_camera_trajectory(
                    subject_trajectory_loc_rot=subject_traj_loc_rot,
                    subject_volume=subject_vol,
                    camera_trajectory=camera_trajectory,
                    padding_mask=padding_mask,
                )
                
                # Update metrics
                metric_callback.update_clatr_metrics("rec", rec['embeddings'][:model.memory_tokens_count].reshape(1, -1), ref_embedding.reshape(1, -1), batch.get('cinematography_prompt', None).reshape(1, -1) if batch.get('cinematography_prompt', None) is not None else None)
            
            original_item = dataset[idx]
            caption = original_item.get('raw_text', None) if isinstance(original_item, dict) else None
            
            sim_data = {
                "subject": batch['subject_trajectory'][0] if 'subject_trajectory' in batch else None,
                "subject_loc_rot": subject_traj_loc_rot[0].cpu(),
                "subject_vol": subject_vol[0].cpu(),
                "camera": batch['camera_trajectory'][0],
                "rec": rec['reconstructed'],
                "padding_mask": batch.get('padding_mask', None),
            }
            
            simulations.append(sim_data)
            
        # Compute final metrics
        metrics.update(metric_callback.compute_clatr_metrics("rec"))
        
        output_dir = prepare_output_directory(cfg.output_dir)
        export_simulation(simulations, output_dir, metrics)
    
    elif 'ETDataset' in cfg.data.dataset.config['_target_']:
        num_samples = min(7, len(dataset)) if cfg.sample_id is None else 1
        sample_indices = [cfg.sample_id] if cfg.sample_id is not None else range(num_samples)
        
        for idx in sample_indices:
            batch = et_collate_fn([dataset[idx]])
            
            subject_traj = batch['subject_trajectory'].to(device)
            subject_loc_rot = torch.cat([subject_traj[:, :, :3], subject_traj[:, :, 6:]], dim=2)
            subject_vol = subject_traj[:, 0:1, 3:6].permute(0, 2, 1)
            
            with torch.no_grad():
                # Get reference embedding - encode the ground truth camera trajectory
                subject_embedding_loc_rot = model.subject_projection_loc_rot(subject_loc_rot)
                subject_embedding_vol = model.subject_projection_vol(subject_vol)
                subject_embedding = torch.cat([subject_embedding_loc_rot, subject_embedding_vol], 1)
                ref_embedding = model.encoder(batch['camera_trajectory'], subject_embedding)[:model.memory_tokens_count]
                
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
                
                # Update metrics
                metric_callback.update_clatr_metrics("rec", rec['embeddings'][:model.memory_tokens_count].reshape(1, -1), ref_embedding.reshape(1, -1), batch.get('caption_feat', None).reshape(1, -1) if batch.get('caption_feat', None) is not None else None)
                metric_callback.update_clatr_metrics("prompt_gen", prompt_gen['embeddings'][:model.memory_tokens_count].reshape(1, -1), ref_embedding.reshape(1, -1), batch.get('caption_feat', None).reshape(1, -1) if batch.get('caption_feat', None) is not None else None)
                if hybrid_gen is not None:
                    metric_callback.update_clatr_metrics("hybrid_gen", hybrid_gen['embeddings'][:model.memory_tokens_count].reshape(1, -1), ref_embedding.reshape(1, -1), batch.get('caption_feat', None).reshape(1, -1) if batch.get('caption_feat', None) is not None else None)
            
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
                "rec": rec['reconstructed'],
                "padding_mask": batch.get('padding_mask', None),
            }
            
            if prompt_gen is not None:
                sim_data["prompt_gen"] = prompt_gen['reconstructed']
            
            if hybrid_gen is not None:
                sim_data["hybrid_gen"] = hybrid_gen['reconstructed']
                
            if caption is not None:
                sim_data["caption"] = caption
                
            simulations.append(sim_data)
            
        # Compute final metrics
        for run_type in ["rec", "prompt_gen", "hybrid_gen"]:
            metrics.update(metric_callback.compute_clatr_metrics(run_type))

        output_dir = prepare_output_directory(cfg.output_dir)
        export_et_trajectories(simulations, output_dir, metrics)
        
    else:
        for idx in range(30):
            batch = collate_fn([dataset[idx]])
            
            # Move all tensors to device
            subject_trajectory_loc_rot = batch['subject_trajectory_loc_rot'].to(device)
            subject_volume = batch['subject_volume'].to(device)
            camera_trajectory = batch['camera_trajectory'].to(device)
            cinematography_prompt = batch.get('cinematography_prompt', None)
            if cinematography_prompt is not None:
                cinematography_prompt = cinematography_prompt.to(device)
            
            with torch.no_grad():
                # Get reference embedding - encode the ground truth camera trajectory
                subject_embedding_loc_rot = model.subject_projection_loc_rot(subject_trajectory_loc_rot)
                subject_embedding_vol = model.subject_projection_vol(subject_volume)
                subject_embedding = torch.cat([subject_embedding_loc_rot, subject_embedding_vol], 1)
                ref_embedding = model.encoder(camera_trajectory, subject_embedding)[:model.memory_tokens_count]
                
                rec = model.generate_camera_trajectory(
                    subject_trajectory_loc_rot=subject_trajectory_loc_rot,
                    subject_volume=subject_volume,
                    camera_trajectory=camera_trajectory
                )
                
                prompt_gen = model.generate_camera_trajectory(
                    subject_trajectory_loc_rot=subject_trajectory_loc_rot,
                    subject_volume=subject_volume,
                    camera_trajectory=camera_trajectory,
                    caption_embedding=cinematography_prompt,
                    teacher_forcing_ratio=1.0
                )
                
                hybrid_gen = model.generate_camera_trajectory(
                    subject_trajectory_loc_rot=subject_trajectory_loc_rot,
                    subject_volume=subject_volume,
                    camera_trajectory=camera_trajectory,
                    caption_embedding=cinematography_prompt,
                    teacher_forcing_ratio=0.4
                )
                
                # Update metrics
                metric_callback.update_clatr_metrics("rec", rec['embeddings'][:model.memory_tokens_count].reshape(1, -1), ref_embedding.reshape(1, -1), cinematography_prompt.reshape(1, -1) if cinematography_prompt is not None else None)
                metric_callback.update_clatr_metrics("prompt_gen", prompt_gen['embeddings'][:model.memory_tokens_count].reshape(1, -1), ref_embedding.reshape(1, -1), cinematography_prompt.reshape(1, -1) if cinematography_prompt is not None else None)
                metric_callback.update_clatr_metrics("hybrid_gen", hybrid_gen['embeddings'][:model.memory_tokens_count].reshape(1, -1), ref_embedding.reshape(1, -1), cinematography_prompt.reshape(1, -1) if cinematography_prompt is not None else None)
            
            simulations.append({
                "subject_loc_rot": subject_trajectory_loc_rot[0].cpu(),
                "subject_vol": subject_volume[0].cpu(),
                "camera": batch['camera_trajectory'][0],
                "rec": rec['reconstructed'],
                "prompt_gen": prompt_gen['reconstructed'],
                "hybrid_gen": hybrid_gen['reconstructed'],
            })
            
        # Compute final metrics
        for run_type in ["rec", "prompt_gen", "hybrid_gen"]:
            metrics.update(metric_callback.compute_clatr_metrics(run_type))
            
        output_dir = prepare_output_directory(cfg.output_dir)
        print (metrics)
        export_simulation(simulations, output_dir, metrics)

if __name__ == "__main__":
    main()
