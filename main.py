import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.camera_trajectory_model import MultiTaskAutoencoder
from data.simulation.dataset import SimulationDataset, batch_collate
from models.clip_embeddings import get_latent_dim, initialize_clip_embeddings
from training.train import train_model
from data.simulation.constants import movement_descriptions, easing_descriptions, angle_descriptions, shot_descriptions


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.training.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.training.device)

    print(f"Using device: {device}")

    clip_embeddings = {
        'movement': initialize_clip_embeddings(movement_descriptions, cfg.model.clip_model_name),
        'easing': initialize_clip_embeddings(easing_descriptions, cfg.model.clip_model_name),
        'angle': initialize_clip_embeddings(angle_descriptions, cfg.model.clip_model_name),
        'shot': initialize_clip_embeddings(shot_descriptions, cfg.model.clip_model_name)
    }

    dataset = SimulationDataset(cfg.data.data_path, clip_embeddings)
    train_dataset, val_dataset = train_test_split(
        dataset, test_size=cfg.data.val_size, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                  collate_fn=batch_collate, num_workers=cfg.data.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                                collate_fn=batch_collate, num_workers=cfg.data.num_workers)

    latent_dim = get_latent_dim(cfg.model.clip_model_name)
    model = MultiTaskAutoencoder(
        input_dim=7,
        subject_dim=9,
        nhead=cfg.model.nhead,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout_rate=cfg.model.dropout,
        seq_length=cfg.model.seq_length,
        latent_dim=latent_dim
    ).to(device)

    train_config = {
        'num_epochs': cfg.training.epochs,
        'patience': cfg.training.patience,
        'learning_rate': cfg.training.lr,
        'weight_decay': cfg.training.weight_decay,
        'initial_noise_std': cfg.training.initial_noise_std,
        'final_noise_std': cfg.training.final_noise_std,
        'initial_mask_ratio': cfg.training.initial_mask_ratio,
        'final_mask_ratio': cfg.training.final_mask_ratio,
        'init_teacher_forcing_ratio': cfg.training.init_teacher_forcing_ratio,
        'device': device
    }

    train_model(model, train_dataloader, val_dataloader, train_config)

    torch.save(model.state_dict(), cfg.output)

    print(f"Training completed and model saved to {cfg.output}")


if __name__ == "__main__":
    main()
