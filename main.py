import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.camera_trajectory_model import MultiTaskAutoencoder
from data.simulation.dataset import SimulationDataset, batch_collate
from models.clip_embeddings import get_latent_dim, initialize_clip_embeddings
from training.train import train_model
from data.simulation.constants import movement_descriptions, easing_descriptions, angle_descriptions, shot_descriptions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera Trajectory Model Training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14",
                        help="CLIP model name")
    parser.add_argument("--data", type=str, default="data/simulation/random_simulation_dataset.json",
                        help="Path to the dataset JSON file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--dim_feedforward", type=int, default=2048,
                        help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument("--final_mask_ratio", type=float, default=0.7,
                        help="Final mask ratio")
    parser.add_argument("--final_noise_std", type=float, default=0.05,
                        help="Final noise standard deviation")
    parser.add_argument("--init_teacher_forcing_ratio", type=float, default=0.8,
                        help="Initial teacher forcing ratio")
    parser.add_argument("--initial_mask_ratio", type=float, default=0.3,
                        help="Initial mask ratio")
    parser.add_argument("--initial_noise_std", type=float, default=0.2,
                        help="Initial noise standard deviation")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of heads in multi-head attention")
    parser.add_argument("--num_decoder_layers", type=int, default=3,
                        help="Number of decoder layers")
    parser.add_argument("--num_encoder_layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    parser.add_argument("--output", type=str, default="camera_trajectory_model.pth",
                        help="Path to save the trained model")
    parser.add_argument("--patience", type=int, default=30,
                        help="Patience for early stopping")
    parser.add_argument("--seq_length", type=int, default=30,
                        help="Sequence length")
    parser.add_argument("--val_size", type=float, default=0.3,
                        help="Validation size for train-validation split")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    clip_embeddings = {
        'movement': initialize_clip_embeddings(movement_descriptions, args.clip_model_name),
        'easing': initialize_clip_embeddings(easing_descriptions, args.clip_model_name),
        'angle': initialize_clip_embeddings(angle_descriptions, args.clip_model_name),
        'shot': initialize_clip_embeddings(shot_descriptions, args.clip_model_name)
    }

    dataset = SimulationDataset(args.data, clip_embeddings)
    train_dataset, val_dataset = train_test_split(
        dataset, test_size=args.val_size, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=batch_collate, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=batch_collate, num_workers=args.num_workers)

    latent_dim = get_latent_dim(args.clip_model_name)
    model = MultiTaskAutoencoder(
        input_dim=7,
        subject_dim=9,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout_rate=args.dropout,
        seq_length=args.seq_length,
        latent_dim=latent_dim
    ).to(device)

    config = {
        'num_epochs': args.epochs,
        'patience': args.patience,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'initial_noise_std': args.initial_noise_std,
        'final_noise_std': args.final_noise_std,
        'initial_mask_ratio': args.initial_mask_ratio,
        'final_mask_ratio': args.final_mask_ratio,
        'init_teacher_forcing_ratio': args.init_teacher_forcing_ratio,
        'device': device
    }

    train_model(model, train_dataloader, val_dataloader, config)

    torch.save(model.state_dict(), args.output)

    print(f"Training completed and model saved to {args.output}")


if __name__ == "__main__":
    main()
