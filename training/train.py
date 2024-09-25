import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from utils.augmentation import apply_mask_and_noise, get_noise_and_mask_values
from training.losses import compute_loss, print_detailed_losses


def train_epoch(model, dataloader, optimizer, noise_std, mask_ratio, teacher_forcing_ratio, device):
    model.train()
    epoch_losses = []

    for batch in tqdm(dataloader, desc="Training"):
        camera_trajectory = batch['camera_trajectory'].to(device)
        subject = batch['subject'].to(device)
        clip_targets = {
            'movement': batch['movement_clip'].to(device),
            'easing': batch['easing_clip'].to(device),
            'camera_angle': batch['angle_clip'].to(device),
            'shot_type': batch['shot_clip'].to(device)
        }

        noisy_trajectory, mask, src_key_padding_mask = apply_mask_and_noise(
            camera_trajectory, mask_ratio, noise_std, device)

        optimizer.zero_grad()

        output = model(noisy_trajectory, subject, src_key_padding_mask,
                       camera_trajectory, teacher_forcing_ratio)

        loss, loss_dict = compute_loss(
            output, camera_trajectory, clip_targets, mask)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss_dict)

    return epoch_losses


def validate(model, dataloader, device):
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            camera_trajectory = batch['camera_trajectory'].to(device)
            subject = batch['subject'].to(device)
            clip_targets = {
                'movement': batch['movement_clip'].to(device),
                'easing': batch['easing_clip'].to(device),
                'camera_angle': batch['angle_clip'].to(device),
                'shot_type': batch['shot_clip'].to(device)
            }

            output = model(camera_trajectory, subject)
            _, loss_dict = compute_loss(
                output, camera_trajectory, clip_targets)
            val_losses.append(loss_dict)

    return val_losses


def train_model(model, train_dataloader, val_dataloader, config):
    device = config['device']
    model = model.to(device)

    optimizer = Adam(model.parameters(),
                     lr=config['learning_rate'],
                     weight_decay=config['weight_decay'])

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(config['num_epochs']):
        current_teacher_forcing_ratio = config['init_teacher_forcing_ratio'] * (
            1 - epoch / config['num_epochs'])
        current_noise_std, current_mask_ratio = get_noise_and_mask_values(
            epoch, config['num_epochs'], config)

        train_losses = train_epoch(model, train_dataloader, optimizer, current_noise_std,
                                   current_mask_ratio, current_teacher_forcing_ratio, device)
        val_losses = validate(model, val_dataloader, device)

        avg_train_loss = np.mean([loss['total'] for loss in train_losses])
        avg_val_loss = np.mean([loss['total'] for loss in val_losses])

        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(
            f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print_detailed_losses("Train", train_losses[-1])
        print_detailed_losses("Validation", val_losses[-1])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model saved!")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("Training completed!")
