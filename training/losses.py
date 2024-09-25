import torch
import numpy as np


def circular_distance_loss(pred, target):
    pred = torch.clamp(pred, min=-np.pi, max=np.pi)
    target = torch.clamp(target, min=-np.pi, max=np.pi)

    distance = torch.abs(pred - target)

    distance = torch.where(distance > np.pi, 2 * np.pi - distance, distance)

    return torch.mean(distance ** 2)


def combined_trajectory_loss(pred, target):
    pred_position = pred[:, :4]
    target_position = target[:, :4]

    pred_angle = pred[:, 4:]
    target_angle = target[:, 4:]

    position_loss = torch.nn.functional.mse_loss(
        pred_position, target_position)
    angle_loss = circular_distance_loss(pred_angle, target_angle)

    return position_loss + angle_loss


def clip_similarity_loss(pred_embedding, target_embedding):
    similarity = torch.nn.functional.cosine_similarity(
        pred_embedding, target_embedding)
    return 1 - similarity.mean()


def total_loss(trajectory_pred, trajectory_target, clip_pred, clip_target):
    trajectory_loss = combined_trajectory_loss(
        trajectory_pred, trajectory_target)

    clip_losses = {}
    for key in clip_pred.keys():
        clip_losses[key] = clip_similarity_loss(
            clip_pred[key], clip_target[key])

    total_clip_loss = sum(clip_losses.values())

    return trajectory_loss + total_clip_loss, {
        'trajectory': trajectory_loss.item(),
        'clip': {k: v.item() for k, v in clip_losses.items()},
        'total': (trajectory_loss + total_clip_loss).item()
    }


def compute_loss(model_output, camera_trajectory, clip_targets, mask=None):
    reconstructed = model_output['reconstructed']
    if mask is not None:
        reconstructed = reconstructed[mask]
        camera_trajectory = camera_trajectory[mask]

    clip_embeddings = {k: model_output[f'{k}_embedding']
                       for k in clip_targets.keys()}
    return total_loss(reconstructed, camera_trajectory, clip_embeddings, clip_targets)


def print_detailed_losses(phase, losses):
    print(f"{phase} Losses:", end=' ')
    for key, value in losses.items():
        if isinstance(value, dict):
            print(f"\n  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue:.4f}", end=' ')
        else:
            print(f"{key}: {value:.4f}", end=' ')
    print()
