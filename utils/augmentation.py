import torch
import math

def cosine_decay(initial_value, final_value, current_epoch, total_epochs):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_epoch / total_epochs))
    return final_value + (initial_value - final_value) * cosine_decay


def linear_increase(initial_value, final_value, current_epoch, total_epochs):
    return initial_value + (final_value - initial_value) * (current_epoch / total_epochs)


def get_noise_and_mask_values(current_epoch, total_epochs, config):
    noise_std = cosine_decay(
        initial_value=config['initial_noise_std'],
        final_value=config['final_noise_std'],
        current_epoch=current_epoch,
        total_epochs=total_epochs
    )

    mask_ratio = linear_increase(
        initial_value=config['initial_mask_ratio'],
        final_value=config['final_mask_ratio'],
        current_epoch=current_epoch,
        total_epochs=total_epochs
    )

    return noise_std, mask_ratio


def apply_mask_and_noise(data, mask_ratio=0.0, noise_std=0.0, device='cuda'):
    mask = torch.bernoulli(torch.full(
        (data.shape[0], data.shape[1]), 1 - mask_ratio, device=device)).bool()

    masked_data = data.clone()
    masked_data[~mask] = 0

    noisy_data = masked_data + \
        torch.normal(mean=0, std=noise_std, size=data.shape, device=device)

    src_key_padding_mask = ~mask

    return noisy_data, mask, src_key_padding_mask
