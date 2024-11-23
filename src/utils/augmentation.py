import torch


def linear_increase(initial_value, final_value, current_epoch, total_epochs):
    return initial_value + (final_value - initial_value) * (current_epoch / total_epochs)


def apply_mask_and_noise(data, valid_len=None, mask_ratio=0.0, noise_std=0.0, device='cuda'):
    batch_size, seq_len = data.shape[0], data.shape[1]

    if valid_len is None:
        padded_mask = torch.bernoulli(torch.full(
            (batch_size, seq_len), 1 - mask_ratio, device=device)).bool()
    else:
        padded_mask = torch.arange(seq_len, device=device)[
            None, :] < valid_len[:, None]

        if mask_ratio > 0:
            for i in range(batch_size):
                valid_length = valid_len[i]

                num_masks = int(valid_length.item() * mask_ratio)

                mask_indices = torch.randperm(
                    valid_length, device=device)[:num_masks]
                padded_mask[i, mask_indices] = False

    noisy_data = data.clone()
    if noise_std > 0:
        noise = torch.normal(mean=0, std=noise_std,
                             size=data.shape, device=device)
        noisy_data = noisy_data + noise

    padded_mask = ~padded_mask
    noisy_data[padded_mask] = 0

    return noisy_data, padded_mask
