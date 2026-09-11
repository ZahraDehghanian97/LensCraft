import torch


def linear_increase(initial_value, final_value, current_epoch, total_epochs):
    """Linearly interpolate over epochs, including both configured endpoints.

    Epoch indices run from ``0`` through ``total_epochs - 1``.  Dividing by
    ``total_epochs`` therefore never reached the final value.  A one-epoch run
    has only its endpoint, so it deliberately uses ``final_value``.
    """
    if total_epochs < 1:
        raise ValueError("total_epochs must be at least 1")

    if total_epochs == 1:
        progress = 1.0
    else:
        progress = min(max(current_epoch / (total_epochs - 1), 0.0), 1.0)
    return initial_value + (final_value - initial_value) * progress


def apply_mask_and_noise(data, valid_len=None, mask_ratio=0.0, noise_std=0.0, device='cuda'):
    batch_size, seq_len = data.shape[0], data.shape[1]

    if valid_len is None:
        padded_mask = torch.bernoulli(torch.full(
            (batch_size, seq_len), 1 - mask_ratio, device=device)).bool()
    else:
        positions = torch.arange(seq_len, device=device)[None, :]
        padded_mask = positions < valid_len[:, None]

        if mask_ratio > 0:
            num_masks = (valid_len.to(torch.float64) * mask_ratio).to(torch.long)
            random_keys = torch.rand((batch_size, seq_len), device=device)
            random_keys.masked_fill_(~padded_mask, float('inf'))
            random_order = random_keys.argsort(dim=1)
            selected = torch.zeros_like(padded_mask).scatter_(
                1, random_order, positions < num_masks[:, None]
            )
            padded_mask = padded_mask & ~selected

    noisy_data = data.clone()
    if noise_std > 0:
        noise = torch.normal(mean=0, std=noise_std,
                             size=data.shape, device=device)
        noisy_data = noisy_data + noise

    padded_mask = ~padded_mask
    noisy_data[padded_mask] = 0

    return noisy_data, padded_mask
