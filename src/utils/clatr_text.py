"""Text token preparation shared by native CLaTr training and evaluation."""

from typing import Tuple

import torch
from torch import Tensor


def prepare_text_features(
    sequences: Tensor, valid_lengths: Tensor, max_tokens: int
) -> Tuple[Tensor, Tensor]:
    """Pad/truncate CLIP tokens and return a True-for-valid attention mask.

    Keep at least one token per caption, matching CLaTr's existing handling
    of empty text. Features retain the source tensor's dtype and device.
    """
    batch_size, _, feature_dim = sequences.shape
    features = sequences.new_zeros((batch_size, max_tokens, feature_dim))
    mask = torch.zeros(
        (batch_size, max_tokens), dtype=torch.bool, device=sequences.device
    )
    for index in range(batch_size):
        length = max(1, min(int(valid_lengths[index].item()), max_tokens))
        features[index, :length] = sequences[index, :length]
        mask[index, :length] = True
    return features, mask
