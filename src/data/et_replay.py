"""Reproduce the base simulation split before constructing ET replay loaders."""

import math

import torch
from torch.utils.data import random_split


def original_simulation_splits(dataset, data_config):
    """Use exactly CameraTrajectoryDataModule's sizes, ordering, and split RNG.

    The adaptation seed must never change base train/validation/test membership.
    Movement filters belong to the original dataset config, before this split;
    evaluation-only test_fraction/test_movement_types never affect replay.
    """
    val_fraction = float(data_config.get('val_size', 0.1))
    test_fraction = float(data_config.get('test_size', 0.1))
    if (not all(math.isfinite(value) for value in (val_fraction, test_fraction))
            or min(val_fraction, test_fraction) < 0 or val_fraction + test_fraction >= 1):
        raise ValueError('Original simulation split fractions must be nonnegative with sum below one')
    seed = data_config.get('split_seed', 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError('Original simulation split_seed must be an integer in [0, 2**63)')
    train_size = int((1 - val_fraction - test_fraction) * len(dataset))
    val_size = int(val_fraction * len(dataset))
    test_size = len(dataset) - train_size - val_size
    if min(train_size, val_size) < 1:
        raise ValueError('Simulation replay requires nonempty original train and validation splits')
    subsets = random_split(dataset, [train_size, val_size, test_size],
                           generator=torch.Generator().manual_seed(seed))
    return dict(zip(('train', 'val', 'test'), subsets))
