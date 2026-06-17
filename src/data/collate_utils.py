import torch


def stack_optional(batch, key):
    if batch and batch[0][key] is None:
        return None
    return torch.stack([item[key] for item in batch])
