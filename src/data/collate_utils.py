import torch


def stack_optional(batch, key):
    if batch and batch[0][key] is None:
        return None
    return torch.stack([item[key] for item in batch])


def collate_trajectories(batch):
    """Batch the trajectory fields shared by all native datasets."""
    return {
        "camera_trajectory": torch.stack([item["camera_trajectory"] for item in batch]),
        "subject_trajectory": stack_optional(batch, "subject_trajectory"),
        "subject_volume": stack_optional(batch, "subject_volume"),
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
    }


def collate_structured_conditioning(batch):
    """Stack structured tokens in [tokens, batch, features] order."""
    return {
        "simulation_instruction": torch.stack(
            [item["simulation_instruction"] for item in batch]
        ).transpose(0, 1),
        "cinematography_prompt": torch.stack(
            [item["cinematography_prompt"] for item in batch]
        ).transpose(0, 1),
        "simulation_instruction_parameters": [
            item["simulation_instruction_parameters"] for item in batch
        ],
        "cinematography_prompt_parameters": [
            item["cinematography_prompt_parameters"] for item in batch
        ],
    }
