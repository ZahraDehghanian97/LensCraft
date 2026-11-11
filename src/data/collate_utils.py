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
    """Stack ten categorical IDs and retain raw annotation metadata separately."""
    from data.categorical_conditioning import cinematography_ids_from_batch

    if not batch:
        raise ValueError("Cannot collate an empty annotation batch")
    if all("cinematography_ids" in item for item in batch):
        ids = cinematography_ids_from_batch({
            "cinematography_ids": torch.stack([item["cinematography_ids"] for item in batch]),
        })
    else:
        ids = cinematography_ids_from_batch({
            "cinematography_prompt_parameters": [item["cinematography_prompt_parameters"] for item in batch],
        })
    presence = ids >= 0
    if all("prompt_none_mask" in item for item in batch):
        declared = torch.stack([item["prompt_none_mask"] for item in batch])
        if declared.dtype != torch.bool or declared.shape != presence.shape:
            raise ValueError("Model annotation presence must be boolean [batch,10]")
        presence &= declared
    result = {"cinematography_ids": ids, "prompt_none_mask": presence}
    for name in ("cinematography_prompt_parameters", "simulation_instruction_parameters"):
        if all(name in item for item in batch):
            result[name] = [item[name] for item in batch]
    if all("structured_presence_mask" in item for item in batch):
        result["structured_presence_mask"] = torch.stack([item["structured_presence_mask"] for item in batch])
    return result
