import torch

from data.convertor.convertor import convert_to_target

SIM_SEQ_LENGTH = 30

NUM_VISIBLE_KEYFRAMES = 26

MEMORY_TEACHER_FORCING_BY_MODE = {
    "reconstruction": 0.0,
    "key_framing": 0.0,
    "prompt_generation": 1.0,
    "key_framing+prompt": 0.5,
    "hybrid_generation": 0.5,
    "source_trajectory": 0.5,
}


def to_simulation_format(batch, dataset_type, *, target_len=SIM_SEQ_LENGTH):
    if dataset_type in ("simulation", "lens_craft"):
        reference = batch.get("simulation_reference")
        if reference is not None and reference["camera_trajectory"].shape[1] == target_len:
            device = batch["camera_trajectory"].device
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in reference.items()
            }
    if dataset_type in ("simulation", "lens_craft") and batch["camera_trajectory"].shape[1] == target_len:
        return (
            batch["camera_trajectory"],
            batch["subject_trajectory"],
            batch["subject_volume"],
            batch["padding_mask"],
        )
    return convert_to_target(
        dataset_type, "simulation",
        batch["camera_trajectory"], batch["subject_trajectory"],
        batch["subject_volume"], batch["padding_mask"], target_len,
        valid_target_len=(
            (~batch["padding_mask"]).sum(dim=1)
            if batch["camera_trajectory"].shape[1] == target_len
            and batch["padding_mask"] is not None else None
        ),
    )


def build_keyframing_mask(batch_size, device, sequence_length=SIM_SEQ_LENGTH):
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    visible_ratio = NUM_VISIBLE_KEYFRAMES / SIM_SEQ_LENGTH
    visible_count = min(
        sequence_length,
        max(1, round(sequence_length * visible_ratio)),
    )
    hidden_count = sequence_length - visible_count
    template = torch.cat([
        # PyTorch key-padding masks use True for hidden/padded tokens.
        torch.zeros(visible_count, dtype=torch.bool, device=device),
        torch.ones(hidden_count, dtype=torch.bool, device=device),
    ])
    mask = torch.empty((batch_size, sequence_length), dtype=torch.bool, device=device)
    for i in range(batch_size):
        mask[i] = template[torch.randperm(sequence_length, device=device)]
    return mask
