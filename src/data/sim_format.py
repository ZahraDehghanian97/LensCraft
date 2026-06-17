import torch

from data.convertor.convertor import convert_to_target

SIM_SEQ_LENGTH = 30

NUM_VISIBLE_KEYFRAMES = 26
NUM_HIDDEN_KEYFRAMES = SIM_SEQ_LENGTH - NUM_VISIBLE_KEYFRAMES  # 4

MEMORY_TEACHER_FORCING_BY_MODE = {
    "reconstruction": 0.0,
    "key_framing": 0.0,
    "prompt_generation": 1.0,
    "key_framing+prompt": 0.5,
    "hybrid_generation": 0.5,
    "source_trajectory": 0.5,
}


def to_simulation_format(batch, dataset_type, *, target_len=SIM_SEQ_LENGTH):
    if dataset_type == "simulation":
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
    )


def build_keyframing_mask(batch_size, device):
    template = torch.cat([
        torch.ones(NUM_VISIBLE_KEYFRAMES, dtype=torch.bool, device=device),
        torch.zeros(NUM_HIDDEN_KEYFRAMES, dtype=torch.bool, device=device),
    ])
    mask = torch.empty((batch_size, SIM_SEQ_LENGTH), dtype=torch.bool, device=device)
    for i in range(batch_size):
        mask[i] = template[torch.randperm(SIM_SEQ_LENGTH, device=device)]
    return mask
