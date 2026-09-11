from numbers import Integral

import torch

from data.convertor.convertor import convert_to_target

SIM_SEQ_LENGTH = 30

NUM_VISIBLE_KEYFRAMES = 4

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
        need_denormal=(batch.get("simulation_normalized", True)
                       if dataset_type in ("simulation", "lens_craft") else True),
        need_normal=(batch.get("simulation_normalized", True)
                     if dataset_type in ("simulation", "lens_craft") else True),
    )


def build_keyframing_mask(
    batch_size,
    device,
    sequence_length=SIM_SEQ_LENGTH,
    *,
    num_keyframes=NUM_VISIBLE_KEYFRAMES,
    padding_mask=None,
    sample_seeds=None,
):
    """Hide all but ``num_keyframes`` valid camera frames in each sequence.

    ``True`` means hidden, including temporal padding. K is an absolute count,
    capped at the number of valid frames; it does not scale with clip length.
    A sample with no valid frames remains fully hidden.

    Supply one integer ``sample_seeds`` entry per sample for reproducible
    evaluation. The same seed and padding choose the same ordering regardless
    of device, batch partitioning, or global RNG state. Increasing K then only
    adds visible frames, making sparse-keyframe sweeps directly comparable.
    Without seeds, masks use the global PyTorch RNG on ``device``.
    """
    for name, value, minimum in (
        ("batch_size", batch_size, 0),
        ("sequence_length", sequence_length, 1),
        ("num_keyframes", num_keyframes, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")

    shape = (batch_size, sequence_length)
    if padding_mask is not None:
        if not torch.is_tensor(padding_mask) or padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must be a boolean tensor")
        if tuple(padding_mask.shape) != shape:
            raise ValueError(f"padding_mask must have shape {shape}")

    if sample_seeds is not None:
        if torch.is_tensor(sample_seeds):
            if sample_seeds.ndim != 1:
                raise ValueError("sample_seeds must have one entry per sample")
            sample_seeds = sample_seeds.tolist()
        else:
            sample_seeds = list(sample_seeds)
        if len(sample_seeds) != batch_size:
            raise ValueError("sample_seeds must have one entry per sample")
        for seed in sample_seeds:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError("sample_seeds entries must be integers")
            if seed < 0 or seed >= 2**64:
                raise ValueError("sample_seeds entries must be in [0, 2**64)")

    sampling_device = torch.device("cpu") if sample_seeds is not None else device
    mask = torch.ones(shape, dtype=torch.bool, device=sampling_device)
    padding = (
        padding_mask.to(sampling_device)
        if padding_mask is not None
        else torch.zeros_like(mask)
    )
    for index in range(batch_size):
        valid_indices = (~padding[index]).nonzero(as_tuple=True)[0]
        generator = None
        if sample_seeds is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(sample_seeds[index]))
        ordering = torch.randperm(
            valid_indices.numel(), generator=generator, device=sampling_device
        )
        visible_indices = valid_indices[ordering[:num_keyframes]]
        mask[index, visible_indices] = False
    return mask.to(device)
