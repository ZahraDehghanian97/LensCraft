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


def _simulation_view_batch(batch, dataset_type, target_len):
    """Choose the same visible observation timeline for poses and context."""
    if dataset_type in ("simulation", "lens_craft"):
        reference = batch.get("simulation_reference")
        if reference is not None and reference["camera_trajectory"].shape[1] == target_len:
            device = batch["camera_trajectory"].device
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in reference.items()
            }
    return batch


def to_simulation_format(batch, dataset_type, *, target_len=SIM_SEQ_LENGTH):
    batch = _simulation_view_batch(batch, dataset_type, target_len)
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


def geometry_context_from_batch(batch, dataset_type="simulation", *, target_len=None):
    """Return measured optics and subject availability on the generation grid.

    ``camera_intrinsics`` means focal length in millimetres and aspect ratio;
    native converter projection defaults and ET's differently scaled
    ``intrinsics`` field are not measurements of this quantity. NaNs retain
    unknown optics. Physical ``geometry_reference`` data are evaluation targets,
    never generator conditions.
    """
    if target_len is None:
        target_len = batch["camera_trajectory"].shape[1]
    if isinstance(target_len, bool) or not isinstance(target_len, Integral) or target_len < 1:
        raise ValueError("target_len must be a positive integer")
    batch = _simulation_view_batch(batch, dataset_type, target_len)
    camera = batch["camera_trajectory"]
    batch_size, source_len = camera.shape[:2]
    if source_len < 1:
        raise ValueError("Geometry context needs at least one source frame")
    padding = batch.get("padding_mask")
    if padding is not None:
        if padding.dtype != torch.bool or padding.shape != (batch_size, source_len):
            raise ValueError("Geometry context padding must be boolean [batch, frames]")
        source_valid = ~padding.to(camera.device)
        valid_len = source_valid.sum(1)
    else:
        source_valid = torch.ones((batch_size, source_len), device=camera.device, dtype=torch.bool)
        valid_len = torch.full((batch_size,), source_len, device=camera.device, dtype=torch.long)

    # Mirror convert_to_target's valid-prefix time grid. Equal-length views
    # retain their original padding; changed lengths span every valid source
    # frame and use the complete requested output grid.
    same_grid = source_len == target_len
    target_valid_len = valid_len if same_grid else torch.where(valid_len > 0, target_len, 0)
    steps = torch.arange(target_len, device=camera.device)[None]
    target_valid = source_valid if same_grid else steps < target_valid_len[:, None]
    if same_grid:
        low = high = steps.expand(batch_size, -1)
        alpha = torch.zeros((batch_size, target_len), device=camera.device)
    else:
        position = (steps / (target_valid_len - 1).clamp_min(1)[:, None])
        position = position.clamp(max=1) * (valid_len - 1).clamp_min(0)[:, None]
        low = position.floor().long()
        high = (low + 1).minimum((valid_len - 1).clamp_min(0)[:, None])
        alpha = position - low

    def endpoints(values):
        shape = (batch_size, target_len) + values.shape[2:]
        index_shape = (batch_size, target_len) + (1,) * (values.ndim - 2)
        return (values.gather(1, index.reshape(index_shape).expand(shape)) for index in (low, high))

    context = {}
    intrinsics = batch.get("camera_intrinsics")
    if intrinsics is not None:
        if (not torch.is_tensor(intrinsics) or intrinsics.shape != (batch_size, source_len, 2)
                or not intrinsics.is_floating_point()):
            raise ValueError("camera_intrinsics must be floating [batch, frames, 2]")
        intrinsics = intrinsics.to(camera.device)
        invalid = torch.isinf(intrinsics) | ((intrinsics <= 0) & torch.isfinite(intrinsics))
        if (invalid & source_valid[..., None]).any():
            raise ValueError("Measured focal length and aspect ratio must be positive and finite")
        before, after = endpoints(intrinsics)
        # Exact source samples need only that source measurement. Otherwise
        # interpolation is available only when both endpoints are measured.
        measured = torch.isfinite(before) & (torch.isfinite(after) | (alpha[..., None] == 0))
        sampled = torch.nan_to_num(before) + alpha[..., None] * (
            torch.nan_to_num(after) - torch.nan_to_num(before)
        )
        context["camera_intrinsics"] = sampled.masked_fill(
            ~measured | ~target_valid[..., None], float("nan"),
        )

    native_unknown_subject = dataset_type in ("et", "ccdm", "gendop")
    for key in ("subject_orientation_available", "subject_dimensions_available"):
        available = batch.get(key)
        if available is None:
            if not native_unknown_subject:
                continue
            available = torch.zeros(batch_size, dtype=torch.bool, device=camera.device)
        if not torch.is_tensor(available) or available.dtype != torch.bool:
            raise ValueError(f"{key} must be boolean [batch] or [batch, frames]")
        available = available.to(camera.device)
        if available.shape == (batch_size,):
            context[key] = available
        elif key == "subject_orientation_available" and available.shape == (batch_size, source_len):
            before, after = endpoints(available)
            context[key] = before & (after | (alpha == 0)) & target_valid
        else:
            raise ValueError(f"Invalid shape for {key}: {tuple(available.shape)}")
    return context


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
