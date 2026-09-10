import torch

from .base_convertor import BaseConvertor
from data.convertor.utils import handle_single_or_batch, resample_batch_trajectories


def _pad_native_features(target, trajectory, subject_trajectory, padding_mask):
    if padding_mask is None:
        return trajectory, subject_trajectory
    if target == "ccdm":
        # CCDM training extends short paths by repeating the final frame.
        counts = (~padding_mask).sum(dim=1)
        last_index = (counts - 1).clamp(min=0)
        last = trajectory[torch.arange(trajectory.shape[0], device=trajectory.device), last_index]
        last = last.masked_fill((counts == 0)[:, None], 0.0)
        trajectory = torch.where(padding_mask[..., None], last[:, None], trajectory)
    elif target != "gendop":
        trajectory = trajectory.masked_fill(padding_mask[..., None], 0.0)
    if subject_trajectory is not None:
        subject_trajectory = subject_trajectory.masked_fill(padding_mask[..., None], 0.0)
    return trajectory, subject_trajectory


@handle_single_or_batch(arg_specs=[
    (2, lambda args: 3 if args["source"] == "gendop" else 2),
    (3, 2),
    (4, lambda args: 2 if args["source"] in ("simulation", "lens_craft") else 1),
    (5, 1),
    (7, 0),
])
def convert_to_target(
    source: str,
    target: str,
    trajectory: torch.Tensor,
    subject_trajectory: torch.Tensor | None = None,
    subject_volume: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    target_len=30,
    valid_target_len=None,
    convertors=None,
    need_denormal=True,
    need_normal=True,
):
    if source == 'lens_craft':
        source = 'simulation'
    if target == 'lens_craft':
        target = 'simulation'
    if convertors is None:
        from .constant import default_convertors
        convertors = default_convertors

    if padding_mask is not None:
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        valid_lengths = (~padding_mask).sum(dim=1)
    else:
        valid_lengths = torch.full(
            (trajectory.shape[0],),
            trajectory.shape[1],
            dtype=torch.long,
            device=trajectory.device,
        )

    source_convertor: BaseConvertor = convertors[source]
    target_convertor: BaseConvertor = convertors[target]

    batch_size = trajectory.shape[0]

    if source == target and trajectory.shape[1] == target_len and valid_target_len is None:
        if need_denormal != need_normal:
            from .constant import default_normalizers
            trajectory, subject_trajectory, subject_volume = default_normalizers[source](
                trajectory, subject_trajectory, subject_volume, need_normal
            )
        trajectory, subject_trajectory = _pad_native_features(
            target, trajectory, subject_trajectory, padding_mask
        )
        return trajectory, subject_trajectory, subject_volume, padding_mask

    if need_denormal:
        from .constant import default_normalizers
        trajectory, subject_trajectory, subject_volume = default_normalizers[source](
            trajectory, subject_trajectory, subject_volume, False
        )

    transform, subject_trajectory, subject_volume = source_convertor.to_standard(
        trajectory, subject_trajectory, subject_volume
    )

    if (
        subject_volume is not None
        and subject_volume.shape[0] == 1
        and batch_size != 1
    ):
        subject_volume = subject_volume.repeat(batch_size, 1)

    transform, padding_mask = resample_batch_trajectories(
        transform, valid_lengths, target_len, valid_target_len
    )
    if subject_trajectory is not None:
        subject_trajectory, padding_mask = resample_batch_trajectories(
            subject_trajectory, valid_lengths, target_len, valid_target_len
        )

    trajectory, subject_trajectory, subject_volume = target_convertor.from_standard(
        transform, subject_trajectory, subject_volume
    )

    if need_normal:
        from .constant import default_normalizers
        trajectory, subject_trajectory, subject_volume = default_normalizers[target](
            trajectory, subject_trajectory, subject_volume, True
        )

    # Native datasets pad features after normalization. In particular, ET's
    # velocity encoding otherwise produces a jump to the origin at the first
    # padded frame, and its character attention consumes those padded values.
    trajectory, subject_trajectory = _pad_native_features(
        target, trajectory, subject_trajectory, padding_mask
    )

    return trajectory, subject_trajectory, subject_volume, padding_mask
