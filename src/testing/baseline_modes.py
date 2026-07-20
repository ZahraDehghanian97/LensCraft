import torch


NORM_ITEM = "prompt_generation"
NO_NORM_ITEM = "prompt_generation_no_norm"
NORM_LENSCRAFT_INIT_ITEM = "prompt_generation_norm_lenscraft_init"

BASELINE_ITEMS = (
    NO_NORM_ITEM,
    NORM_ITEM,
    NORM_LENSCRAFT_INIT_ITEM,
)


def place_trajectory_at_first_position(
    trajectory: torch.Tensor,
    first_position: torch.Tensor,
) -> torch.Tensor:
    """Translate a path so frame zero matches ``first_position``.

    Only xyz is changed. The generated rotations and all frame-to-frame
    translation deltas are preserved.
    """
    if first_position.shape[-2:] != (1, 3):
        raise ValueError(
            "first_position must have shape [..., 1, 3], got "
            f"{tuple(first_position.shape)}"
        )
    if trajectory.shape[:-2] != first_position.shape[:-2]:
        raise ValueError(
            "trajectory and first_position batch dimensions must match: "
            f"{tuple(trajectory.shape)} vs {tuple(first_position.shape)}"
        )

    placed = trajectory.clone()
    offset = first_position - placed[..., :1, :3]
    placed[..., :3] += offset
    return placed
