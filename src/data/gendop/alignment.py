import torch


def place_relative_path_at_first_pose(
    relative_path: torch.Tensor,
    ground_truth_path: torch.Tensor,
) -> torch.Tensor:
    return ground_truth_path[..., :1, :, :] @ relative_path
