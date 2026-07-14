import torch

from data.simulation.dataset import SimulationDataset

ET_SUBJECT_HEIGHT = 1.7


def recenter_rescale_sim(
    camera_trajectory: torch.Tensor,
    subject_trajectory: torch.Tensor,
    subject_volume: torch.Tensor,
    target_height: float = ET_SUBJECT_HEIGHT,
    normalized: bool = True,
):
    if normalized:
        camera_trajectory, subject_trajectory, subject_volume = (
            SimulationDataset.normalize_item(
                camera_trajectory, subject_trajectory, subject_volume, False
            )
        )
    camera_trajectory = camera_trajectory.clone()
    subject_trajectory = subject_trajectory.clone()

    origin = subject_trajectory[..., :1, :3].clone()
    height = subject_volume[..., 1].reshape(-1, 1, 1)
    scale = target_height / height.clamp(min=1e-6)

    subject_trajectory[..., :3] = (subject_trajectory[..., :3] - origin) * scale
    camera_trajectory[..., :3] = (camera_trajectory[..., :3] - origin) * scale
    subject_volume = subject_volume * scale.reshape(
        scale.shape[0], *([1] * (subject_volume.dim() - 1))
    )

    return camera_trajectory, subject_trajectory, subject_volume, origin, scale


def undo_recenter_rescale(
    camera_trajectory: torch.Tensor,
    origin: torch.Tensor,
    scale: torch.Tensor,
):
    camera_trajectory = camera_trajectory.clone()
    camera_trajectory[..., :3] = camera_trajectory[..., :3] / scale + origin
    return camera_trajectory
