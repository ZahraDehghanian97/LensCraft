"""Place a generated native CCDM path in a simulation subject's world frame."""
import torch

from data.ccdm.convertor import CCDMConvertor
from data.simulation.convertor import SIMConvertor


def align_ccdm_to_subject(camera: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
    """Inputs and output are denormalized simulation poses on the same timeline.

    Only subject poses determine placement; no reference camera values enter.
    Both camera positions and rotations follow the subject's full rotation.
    """
    if camera.shape[:-1] != subject.shape[:-1] or camera.shape[-1] != 6 or subject.shape[-1] != 6:
        raise ValueError("CCDM placement requires matching camera/subject [batch, time, 6] poses")
    convertor = SIMConvertor()
    transform, subject_transform, _ = convertor.to_standard(camera, subject)
    placed = CCDMConvertor.place_in_subject_frame(transform, subject_transform)
    return convertor.from_standard(placed)[0]
