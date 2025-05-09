import torch

from base_convertor import BaseConvertor
from data.ccdm.convertor import CCDMConvertor
from data.simulation.convertor import SIMConvertor
from data.et.convertor import ETConvertor
from data.convertor.utils import resample_batch_trajectories


def convert(
    source_convertor: BaseConvertor,
    target_convertor: BaseConvertor,
    trajectory: torch.Tensor,
    subject_trajectory: torch.Tensor | None = None,
    subject_volume: torch.Tensor | None = None,
    current_valid_len=300,
    target_len=30
):
    transform, subject_trajectory, subject_volume = source_convertor.to_standard(trajectory, subject_trajectory, subject_volume)
    transform, padding_mask = resample_batch_trajectories(transform, current_valid_len, target_len)
    subject_trajectory, padding_mask = resample_batch_trajectories(subject_trajectory, current_valid_len, target_len)
    trajectory, subject_trajectory, subject_volume = target_convertor.from_standard(trajectory, subject_trajectory, subject_volume)
    return trajectory, subject_trajectory, subject_volume, padding_mask


convertors = {
    "ccdm": CCDMConvertor(),
    "et": ETConvertor(),
    "simulation": SIMConvertor(),
}

def covert_to_target(
    source: str,
    target: str,
    trajectory: torch.Tensor,
    subject_trajectory: torch.Tensor | None = None,
    subject_volume: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
):
    if source == target:
        return trajectory, subject_trajectory, subject_volume, padding_mask
    
    return convert(convertors[source], convertors[target], trajectory, subject_trajectory, subject_volume)