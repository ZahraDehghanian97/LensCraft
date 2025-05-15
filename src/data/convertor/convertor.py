import torch

from .base_convertor import BaseConvertor
from data.convertor.utils import handle_single_or_batch, resample_batch_trajectories

@handle_single_or_batch(arg_specs=[(2, 2), (3, 2), (5, 0), (7, 0)])
def convert(
    source_convertor: BaseConvertor,
    target_convertor: BaseConvertor,
    trajectory: torch.Tensor,
    subject_trajectory: torch.Tensor | None = None,
    subject_volume: torch.Tensor | None = None,
    current_valid_len: torch.Tensor | None = None,
    target_len=30,
    valid_target_len: torch.Tensor | None = None,
):
    transform, subject_trajectory, subject_volume = source_convertor.to_standard(trajectory, subject_trajectory, subject_volume)
    transform, padding_mask = resample_batch_trajectories(transform, current_valid_len, target_len, valid_target_len)
    subject_trajectory, padding_mask = resample_batch_trajectories(subject_trajectory, current_valid_len, target_len, valid_target_len)
    trajectory, subject_trajectory, subject_volume = target_convertor.from_standard(transform, subject_trajectory, subject_volume)
    return trajectory, subject_trajectory, subject_volume, padding_mask


@handle_single_or_batch(arg_specs=[(2, 2), (3, 2), (5, 1), (7, 0)])
def convert_to_target(
    source: str,
    target: str,
    trajectory: torch.Tensor,
    subject_trajectory: torch.Tensor | None = None,
    subject_volume: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    target_len=30,
    valid_target_len=None,
    convertors=None
):
    if source == 'lens_craft':
        source = 'simulation'
    if target == 'lens_craft':
        target = 'simulation'
    if source == target:
        return trajectory, subject_trajectory, subject_volume, padding_mask
    
    
    if convertors is None:
        from .constant import default_convertors
        convertors = default_convertors
    if padding_mask is not None:
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        
        valid_lengths = (~padding_mask).sum(dim=1)
    else:
        valid_lengths = torch.full((trajectory.shape[0],), trajectory.shape[1], dtype=torch.long, device=trajectory.device)
    
    return convert(
        convertors[source], 
        convertors[target], 
        trajectory, 
        subject_trajectory, 
        subject_volume,
        valid_lengths,
        target_len,
        valid_target_len
    )
