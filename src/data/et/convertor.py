import torch
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor
from data.et.config import STANDARDIZATION_CONFIG

class ETConvertor(BaseConvertor):
    def __init__(self, standardize: bool=True, mean_std=STANDARDIZATION_CONFIG):
        self.augmentation = None
        self.standardize = standardize
        if self.standardize:
            self.norm_mean = torch.Tensor(mean_std["norm_mean"])
            self.norm_std = torch.Tensor(mean_std["norm_std"])
            self.shift_mean = torch.Tensor(mean_std["shift_mean"])
            self.shift_std = torch.Tensor(mean_std["shift_std"])
            self.velocity = mean_std["velocity"]
    
    @handle_single_or_batch(arg_specs=[(1, 3)])
    def get_feature(self, raw_matrix_trajectory):
        matrix_trajectory = torch.clone(raw_matrix_trajectory)
        device = matrix_trajectory.device

        raw_trans = torch.clone(matrix_trajectory[..., :3, 3])
        if self.velocity:
            velocity = raw_trans[1:] - raw_trans[:-1]
            raw_trans = torch.cat([raw_trans[0][None], velocity])
        if self.standardize:
            raw_trans[0] -= self.shift_mean.to(device)
            raw_trans[0] /= self.shift_std.to(device)
            raw_trans[1:] -= self.norm_mean.to(device)
            raw_trans[1:] /= self.norm_std.to(device)

        rot_matrices = matrix_trajectory[..., :3, :3]
        rot6d = matrix_to_rotation_6d(rot_matrices)
        
        return torch.cat([rot6d.reshape(-1, 6), raw_trans], dim=-1).permute(1, 0)

    @handle_single_or_batch(arg_specs=[(1, 2)])
    def get_matrix(self, raw_rot6d_trajectory):
        rot6d_trajectory = torch.clone(raw_rot6d_trajectory)
        device = rot6d_trajectory.device

        num_cams = rot6d_trajectory.shape[1]
        matrix_trajectory = torch.eye(4, device=device).expand(num_cams, 4, 4).clone()

        raw_trans = rot6d_trajectory[6:].permute(1, 0)
        if self.standardize:
            raw_trans[0] *= self.shift_std.to(device)
            raw_trans[0] += self.shift_mean.to(device)
            raw_trans[1:] *= self.norm_std.to(device)
            raw_trans[1:] += self.norm_mean.to(device)
        if self.velocity:
            raw_trans = torch.cumsum(raw_trans, dim=0)
        matrix_trajectory[..., :3, 3] = raw_trans

        rot6d = rot6d_trajectory[:6].permute(1, 0).reshape(-1, 6)
        matrix_trajectory[..., :3, :3] = rotation_6d_to_matrix(rot6d)

        return matrix_trajectory

    @handle_single_or_batch(arg_specs=[(1, 2), (1, 2)])
    def to_standard(
        self,
        trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = trajectory.device
        dtype = trajectory.dtype
        batch_size, seq_len = trajectory.shape[:2]

        transform = self.get_matrix(trajectory)

        if subject_volume is None:
            subject_volume = torch.tensor([[0.5, 1.7, 0.3]], dtype=dtype, device=device)

        if subject_trajectory is not None:
            subject_positions = subject_trajectory[..., :3]
        else:
            subject_positions = torch.zeros((batch_size, seq_len, 3), dtype=dtype, device=device)

        subject_transform = torch.eye(4, device=device, dtype=dtype).expand(batch_size, seq_len, 4, 4).clone()
        subject_transform[..., :3, :3] = torch.eye(3, device=device, dtype=dtype)
        subject_transform[..., :3, 3] = subject_positions

        return transform, subject_transform, subject_volume


    @handle_single_or_batch(arg_specs=[(1, 3), (1, 3)])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:        
        processed_subject_trajectory = None
        if subject_trajectory is not None:
            processed_subject_trajectory = subject_trajectory[..., :3]
            
        trajectory = self.get_feature(transform)
        subject_volume = None
        return trajectory, processed_subject_trajectory, subject_volume
