import torch
from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor
from data.et.config import STANDARDIZATION_CONFIG
from utils.pytorch3d_transform import matrix_to_rotation_6d, rotation_6d_to_matrix

class ETConvertor(BaseConvertor):
    def __init__(self):
        self.augmentation = None
        self.velocity = STANDARDIZATION_CONFIG["velocity"]
    
    @handle_single_or_batch(arg_specs=[(1, 3)])
    def get_feature(self, raw_matrix_trajectory):
        matrix_trajectory = torch.clone(raw_matrix_trajectory)

        raw_trans = torch.clone(matrix_trajectory[..., :3, 3])
        if self.velocity:
            velocity = raw_trans[:, 1:] - raw_trans[:, :-1]
            raw_trans = torch.cat([raw_trans[:, 0:1], velocity], dim=1)

        rot_matrices = matrix_trajectory[..., :3, :3]
        rot6d = matrix_to_rotation_6d(rot_matrices)
        
        return torch.cat([rot6d, raw_trans], dim=-1)
        

    @handle_single_or_batch(arg_specs=[(1, 2)])
    def get_matrix(self, rot6d_trajectory):
        device = rot6d_trajectory.device
        batch_size = rot6d_trajectory.shape[0]
        num_cams = rot6d_trajectory.shape[1]
        
        matrix_trajectory = torch.eye(4, device=device).expand(batch_size, num_cams, 4, 4).clone()

        raw_trans = rot6d_trajectory[..., 6:]
        if self.velocity:
            raw_trans = torch.cumsum(raw_trans, dim=1)
        matrix_trajectory[..., :3, 3] = raw_trans
        matrix_trajectory[..., :3, :3] = rotation_6d_to_matrix(rot6d_trajectory[..., :6])

        return matrix_trajectory

    @handle_single_or_batch(arg_specs=[(1, 3), (2, 3)])
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


    @handle_single_or_batch(arg_specs=[(1, 3), (2, 3)])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:        
        processed_subject_trajectory = None
        if subject_trajectory is not None:
            processed_subject_trajectory = subject_trajectory[..., :3, 3]
            
        trajectory = self.get_feature(transform)
        subject_volume = None
        return trajectory, processed_subject_trajectory, subject_volume
