import torch
from utils.pytorch3d_transform import matrix_to_euler_angles, euler_angles_to_matrix
from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor

class SIMConvertor(BaseConvertor):
    @handle_single_or_batch(arg_specs=[(1, 3)])
    def transform_to_sim6dof(self, transform):
        position = transform[..., :3, 3]
        
        rotation_matrix = transform[..., :3, :3]
        rotation = matrix_to_euler_angles(rotation_matrix, convention="XYZ")
        
        return torch.cat([position, rotation], dim=-1)
    
    @handle_single_or_batch(arg_specs=[(1, 2)])
    def sim6dof_to_transform(self, trajectory):
        batch_size, seq_len = trajectory.shape[:2]
        device = trajectory.device
        dtype = trajectory.dtype
        
        transform = torch.eye(4, device=device, dtype=dtype).expand(batch_size, seq_len, 4, 4).clone()
        transform[..., :3, 3] = trajectory[..., :3]
        transform[..., :3, :3] = euler_angles_to_matrix(trajectory[..., 3:], convention="XYZ")
        
        return transform

    @handle_single_or_batch(arg_specs=[(1, 2), (2, 2)])
    def to_standard(
        self,
        trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = self.sim6dof_to_transform(trajectory)
        
        if subject_trajectory is not None:
            batch_size, seq_len = subject_trajectory.shape[:2]
            device = subject_trajectory.device
            dtype = subject_trajectory.dtype
            
            subject_transform = torch.eye(4, device=device, dtype=dtype).expand(batch_size, seq_len, 4, 4).clone()
            
            subject_transform[..., :3, 3] = subject_trajectory[..., :3]
            euler_angles = subject_trajectory[..., 3:]
            
            for b in range(batch_size):
                for t in range(seq_len):
                    rotation_matrix = euler_angles_to_matrix(euler_angles[b, t], convention="XYZ")
                    subject_transform[b, t, :3, :3] = rotation_matrix
        else:
            subject_transform = torch.eye(4, device=trajectory.device, dtype=trajectory.dtype).expand_as(transform).clone()

        return transform, subject_transform, subject_volume.squeeze(1) if subject_volume is not None else None


    @handle_single_or_batch(arg_specs=[(1, 3), (2, 3)])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory = self.transform_to_sim6dof(transform)
        
        if subject_trajectory is not None:
            subject_position = subject_trajectory[..., :3, 3]
            subject_rotation_matrix = subject_trajectory[..., :3, :3]
            subject_rotation = matrix_to_euler_angles(subject_rotation_matrix, convention="XYZ")
            
            subject_traj = torch.cat([subject_position, subject_rotation], dim=-1)
        else:
            subject_traj = None
        
        return trajectory, subject_traj, subject_volume.unsqueeze(1) if subject_volume is not None else None
