import torch
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor

class SIMConvertor(BaseConvertor):
    @handle_single_or_batch(arg_index=[1])
    def transform_to_sim6dof(self, transform):
        position = transform[:, :3, 3]
        
        rotation_matrix = transform[:, :3, :3]
        rotation = matrix_to_euler_angles(rotation_matrix, convention="XYZ")
        
        sim6dof = torch.cat([position, rotation], dim=1)
        return sim6dof.transpose(1, 0)
    
    @handle_single_or_batch(arg_index=[1])
    def sim6dof_to_transform(self, trajectory):
        trajectory = trajectory.transpose(1, 0)
        device = trajectory.device
        batch_size = trajectory.shape[0]
        
        position = trajectory[:, :3]
        euler = trajectory[:, 3:]
        
        transform = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        transform[:, :3, 3] = position
        
        rotation_matrix = euler_angles_to_matrix(euler, convention="XYZ")
        transform[:, :3, :3] = rotation_matrix
        
        return transform

    @handle_single_or_batch(arg_index=[1, 2])
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
            subject_transform[..., :3, :3] = torch.eye(3, device=device, dtype=dtype)
        else:
            subject_transform = torch.eye(4, device=trajectory.device, dtype=trajectory.dtype).expand_as(transform).clone()

        return transform, subject_transform, subject_volume


    @handle_single_or_batch(arg_index=[1, 2])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        trajectory = self.transform_to_sim6dof(transform)
        return trajectory, subject_trajectory, subject_volume
