import torch
import torch.nn.functional as F
from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor

class CCDMConvertor(BaseConvertor):
    """
    Transformer implementation using CCDM (Camera-Centered Directional Model).
    Provides conversion to and from 4x4 transformation matrices.
    """
    def __init__(self, hfov_deg: float = 45.0, aspect: float = 16 / 9):
        self.hfov_deg = hfov_deg
        self.aspect = aspect

    @staticmethod
    def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle representation to a rotation matrix."""
        axis = F.normalize(axis, dim=-1, eps=1e-10)
        
        a_x, a_y, a_z = axis.unbind(-1)
        zeros = torch.zeros_like(a_x)
        K = torch.stack([
            zeros, -a_z, a_y,
            a_z, zeros, -a_x,
            -a_y, a_x, zeros
        ], dim=-1).reshape(axis.shape[:-1] + (3, 3))

        eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
        eye = eye.expand(axis.shape[:-1] + (3, 3))
        angle = angle.unsqueeze(-1).unsqueeze(-1)
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        axis_outer = torch.einsum('...i,...j->...ij', axis, axis)

        return eye * cos_a + sin_a * K + (1.0 - cos_a) * axis_outer

    @handle_single_or_batch(arg_index=[1, 2])
    def convert_ccdm_to_transform(
        self,
        ccdm: torch.Tensor,
        subject_position: torch.Tensor | None = None
    ) -> torch.Tensor:
        x, y, z, p_x, p_y = ccdm.unbind(-1)
        rel_position = torch.stack([x, y, z], dim=-1)

        if subject_position is None:
            subject_position = torch.zeros_like(rel_position)
        cam_position = subject_position + rel_position

        fwd = F.normalize(-rel_position, dim=-1, eps=1e-10)
        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=ccdm.dtype, device=ccdm.device)
        world_up = world_up.expand_as(fwd)

        right = F.normalize(torch.cross(world_up, fwd, dim=-1), dim=-1, eps=1e-10)
        up = torch.cross(fwd, right, dim=-1)
        R_cam = torch.stack([right, up, fwd], dim=-1)

        hfov = torch.deg2rad(torch.tensor(self.hfov_deg, dtype=ccdm.dtype, device=ccdm.device))
        vfov = 2.0 * torch.atan(torch.tan(hfov / 2.0) / self.aspect)

        yaw = -torch.atan(p_x * torch.tan(hfov / 2.0))
        pitch = torch.atan(p_y * torch.tan(vfov / 2.0))

        R_yaw = self._axis_angle_to_matrix(world_up, yaw)
        rotated_right = torch.einsum('...ij,...j->...i', R_yaw, right)
        R_pitch = self._axis_angle_to_matrix(rotated_right, pitch)

        rot_mat = (
            torch.bmm(torch.bmm(R_pitch, R_yaw), R_cam)
            if ccdm.dim() > 1
            else torch.matmul(torch.matmul(R_pitch, R_yaw), R_cam)
        )

        batch_dims = ccdm.shape[:-1]
        transform = torch.eye(4, dtype=ccdm.dtype, device=ccdm.device)
        transform = transform.expand(*batch_dims, 4, 4).clone()
        transform[..., :3, :3] = rot_mat
        transform[..., :3, 3] = cam_position
        return transform

    @handle_single_or_batch(arg_index=[1, 2])
    def transform_to_ccdm(
        self,
        transform: torch.Tensor,
        subject_position: torch.Tensor | None = None
    ) -> torch.Tensor:
        cam_position = transform[..., :3, 3]
        rot_mat = transform[..., :3, :3]

        if subject_position is None:
            subject_position = torch.zeros_like(cam_position)

        rel_position = cam_position - subject_position
        v_world = -rel_position
        q_cam = torch.einsum('...ji,...j->...i', rot_mat, v_world)

        hfov = torch.deg2rad(torch.tensor(self.hfov_deg, dtype=transform.dtype, device=transform.device))
        vfov = 2.0 * torch.atan(torch.tan(hfov / 2.0) / self.aspect)

        p_x = (q_cam[..., 0] / (q_cam[..., 2] + 1e-10)) / torch.tan(hfov / 2.0)
        p_y = (q_cam[..., 1] / (q_cam[..., 2] + 1e-10)) / torch.tan(vfov / 2.0)

        return torch.cat([rel_position, p_x.unsqueeze(-1), p_y.unsqueeze(-1)], dim=-1)

    @handle_single_or_batch(arg_index=[1, 2])
    def to_standard(
        self,
        trajectory: torch.Tensor,
        subject_trajectory: None = None,
        subject_volume: None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        subject_trajectory = torch.zeros(trajectory.shape[:2] + (6), dtype=torch.float32)
        subject_position = subject_trajectory[..., :3]
        subject_volume = torch.tensor([[0.5, 1.7, 0.3]], dtype=torch.float32)

        transform = self.convert_ccdm_to_transform(trajectory, subject_position)
        return transform, subject_trajectory, subject_volume

    @handle_single_or_batch(arg_index=[1, 2])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        subject_position = (
            subject_trajectory[..., :3] if subject_trajectory is not None else None
        )
        trajectory = self.transform_to_ccdm(transform, subject_position)
        subject_trajectory = None
        subject_volume = None
        return trajectory, subject_trajectory, subject_volume
