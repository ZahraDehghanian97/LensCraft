import torch
import torch.nn.functional as F
from utils.pytorch3d_transform import axis_angle_to_matrix
from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor

class CCDMConvertor(BaseConvertor):
    def __init__(self, hfov_deg: float = 25.0, aspect: float = 1):
        self.hfov_deg = hfov_deg
        self.aspect = aspect
    
    @handle_single_or_batch(arg_specs=[(1, 2), (2, 2)])
    def convert_ccdm_to_transform(self, ccdm: torch.Tensor, subject_position: torch.Tensor | None = None) -> torch.Tensor:
        eps = 1e-8
        r = ccdm[..., 0:3]
        p_x, p_y = ccdm[..., 3:4], ccdm[..., 4:5]
        subject_position = subject_position if subject_position is not None else torch.zeros_like(r)
        c = r + subject_position
        
        hfov = torch.deg2rad(torch.tensor(self.hfov_deg, dtype=ccdm.dtype, device=ccdm.device))
        vfov = 2 * torch.atan(torch.tan(hfov / 2) / self.aspect)
        k_h, k_v = torch.tan(hfov / 2), torch.tan(vfov / 2)
        
        r_norm = torch.linalg.norm(r, dim=-1, keepdim=True)
        q_z_abs = r_norm / (torch.sqrt(1 + (p_x * k_h)**2 + (p_y * k_v)**2) + eps)
        q_x_abs, q_y_abs = q_z_abs * p_x * k_h, q_z_abs * p_y * k_v
        
        candidates = []
        for q_z_sign in [1, -1]:
            q_x, q_y, q_z = q_x_abs * q_z_sign, q_y_abs * q_z_sign, q_z_abs * q_z_sign
            den_phi = torch.sqrt(q_y**2 + q_z**2) + eps
            alpha = torch.atan2(q_z, q_y)
            beta = torch.acos(torch.clamp(-r[..., 1:2] / den_phi, -1.0, 1.0))
            
            for phi in [alpha + beta, alpha - beta]:
                sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
                A, B = q_x, -q_y * sin_phi + q_z * cos_phi
                den = A**2 + B**2 + eps
                cos_psi = (-A * r[..., 0:1] - B * r[..., 2:3]) / den
                sin_psi = (A * r[..., 2:3] - B * r[..., 0:1]) / den
                norm = torch.sqrt(sin_psi**2 + cos_psi**2) + eps
                sin_psi, cos_psi = sin_psi / norm, cos_psi / norm
                
                x_c = torch.cat([cos_psi, torch.zeros_like(cos_psi), -sin_psi], dim=-1)
                z_c = torch.cat([cos_phi * sin_psi, sin_phi, cos_phi * cos_psi], dim=-1)
                y_c = torch.cross(z_c, x_c, dim=-1)
                R = torch.stack([x_c, y_c, z_c], dim=-1)
                
                candidates.append((R, y_c[..., 1], torch.sum(z_c * r, dim=-1) > 0))
        
        best_R = candidates[0][0]
        best_score = candidates[0][1] * candidates[0][2].float() - 10 * (~candidates[0][2]).float()
        
        for R, up_dot, in_front in candidates[1:]:
            score = up_dot * in_front.float() - 10 * (~in_front).float()
            update_mask = score > best_score
            best_score = torch.where(update_mask, score, best_score)
            best_R = torch.where(update_mask[..., None, None], R, best_R)
        
        transform = torch.eye(4, dtype=ccdm.dtype, device=ccdm.device).expand(*ccdm.shape[:-1], 4, 4).clone()
        transform[..., :3, :3] = best_R
        transform[..., :3, 3] = c
        return transform

    @handle_single_or_batch(arg_specs=[(1, 3), (2, 2)])
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

    @handle_single_or_batch(arg_specs=[(1, 2), (2, 2)])
    def to_standard(
        self,
        trajectory: torch.Tensor,
        subject_trajectory: None = None,
        subject_volume: None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = trajectory.shape[:2]
        device = trajectory.device
        dtype = trajectory.dtype

        subject_position = torch.zeros((batch_size, seq_len, 3), device=device, dtype=dtype)
        subject_rot = torch.eye(3, device=device, dtype=dtype).expand(batch_size, seq_len, 3, 3)
        
        subject_transform = torch.eye(4, device=device, dtype=dtype).expand(batch_size, seq_len, 4, 4).clone()
        subject_transform[..., :3, :3] = subject_rot
        subject_transform[..., :3, 3] = subject_position

        transform = self.convert_ccdm_to_transform(trajectory, subject_position)
        subject_volume = torch.tensor([[0.5, 1.7, 0.3]], dtype=dtype, device=device)

        return transform, subject_transform, subject_volume


    @handle_single_or_batch(arg_specs=[(1, 3), (2, 3)])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        subject_position = (
            subject_trajectory[..., :3, 3] if subject_trajectory is not None else None
        )
        trajectory = self.transform_to_ccdm(transform, subject_position)
        subject_trajectory = None
        subject_volume = None
        return trajectory, subject_trajectory, subject_volume
