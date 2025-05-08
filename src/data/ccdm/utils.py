import torch
import torch.nn.functional as F
from data.convertor_utils import handle_single_or_batch

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
    
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(axis.shape[:-1] + (3, 3))
    angle = angle[..., None, None]
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    
    axis_outer = torch.einsum('...i,...j->...ij', axis, axis)
    
    return eye * cos_a + sin_a * K + (1.0 - cos_a) * axis_outer

@handle_single_or_batch()
def convert_ccdm_to_transform(ccdm: torch.Tensor, hfov_deg: float = 45.0, aspect: float = 16 / 9) -> torch.Tensor:
    """Convert from ccdm format to a 4x4 transformation matrix."""
    x, y, z, p_x, p_y = ccdm.unbind(-1)
    position = torch.stack([x, y, z], dim=-1)
    
    fwd = -position
    fwd = F.normalize(fwd, dim=-1, eps=1e-10)
    
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=ccdm.dtype, device=ccdm.device).expand_as(fwd)
    right = F.normalize(torch.cross(world_up, fwd, dim=-1), dim=-1, eps=1e-10)
    up = torch.cross(fwd, right, dim=-1)  # Already normalized
    
    R_cam = torch.stack([right, up, fwd], dim=-1)
    
    hfov = torch.deg2rad(torch.tensor(hfov_deg, dtype=ccdm.dtype, device=ccdm.device))
    vfov = 2.0 * torch.atan(torch.tan(hfov / 2.0) / aspect)
    
    yaw = -torch.atan(p_x * torch.tan(hfov / 2.0))
    pitch = torch.atan(p_y * torch.tan(vfov / 2.0))
    
    R_yaw = _axis_angle_to_matrix(world_up, yaw)
    rotated_right = torch.einsum('...ij,...j->...i', R_yaw, right)
    
    R_pitch = _axis_angle_to_matrix(rotated_right, pitch)
    
    rot_mat = torch.bmm(torch.bmm(R_pitch, R_yaw), R_cam) if ccdm.dim() > 1 else torch.matmul(torch.matmul(R_pitch, R_yaw), R_cam)
    
    batch_dims = position.shape[:-1]
    transform = torch.zeros((*batch_dims, 4, 4), dtype=ccdm.dtype, device=ccdm.device)
    transform[..., :3, :3] = rot_mat
    transform[..., :3, 3] = position
    transform[..., 3, 3] = 1.0
    
    return transform

@handle_single_or_batch()
def transform_to_ccdm(transform: torch.Tensor, hfov_deg: float = 45.0, aspect: float = 16 / 9) -> torch.Tensor:
    """Convert from 4x4 transformation matrix to ccdm format."""
    position = transform[..., :3, 3]
    rot_mat = transform[..., :3, :3]
    
    v_world = -position
    q_cam = torch.einsum('...ji,...j->...i', rot_mat, v_world)
    
    hfov = torch.deg2rad(torch.tensor(hfov_deg, dtype=transform.dtype, device=transform.device))
    vfov = 2.0 * torch.atan(torch.tan(hfov / 2.0) / aspect)
    
    p_x = (q_cam[..., 0] / (q_cam[..., 2] + 1e-10)) / torch.tan(hfov / 2.0)
    p_y = (q_cam[..., 1] / (q_cam[..., 2] + 1e-10)) / torch.tan(vfov / 2.0)
    
    return torch.cat([position, p_x.unsqueeze(-1), p_y.unsqueeze(-1)], dim=-1)
