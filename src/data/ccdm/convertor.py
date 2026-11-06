import math
import torch
from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor

class CCDMConvertor(BaseConvertor):
    # The native right-handed projection frame is X-left/Y-up/Z-forward
    # (Unity's replay negates native screen X). Stored screen Y is positive
    # down: camcontrol.cs uses Screen.height - WorldToScreenPoint(...).y.
    # Standard poses use OpenCV's
    # X-right/Y-down/Z-forward camera frame. This changes camera axes only;
    # the subject-relative world position stays in the dataset's coordinates.
    _NATIVE_TO_CV = (-1.0, -1.0, 1.0)
    _SUBJECT_BASIS = (-1.0, 1.0, -1.0)

    @classmethod
    def native_subject_rotation(cls, reference):
        return torch.diag(reference.new_tensor(cls._SUBJECT_BASIS))

    @classmethod
    def place_in_subject_frame(cls, transform, subject_transform):
        """Place native CCDM world poses at the supplied subject pose."""
        rotation = subject_transform[..., :3, :3] @ cls.native_subject_rotation(transform).T
        result = transform.clone()
        result[..., :3, :3] = rotation @ transform[..., :3, :3]
        result[..., :3, 3] = (rotation @ transform[..., :3, 3, None]).squeeze(-1) + subject_transform[..., :3, 3]
        return result

    @classmethod
    def to_native_subject_frame(cls, transform, subject_transform):
        """Undo subject placement before encoding native subject-relative features."""
        rotation = cls.native_subject_rotation(transform) @ subject_transform[..., :3, :3].transpose(-1, -2)
        result = transform.clone()
        result[..., :3, :3] = rotation @ transform[..., :3, :3]
        offset = transform[..., :3, 3] - subject_transform[..., :3, 3]
        result[..., :3, 3] = (rotation @ offset[..., None]).squeeze(-1)
        return result

    def __init__(self, hfov_deg: float | None = None, aspect: float = 16 / 9,
                 *, vfov_deg: float | None = None):
        # Unity Camera.fieldOfView is vertical. The paper's default 45 degrees
        # is modeled as vertical here; native calibration/aspect are unavailable.
        # Explicit horizontal-FOV callers retain that parameter's meaning.
        if hfov_deg is not None and vfov_deg is not None:
            raise ValueError('Specify horizontal or vertical FOV, not both')
        if not math.isfinite(aspect) or aspect <= 0:
            raise ValueError('Aspect must be finite and positive')
        if hfov_deg is None:
            vfov_deg = 45.0 if vfov_deg is None else float(vfov_deg)
            if not math.isfinite(vfov_deg) or not 0 < vfov_deg < 180:
                raise ValueError('Vertical FOV must be between zero and 180 degrees')
            hfov_deg = math.degrees(2 * math.atan(math.tan(math.radians(vfov_deg) / 2) * aspect))
        else:
            hfov_deg = float(hfov_deg)
            if not math.isfinite(hfov_deg) or not 0 < hfov_deg < 180:
                raise ValueError('Horizontal FOV must be between zero and 180 degrees')
            vfov_deg = math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) / aspect))
        self.hfov_deg, self.vfov_deg, self.aspect = hfov_deg, vfov_deg, aspect

    @handle_single_or_batch(arg_specs=[(1, 2), (2, 2)])
    def convert_ccdm_to_transform(self, ccdm: torch.Tensor, subject_position: torch.Tensor | None = None) -> torch.Tensor:
        r = ccdm[..., 0:3]
        p_x, p_y = ccdm[..., 3:4], ccdm[..., 4:5]
        subject_position = subject_position if subject_position is not None else torch.zeros_like(r)
        c = r + subject_position

        hfov = torch.deg2rad(torch.tensor(self.hfov_deg, dtype=ccdm.dtype, device=ccdm.device))
        vfov = 2 * torch.atan(torch.tan(hfov / 2) / self.aspect)
        k_h, k_v = torch.tan(hfov / 2), torch.tan(vfov / 2)

        r_norm = torch.linalg.norm(r, dim=-1, keepdim=True)
        q_z_abs = r_norm / torch.sqrt(1 + (p_x * k_h)**2 + (p_y * k_v)**2)
        q_x_abs, q_y_abs = q_z_abs * p_x * k_h, -q_z_abs * p_y * k_v

        candidates = []
        for q_z_sign in [1, -1]:
            q_x, q_y, q_z = q_x_abs * q_z_sign, q_y_abs * q_z_sign, q_z_abs * q_z_sign
            den_phi = torch.sqrt(q_y**2 + q_z**2).clamp_min(torch.finfo(ccdm.dtype).tiny)
            alpha = torch.atan2(q_z, q_y)
            beta = torch.acos(torch.clamp(-r[..., 1:2] / den_phi, -1.0, 1.0))

            for phi in [alpha + beta, alpha - beta]:
                sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
                A, B = q_x, -q_y * sin_phi + q_z * cos_phi
                # atan2 preserves a unit yaw basis even directly above/below
                # the subject, where both horizontal components vanish.
                psi = torch.atan2(
                    A * r[..., 2:3] - B * r[..., 0:1],
                    -A * r[..., 0:1] - B * r[..., 2:3],
                )
                sin_psi, cos_psi = torch.sin(psi), torch.cos(psi)

                x_c = torch.cat([cos_psi, torch.zeros_like(cos_psi), -sin_psi], dim=-1)
                z_c = torch.cat([cos_phi * sin_psi, sin_phi, cos_phi * cos_psi], dim=-1)
                y_c = torch.cross(z_c, x_c, dim=-1)
                R = torch.stack([x_c, y_c, z_c], dim=-1)

                candidates.append((R, y_c[..., 1], torch.sum(z_c * r, dim=-1) < 0))

        best_R = candidates[0][0]
        best_score = candidates[0][1] * candidates[0][2].float() - 10 * (~candidates[0][2]).float()

        for R, up_dot, in_front in candidates[1:]:
            score = up_dot * in_front.float() - 10 * (~in_front).float()
            update_mask = score > best_score
            best_score = torch.where(update_mask, score, best_score)
            best_R = torch.where(update_mask[..., None, None], R, best_R)

        transform = torch.eye(4, dtype=ccdm.dtype, device=ccdm.device).expand(*ccdm.shape[:-1], 4, 4).clone()
        flip = ccdm.new_tensor(self._NATIVE_TO_CV)
        transform[..., :3, :3] = best_R * flip
        transform[..., :3, 3] = c
        return transform

    @handle_single_or_batch(arg_specs=[(1, 3), (2, 2)])
    def transform_to_ccdm(
        self,
        transform: torch.Tensor,
        subject_position: torch.Tensor | None = None
    ) -> torch.Tensor:
        cam_position = transform[..., :3, 3]
        flip = transform.new_tensor(self._NATIVE_TO_CV)
        rot_mat = transform[..., :3, :3] * flip

        if subject_position is None:
            subject_position = torch.zeros_like(cam_position)

        rel_position = cam_position - subject_position
        v_world = -rel_position
        q_cam = torch.einsum('...ji,...j->...i', rot_mat, v_world)

        hfov = torch.deg2rad(torch.tensor(self.hfov_deg, dtype=transform.dtype, device=transform.device))
        vfov = 2.0 * torch.atan(torch.tan(hfov / 2.0) / self.aspect)

        depth = q_cam[..., 2]
        # A relative floor keeps projection finite at grazing angles without
        # changing ordinary projections when scene units are very small.
        min_depth = torch.linalg.vector_norm(q_cam, dim=-1) * torch.finfo(transform.dtype).eps
        min_depth = min_depth.clamp_min(torch.finfo(transform.dtype).tiny)
        safe_depth = depth.abs().maximum(min_depth)
        safe_depth = torch.where(depth < 0, -safe_depth, safe_depth)
        p_x = (q_cam[..., 0] / safe_depth) / torch.tan(hfov / 2.0)
        p_y = -(q_cam[..., 1] / safe_depth) / torch.tan(vfov / 2.0)

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
        # Native captions place the subject's front at world +Z and left at
        # world +X. Simulation's zero-heading subject faces -Z (left=-X).
        # Encode that fixed subject basis as a 180-degree yaw; camera world
        # positions and orientations must not be reflected or rescaled.
        subject_rot = self.native_subject_rotation(trajectory).expand(batch_size, seq_len, 3, 3)

        subject_transform = torch.eye(4, device=device, dtype=dtype).expand(batch_size, seq_len, 4, 4).clone()
        subject_transform[..., :3, :3] = subject_rot
        subject_transform[..., :3, 3] = subject_position

        transform = self.convert_ccdm_to_transform(trajectory, subject_position)
        subject_volume = torch.tensor([0.5, 1.7, 0.3], dtype=dtype, device=device).expand(batch_size, -1)

        return transform, subject_transform, subject_volume


    @handle_single_or_batch(arg_specs=[(1, 3), (2, 3)])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None, None]:
        if subject_trajectory is not None:
            transform = self.to_native_subject_frame(transform, subject_trajectory)
        trajectory = self.transform_to_ccdm(transform)
        subject_trajectory = None
        subject_volume = None
        return trajectory, subject_trajectory, subject_volume
