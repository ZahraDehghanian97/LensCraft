"""Camera geometry terms shared by the categorical training objective."""
import torch

from utils.pytorch3d_transform import euler_angles_to_matrix, symmetric_orthogonalization

TRAJECTORY_LOSS_KEYS = (
    "first_frame", "relative", "speed", "rotation_absolute", "rotation_raw"
)
KEYFRAME_LOSS_KEYS = ("keyframe_position", "keyframe_rotation")
OPTIONAL_POSITION_LOSS_KEYS = ("position_absolute",)


class CameraTrajectoryLoss:
    """Geometry-only helper; categorical CE and KL live in categorical_loss."""

    def __init__(self, losses_list=None, rotation_weight=1.0):
        self.losses_list = dict(losses_list or {})
        if any(self.losses_list.get(key, 0) for key in TRAJECTORY_LOSS_KEYS):
            self.losses_list.setdefault("rotation_absolute", 2.0)
            self.losses_list.setdefault("rotation_raw", 1.0)
        self.rotation_weight = rotation_weight

    def _euler_traj_to_matrix(self, trajectory):
        with torch.autocast(device_type=trajectory.device.type, enabled=False):
            trajectory = trajectory if trajectory.dtype == torch.float64 else trajectory.float()
            rotation = euler_angles_to_matrix(trajectory[..., 3:6], "XYZ")
            return torch.cat([trajectory[..., :3], rotation.flatten(-2)], dim=-1)

    def __call__(self, model_output, camera_trajectory, semantic_target=None, batch=None,
                 tgt_key_padding_mask=None, known_mask=None):
        terms = self.compute_trajectory_losses(
            model_output['reconstructed_raw_matrix'], self._euler_traj_to_matrix(camera_trajectory),
            tgt_key_padding_mask=tgt_key_padding_mask,
            projected_pred=model_output.get('reconstructed_rot_matrix'),
            known_mask=model_output.get('known_mask') if known_mask is None else known_mask,
        )
        total = sum((weight * terms[name] for name, weight in self.losses_list.items()
                     if weight and name in terms), camera_trajectory.new_zeros(()))
        terms['total'] = total.detach().item()
        return total, terms

    def compute_trajectory_losses(
        self, pred, target, tgt_key_padding_mask=None, projected_pred=None,
        known_mask=None,
    ):
        """Return geometry objectives for raw 12-D (position + matrix) outputs.

        Temporal errors use the emitted SO(3) rotations. Applying them to raw
        matrices admits reflections: a common improper orthogonal factor Q
        cancels in (Q R_i).T @ (Q R_j). Absolute supervision anchors every
        emitted frame, and the raw target term keeps the head near proper
        rotations even when the SO(3) projection is ambiguous.

        ``known_mask`` marks supplied poses, independently of temporal padding.
        Keyframe terms average over those valid poses only. The rotation term
        is the mean squared chordal distance between proper SO(3) matrices,
        ``8 sin(theta / 2)^2 / 9``, scaled by ``rotation_weight``. This avoids
        the endpoint derivatives of acos; evaluation reports geodesic degrees.
        Terms contribute according to their configured geometry coefficients.

        ``position_absolute`` is an optional all-valid-frame position MSE.
        It anchors offsets throughout the path, independently of frame zero
        and the relative/step objectives. Its units match the positions passed
        to this loss (normalized simulation coordinates during training).
        """
        if pred.ndim != 3 or pred.shape[-1] != 12 or pred.shape != target.shape:
            raise ValueError("Expected matching (batch, frames, 12) trajectories")
        if pred.shape[1] == 0:
            raise ValueError("Trajectories must contain at least one frame")
        if projected_pred is not None and projected_pred.shape != pred.shape:
            raise ValueError("Projected trajectory must match the raw trajectory shape")
        if tgt_key_padding_mask is None:
            valid = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
        else:
            if tgt_key_padding_mask.shape != pred.shape[:2]:
                raise ValueError("Padding mask must match (batch, frames)")
            valid = ~tgt_key_padding_mask.to(device=pred.device, dtype=torch.bool)
        if known_mask is None:
            known_valid = torch.zeros_like(valid)
        else:
            if (not torch.is_tensor(known_mask) or known_mask.dtype != torch.bool
                    or known_mask.shape != pred.shape[:2]):
                raise ValueError("known_mask must be a boolean (batch, frames) tensor")
            known_valid = known_mask.to(device=pred.device) & valid

        # Float casts alone do not prevent autocast from reducing precision
        # again in the relative/speed matrix products.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            dtype = torch.float64 if pred.dtype == torch.float64 else torch.float32
            pred, target = pred.to(dtype), target.to(dtype)
            p_pred = torch.where(valid[..., None], pred[..., :3], 0.0)
            p_tgt = torch.where(valid[..., None], target[..., :3], 0.0)
            identity = torch.eye(3, dtype=dtype, device=pred.device)
            matrix_valid = valid[..., None, None]
            raw_rot = torch.where(
                matrix_valid, pred[..., 3:].reshape(*pred.shape[:2], 3, 3), identity,
            )
            R_tgt = torch.where(
                matrix_valid, target[..., 3:].reshape(*target.shape[:2], 3, 3), identity,
            )
            if projected_pred is None:
                # Sanitize padding before SVD, including non-finite padding.
                R_pred = symmetric_orthogonalization(raw_rot)
            else:
                R_pred = torch.where(
                    matrix_valid,
                    projected_pred[..., 3:].to(dtype).reshape(*pred.shape[:2], 3, 3),
                    identity,
                )

            def masked_mean(values, mask):
                return values.masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1)

            def position_error(a, b, mask):
                return masked_mean((a - b).square().mean(dim=-1), mask)

            def rotation_error(a, b, mask):
                return self.rotation_weight * masked_mean(
                    (a - b).square().mean(dim=(-2, -1)), mask,
                )

            first_valid = valid[:, :1]
            relative_valid = first_valid & valid[:, 1:]
            speed_valid = valid[:, :-1] & valid[:, 1:]
            losses = {
                "first_frame": (
                    position_error(p_pred[:, :1], p_tgt[:, :1], first_valid)
                    + rotation_error(R_pred[:, :1], R_tgt[:, :1], first_valid)
                ),
                "relative": (
                    position_error(
                        p_pred[:, 1:] - p_pred[:, :1],
                        p_tgt[:, 1:] - p_tgt[:, :1], relative_valid,
                    )
                    + rotation_error(
                        R_pred[:, :1].transpose(-1, -2) @ R_pred[:, 1:],
                        R_tgt[:, :1].transpose(-1, -2) @ R_tgt[:, 1:], relative_valid,
                    )
                ),
                "speed": (
                    position_error(
                        p_pred[:, 1:] - p_pred[:, :-1],
                        p_tgt[:, 1:] - p_tgt[:, :-1], speed_valid,
                    )
                    + rotation_error(
                        R_pred[:, :-1].transpose(-1, -2) @ R_pred[:, 1:],
                        R_tgt[:, :-1].transpose(-1, -2) @ R_tgt[:, 1:], speed_valid,
                    )
                ),
                "rotation_absolute": rotation_error(R_pred, R_tgt, valid),
                "rotation_raw": rotation_error(raw_rot, R_tgt, valid),
                "keyframe_position": position_error(p_pred, p_tgt, known_valid),
                "keyframe_rotation": rotation_error(R_pred, R_tgt, known_valid),
            }
            if self.losses_list.get("position_absolute", 0):
                losses["position_absolute"] = position_error(p_pred, p_tgt, valid)
            return losses
