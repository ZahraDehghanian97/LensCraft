import torch
from .contrastive_loss import ContrastiveLoss
from .clip_loss import ClipLoss

from data.simulation.utils import load_clip_means
from utils.naming import clip_embedding_name
from utils.pytorch3d_transform import euler_angles_to_matrix, symmetric_orthogonalization


TRAJECTORY_LOSS_KEYS = (
    "first_frame", "relative", "speed", "rotation_absolute", "rotation_raw"
)


def rotation_frobenius_loss(R_pred, R_target):
    return ((R_pred - R_target) ** 2).mean(dim=(-2, -1)).mean()


class CameraTrajectoryLoss:
    def __init__(self,
                 contrastive_loss_margin: int=5,
                 losses_list: dict=None,
                 weighted_clip_loss: bool=False,
                 weight_power: int=1,
                 clip_weights: dict=None,
                 contrastive_loss_version: int=1,
                 clip_embeddings: dict=None,
                 encoder_loss_function: str="clip",
                 rotation_weight: float=1.0
                 ):
        self.clip_loss = ClipLoss(clip_weights=clip_weights, weight_power=weight_power)
        self.contrastive_loss_margin = contrastive_loss_margin
        self.losses_list = dict(losses_list or {})
        # Saved training configs predate these terms. Keep the correction
        # active when those configs are reused, while honoring explicit zeros
        # and leaving encoder-only objectives alone.
        if any(self.losses_list.get(key, 0) for key in TRAJECTORY_LOSS_KEYS):
            self.losses_list.setdefault("rotation_absolute", 2.0)
            self.losses_list.setdefault("rotation_raw", 1.0)
        self.weighted_clip_loss = weighted_clip_loss
        self.clip_weights = clip_weights
        self.clip_embeddings = clip_embeddings

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_means = load_clip_means()

        self.contrastive_loss = ContrastiveLoss(
            clip_embeddings=clip_embeddings,
            device=self.device,
            embedding_means=self.embedding_means,
            get_embedding_name_func=clip_embedding_name
        )

        self.contrastive_loss_version = contrastive_loss_version
        if self.contrastive_loss_version == 1:
            self.compute_contrastive_loss = self.contrastive_loss.compute_v1
        elif self.contrastive_loss_version == 2:
            self.compute_contrastive_loss = self.contrastive_loss.compute_v2
        elif self.contrastive_loss_version == 3:
            self.compute_contrastive_loss = self.contrastive_loss.compute_v3
        else:
            raise ValueError(f"Contrastive loss version should be 1 or 2, you passed {self.contrastive_loss_version}")

        self.encoder_loss_function = encoder_loss_function
        self.rotation_weight = rotation_weight

    def __call__(self, model_output, camera_trajectory, clip_target, batch, tgt_key_padding_mask=None):
        clip_pred = model_output['embeddings']
        cycle_embeddings = model_output.get('cycle_embeddings', None)

        trajectory_pred = model_output['reconstructed_raw_matrix']
        trajectory_target = self._euler_traj_to_matrix(camera_trajectory)

        return self.compute_total_loss(
            trajectory_pred,
            trajectory_target,
            clip_pred,
            clip_target,
            batch,
            cycle_embeddings,
            tgt_key_padding_mask=tgt_key_padding_mask,
            projected_pred=model_output.get('reconstructed_rot_matrix'),
        )

    def _euler_traj_to_matrix(self, traj6):
        with torch.autocast(device_type=traj6.device.type, enabled=False):
            traj6 = traj6 if traj6.dtype == torch.float64 else traj6.float()
            pos = traj6[..., :3]
            rot = euler_angles_to_matrix(traj6[..., 3:6], "XYZ")
            return torch.cat([pos, rot.reshape(*traj6.shape[:-1], 9)], dim=-1)

    def compute_total_loss(
        self, trajectory_pred, trajectory_target, clip_pred, clip_target, batch,
        cycle_embeddings=None, tgt_key_padding_mask=None, projected_pred=None,
    ):
        loss_dict = dict()

        if self.losses_list.get("clip", 0):
            clip_losses, total_clip_loss = self.clip_loss.compute(
                clip_target=clip_target,
                clip_pred=clip_pred,
                weighted_clip_loss=self.weighted_clip_loss,
                prompt_none_mask=batch.get("prompt_none_mask", None),
                encoder_loss_function=self.encoder_loss_function
            )

            loss_dict["clip"] = total_clip_loss
            loss_dict["clip_elements"] = {i: clip_losses[i] for i in range(len(clip_losses))}



        if self.losses_list.get("cycle", 0) and cycle_embeddings is not None:
            cycle_losses, total_cycle_loss = self.clip_loss.compute(
                clip_target=clip_pred,
                clip_pred=cycle_embeddings,
                weighted_clip_loss=self.weighted_clip_loss,
                encoder_loss_function=self.encoder_loss_function
            )

            loss_dict["cycle"] = total_cycle_loss
            loss_dict["cycle_elements"] = {i: cycle_losses[i] for i in range(len(cycle_losses))}

        elif self.losses_list.get("cycle", 0) and cycle_embeddings is None:
            loss_dict["cycle"] = trajectory_pred.new_zeros(())


        if any(self.losses_list.get(key, 0) for key in TRAJECTORY_LOSS_KEYS):
            loss_dict.update(self.compute_trajectory_losses(
                trajectory_pred, trajectory_target,
                tgt_key_padding_mask=tgt_key_padding_mask,
                projected_pred=projected_pred,
            ))


        if self.losses_list.get("contrastive", 0):
            contrastive_loss = self.compute_contrastive_loss(clip_pred, clip_target, batch)
            loss_dict["contrastive"] = contrastive_loss



        total_loss = trajectory_pred.new_zeros(
            (), dtype=torch.float64 if trajectory_pred.dtype == torch.float64 else torch.float32,
        )
        for loss_key in self.losses_list:
            total_loss = total_loss + self.losses_list[loss_key] * loss_dict.get(loss_key, 0)

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


    def compute_trajectory_losses(
        self, pred, target, tgt_key_padding_mask=None, projected_pred=None,
    ):
        """Return geometry objectives for raw 12-D (position + matrix) outputs.

        Temporal errors use the emitted SO(3) rotations. Applying them to raw
        matrices admits reflections: a common improper orthogonal factor Q
        cancels in (Q R_i).T @ (Q R_j). Absolute supervision anchors every
        emitted frame, and the raw target term keeps the head near proper
        rotations even when the SO(3) projection is ambiguous.
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
            return {
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
            }

    def compute_trajectory_loss(
        self, pred, target, tgt_key_padding_mask=None, projected_pred=None,
    ):
        """Compatibility wrapper returning the three original geometry terms."""
        terms = self.compute_trajectory_losses(
            pred, target, tgt_key_padding_mask, projected_pred,
        )
        return tuple(terms[key] for key in ("first_frame", "relative", "speed"))
