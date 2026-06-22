import torch
from torch.nn.functional import mse_loss
from .contrastive_loss import ContrastiveLoss
from .clip_loss import ClipLoss

from data.simulation.utils import load_clip_means, cinematography_struct_size
from utils.naming import clip_embedding_name
from utils.pytorch3d_transform import euler_angles_to_matrix


def rotation_frobenius_loss(R_pred, R_target):
    return ((R_pred - R_target) ** 2).mean(dim=(-2, -1)).mean()


@torch.no_grad()
def report_traj_term_scales(pred, target):
    """pred/target are the 12-D (pos + flat R) tensors."""
    p_p, R_p = pred[..., :3], pred[..., 3:].reshape(*pred.shape[:-1], 3, 3)
    p_t, R_t = target[..., :3], target[..., 3:].reshape(*target.shape[:-1], 3, 3)
    rot = lambda a, b: ((a - b) ** 2).mean(dim=(-2, -1)).mean().item()

    pos_ff = mse_loss(p_p[:, 0:1], p_t[:, 0:1]).item()
    rot_ff = rot(R_p[:, 0:1], R_t[:, 0:1])
    rel_p = mse_loss(p_p[:, 1:] - p_p[:, 0:1], p_t[:, 1:] - p_t[:, 0:1]).item()
    rel_r = rot(R_p[:, 0:1].transpose(-1, -2) @ R_p[:, 1:],
                R_t[:, 0:1].transpose(-1, -2) @ R_t[:, 1:])
    spd_p = mse_loss(p_p[:, 1:] - p_p[:, :-1], p_t[:, 1:] - p_t[:, :-1]).item()
    spd_r = rot(R_p[:, :-1].transpose(-1, -2) @ R_p[:, 1:],
                R_t[:, :-1].transpose(-1, -2) @ R_t[:, 1:])

    pos_tot, rot_tot = pos_ff + rel_p + spd_p, rot_ff + rel_r + spd_r
    print(f"first_frame  pos={pos_ff:.4f}  rot={rot_ff:.4f}")
    print(f"relative     pos={rel_p:.4f}  rot={rel_r:.4f}")
    print(f"speed        pos={spd_p:.4f}  rot={spd_r:.4f}")
    print(f"TOTAL        pos={pos_tot:.4f}  rot={rot_tot:.4f}")
    print(f"-> rotation_weight for pos≈rot: {pos_tot / max(rot_tot, 1e-9):.3f}")

class CameraTrajectoryLoss:
    def __init__(self,
                 contrastive_loss_margin: int=5,
                 losses_list: list=[],
                 weighted_clip_loss: bool=False,
                 weight_power: int=1,
                 clip_weights: dict=None,
                 contrastive_loss_version: int=1,
                 clip_embeddings: dict=None,
                 encoder_loss_function: str="clip",
                 rotation_weight: float=1.0   # global position:rotation balance (svd9d matrix loss)
                 ):
        self.clip_loss = ClipLoss(clip_weights=clip_weights, weight_power=weight_power)
        self.contrastive_loss_margin = contrastive_loss_margin
        self.losses_list = losses_list
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

        trajectory_pred = model_output['reconstructed_rot_matrix']  # (..., 12): pos + R_flat
        trajectory_target = self._euler_traj_to_matrix(camera_trajectory)

        if not getattr(self, "_scale_reported", False):
            report_traj_term_scales(trajectory_pred, trajectory_target)
            self._scale_reported = True

        return self.compute_total_loss(
            trajectory_pred,
            trajectory_target,
            clip_pred,
            clip_target,
            batch,
            cycle_embeddings
        )

    def _euler_traj_to_matrix(self, traj6):
        pos = traj6[..., :3]
        rot = euler_angles_to_matrix(traj6[..., 3:6], "XYZ")
        return torch.cat([pos, rot.reshape(*traj6.shape[:-1], 9)], dim=-1)

    def compute_total_loss(self, trajectory_pred, trajectory_target, clip_pred, clip_target, batch, cycle_embeddings=None):
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
            n_high = cinematography_struct_size
            cycle_losses, total_cycle_loss = self.clip_loss.compute(
                clip_target=clip_pred[:n_high],
                clip_pred=cycle_embeddings[:n_high],
                weighted_clip_loss=self.weighted_clip_loss,
                encoder_loss_function=self.encoder_loss_function
            )

            loss_dict["cycle"] = total_cycle_loss
            loss_dict["cycle_elements"] = {i: cycle_losses[i] for i in range(len(cycle_losses))}

        elif self.losses_list.get("cycle", 0) and cycle_embeddings is None:
            loss_dict["cycle"] = torch.tensor(0)


        if self.losses_list.get("first_frame", 0) or self.losses_list.get("relative", 0) or self.losses_list.get("speed", 0):
            first_frame_loss, relative_loss, speed_loss = self.compute_trajectory_loss(trajectory_pred, trajectory_target)

            if self.losses_list.get("first_frame", 0):
                loss_dict["first_frame"] = first_frame_loss

            if self.losses_list.get("relative", 0):
                loss_dict["relative"] = relative_loss

            if self.losses_list.get("speed", 0):
                loss_dict["speed"] = speed_loss


        if self.losses_list.get("contrastive", 0):
            contrastive_loss = self.compute_contrastive_loss(clip_pred, clip_target, batch)
            loss_dict["contrastive"] = contrastive_loss



        total_loss = 0
        for loss_key in self.losses_list:
            total_loss += self.losses_list[loss_key] * loss_dict.get(loss_key, 0)

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


    def compute_trajectory_loss(self, pred, target):
        rw = self.rotation_weight
        def split(t):
            return t[..., :3], t[..., 3:].reshape(*t.shape[:-1], 3, 3)
        p_pred, R_pred = split(pred)
        p_tgt, R_tgt = split(target)

        first_frame_loss = (
            mse_loss(p_pred[:, 0:1], p_tgt[:, 0:1])
            + rw * rotation_frobenius_loss(R_pred[:, 0:1], R_tgt[:, 0:1])
        )

        rel_pos = mse_loss(p_pred[:, 1:] - p_pred[:, 0:1], p_tgt[:, 1:] - p_tgt[:, 0:1])
        R_pred_rel = torch.matmul(R_pred[:, 0:1].transpose(-1, -2), R_pred[:, 1:])
        R_tgt_rel = torch.matmul(R_tgt[:, 0:1].transpose(-1, -2), R_tgt[:, 1:])
        relative_loss = rel_pos + rw * rotation_frobenius_loss(R_pred_rel, R_tgt_rel)

        spd_pos = mse_loss(p_pred[:, 1:] - p_pred[:, :-1], p_tgt[:, 1:] - p_tgt[:, :-1])
        R_pred_spd = torch.matmul(R_pred[:, :-1].transpose(-1, -2), R_pred[:, 1:])
        R_tgt_spd = torch.matmul(R_tgt[:, :-1].transpose(-1, -2), R_tgt[:, 1:])
        speed_loss = spd_pos + rw * rotation_frobenius_loss(R_pred_spd, R_tgt_spd)

        return first_frame_loss, relative_loss, speed_loss
