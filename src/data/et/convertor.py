import torch
import torch.nn.functional as F
from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

class ETConvertor(BaseConvertor):
    def __init__(self, standardize: bool, mean_std):
        self.augmentation = None
        self.standardize = standardize
        if self.standardize:
            self.norm_mean = torch.Tensor(mean_std["norm_mean"])
            self.norm_std = torch.Tensor(mean_std["norm_std"])
            self.shift_mean = torch.Tensor(mean_std["shift_mean"])
            self.shift_std = torch.Tensor(mean_std["shift_std"])
            self.velocity = mean_std["velocity"]
    
    @handle_single_or_batch(arg_index=[1])
    def get_feature(self, raw_matrix_trajectory):
        matrix_trajectory = torch.clone(raw_matrix_trajectory)

        raw_trans = torch.clone(matrix_trajectory[:, :3, 3])
        if self.velocity:
            velocity = raw_trans[1:] - raw_trans[:-1]
            raw_trans = torch.cat([raw_trans[0][None], velocity])
        if self.standardize:
            raw_trans[0] -= self.shift_mean
            raw_trans[0] /= self.shift_std
            raw_trans[1:] -= self.norm_mean
            raw_trans[1:] /= self.norm_std

        # Compute the 6D continuous rotation
        raw_rot = matrix_trajectory[:, :3, :3]
        rot6d = raw_rot[:, :, :2].permute(0, 2, 1).reshape(-1, 6)

        # Stack rotation 6D and translation
        rot6d_trajectory = torch.hstack([rot6d, raw_trans]).permute(1, 0)

        return rot6d_trajectory

    @handle_single_or_batch(arg_index=[1])
    def get_matrix(self, raw_rot6d_trajectory):
        rot6d_trajectory = torch.clone(raw_rot6d_trajectory)
        device = rot6d_trajectory.device

        num_cams = rot6d_trajectory.shape[1]
        matrix_trajectory = torch.eye(4, device=device)[None].repeat(num_cams, 1, 1)

        raw_trans = rot6d_trajectory[6:].permute(1, 0)
        if self.standardize:
            raw_trans[0] *= self.shift_std.to(device)
            raw_trans[0] += self.shift_mean.to(device)
            raw_trans[1:] *= self.norm_std.to(device)
            raw_trans[1:] += self.norm_mean.to(device)
        if self.velocity:
            raw_trans = torch.cumsum(raw_trans, dim=0)
        matrix_trajectory[:, :3, 3] = raw_trans

        rot6d = rot6d_trajectory[:6].permute(1, 0)
        raw_rot = compute_rotation_matrix_from_ortho6d(rot6d)
        matrix_trajectory[:, :3, :3] = raw_rot

        return matrix_trajectory

    @handle_single_or_batch(arg_index=[1, 2])
    def to_standard(
        self,
        trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        subject_positions = subject_trajectory[..., :3]
        subject_trajectory = torch.zeros(trajectory.shape[:2] + (6), dtype=torch.float32)
        subject_trajectory[..., :3] = subject_positions
        subject_volume = torch.tensor([[0.5, 1.7, 0.3]], dtype=torch.float32)

        transform = self.get_matrix(trajectory)
        return transform, subject_trajectory, subject_volume

    @handle_single_or_batch(arg_index=[1, 2])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        subject_trajectory = (
            subject_trajectory[..., :3] if subject_trajectory is not None else None
        )
        trajectory = self.get_feature(transform)
        subject_volume = None
        return trajectory, subject_trajectory, subject_volume
