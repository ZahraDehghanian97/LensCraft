import numpy as np
import torch
import torch.nn.functional as F

from .config import STANDARDIZATION_CONFIG

class TrajectoryConverter:
    def __init__(self):
        self.norm_mean = torch.tensor(STANDARDIZATION_CONFIG["norm_mean"])
        self.norm_std = torch.tensor(STANDARDIZATION_CONFIG["norm_std"])
        self.shift_mean = torch.tensor(STANDARDIZATION_CONFIG["shift_mean"])
        self.shift_std = torch.tensor(STANDARDIZATION_CONFIG["shift_std"])
        self.norm_mean_h = torch.tensor(STANDARDIZATION_CONFIG["norm_mean_h"])
        self.norm_std_h = torch.tensor(STANDARDIZATION_CONFIG["norm_std_h"])
        self.velocity = STANDARDIZATION_CONFIG["velocity"]

    def convert_and_save_outputs(
        self, 
        output: torch.Tensor,
        output_path: str,
        is_camera: bool = True
    ):        
        if is_camera:
            traj_matrices = self._convert_camera_trajectory(output)
            self._save_kitti_trajectory(traj_matrices, output_path)
        else:
            char_positions = self._convert_character_trajectory(output)
            np.save(output_path, char_positions)

    def _convert_camera_trajectory(self, rot6d_trajectory: torch.Tensor) -> torch.Tensor:
        device = rot6d_trajectory.device
        num_cams = rot6d_trajectory.shape[0]
        
        matrices = torch.eye(4, device=device).unsqueeze(0).repeat(num_cams, 1, 1)
        
        translation = rot6d_trajectory[:, 6:]
        if self.velocity:
            translation[0] = translation[0] * self.shift_std.to(device) + self.shift_mean.to(device)
            translation[1:] = translation[1:] * self.norm_std.to(device) + self.norm_mean.to(device)
            translation = torch.cumsum(translation, dim=0)
        matrices[:, :3, 3] = translation

        rot6d = rot6d_trajectory[:, :6].reshape(-1, 3, 2)
        x_raw = rot6d[:, :, 0]
        y_raw = rot6d[:, :, 1]
        
        x = F.normalize(x_raw, dim=1)
        z = torch.cross(x, y_raw, dim=1)
        z = F.normalize(z, dim=1)
        y = torch.cross(z, x, dim=1)
        
        rotation = torch.stack([x, y, z], dim=2)
        matrices[:, :3, :3] = rotation

        return matrices

    def _convert_character_trajectory(self, char_trajectory: torch.Tensor) -> np.ndarray:
        positions = char_trajectory.clone()
        
        if self.velocity:
            positions[0] = positions[0] * self.norm_std_h[:3].to(positions.device) + self.norm_mean_h[:3].to(positions.device)
            positions[1:] = positions[1:] * self.norm_std_h.to(positions.device) + self.norm_mean_h.to(positions.device)
            positions = torch.cumsum(positions, dim=0)
            
        return positions.cpu().numpy()

    def _save_kitti_trajectory(self, matrices: torch.Tensor, output_file: str):
        with open(output_file, 'w') as f:
            for matrix in matrices:
                matrix_np = matrix[:3, :].cpu().numpy().flatten()
                line = ' '.join(map('{:.6e}'.format, matrix_np))
                f.write(line + '\n')
