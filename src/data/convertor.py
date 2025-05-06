import torch
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d






def subject_et_to_sim(subject_trajectory: torch.tensor, 
                      seq_len: int):
    
    char_positions = subject_trajectory[:3].transpose(0, 1)
    subject_trajectory = []

    for pos in char_positions:
        subject_frame = [
            pos[0].item(), pos[1].item(), pos[2].item(),
            0, 0, 0  # Default rotation values
        ]
        subject_trajectory.append(subject_frame)

    subject_trajectory = torch.tensor(subject_trajectory)

    subject_volume = torch.tensor([[0.5, 1.7, 0.3]])  # Default size values
    print("salam", subject_volume.dtype, subject_trajectory.dtype)

    return subject_trajectory, subject_volume



def subject_ccdm_to_sim(subject_trajectory: None,
                      seq_len: int):
    subject_loc_rot = torch.zeros((seq_len, 6), dtype=torch.float32)
    subject_volume = torch.tensor([[0.5, 1.7, 0.3]], dtype=torch.float32)
    
    return subject_loc_rot, subject_volume



def subject_sim_to_et():
    pass



def subject_sim_to_ccdm():
    pass








def camera_et_to_sim(camera_trajectory: torch.tensor, 
                     seq_len: int):
    is_batch = True

    if len(camera_trajectory.shape) == 3:
        camera_trajectory = camera_trajectory.unsqueeze(0)
        is_batch = False

    resized = fix_traj_length(camera_trajectory, seq_len)
    camera_trajectory_resized = et_to_6dof(resized)
    
    if not is_batch:
        camera_trajectory_resized = camera_trajectory_resized.squeeze()
    return torch.tensor(camera_trajectory_resized)



def camera_ccdm_to_sim(camera_trajectory: torch.tensor, 
                       seq_len: int, 
                       tan_half_fov_x, 
                       tan_half_fov_y) -> torch.tensor:
    traj_length = len(camera_trajectory)
    padding_mask = None
    
    if traj_length < seq_len:
        padding = camera_trajectory[-1:].repeat(seq_len - traj_length, 1)
        camera_trajectory = torch.cat([camera_trajectory, padding], dim=0)
        padding_mask = torch.cat([
            torch.ones(traj_length, dtype=torch.bool),
            torch.zeros(seq_len - traj_length, dtype=torch.bool)
        ])
    else:
        indices = torch.linspace(0, traj_length - 1, seq_len).long()
        camera_trajectory = camera_trajectory[indices]
            
    x, y, z = camera_trajectory[:,0], camera_trajectory[:,1], camera_trajectory[:,2]
    px, py = camera_trajectory[:,3], camera_trajectory[:,4]

    yaw_center_rad = torch.atan2(x, z)
    pitch_center_rad = torch.atan2(y, torch.hypot(x, z))

    delta_pitch_rad = -torch.atan(py * tan_half_fov_y)
    delta_yaw_rad = -torch.atan(px * tan_half_fov_x)

    yaw = torch.rad2deg(yaw_center_rad + delta_yaw_rad)
    pitch = torch.rad2deg(pitch_center_rad + delta_pitch_rad)
    roll = torch.zeros_like(yaw)
    # focal = torch.full_like(yaw, focal_length_mm)
    
    return torch.stack([x, y, z, yaw, pitch, roll], dim=1), padding_mask



def camera_sim_to_et():
    pass



def camera_sim_to_ccdm():
    pass





def fix_traj_length(trajectories, target_frames=30):
    """
    Upsample or downsample trajectories to specified number of frames.
    """
    N, T, _, _ = trajectories.shape
    resized_trajectories = np.zeros((N, target_frames, 4, 4))
    resized_trajectories[:, :, 3, 3] = 1  
    
    times = np.linspace(0, 1, T)
    target_times = np.linspace(0, 1, target_frames)
    
    for i in range(N):
        positions = trajectories[i, :, :3, 3]
        rotations = R.from_matrix(trajectories[i, :, :3, :3])
        
        resized_trajectories[i, :, :3, 3] = interp1d(times, positions, axis=0, 
                                                    kind='linear', fill_value='extrapolate')(target_times)
        resized_trajectories[i, :, :3, :3] = Slerp(times, rotations)(target_times).as_matrix()
    
    return resized_trajectories


def et_to_6dof(trajectories):
    """
    Convert camera trajectories to 6DoF: position (3), Euler angles (3)
    """
    N, T, _, _ = trajectories.shape
    result = np.zeros((N, T, 6))
    result[:, :, :3] = trajectories[:, :, :3, 3]
    for i in range(N):
        for t in range(T):
            rotation_matrix = trajectories[i, t, :3, :3]
            result[i, t, 3:6] = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
    
    return result


