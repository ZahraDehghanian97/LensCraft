import torch
from data.convertor_utils import fix_camera_traj_length, et_to_6dof, fix_subject_traj_length



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
    subject_trajectory_resized = fix_subject_traj_length(subject_trajectory, target_frames=30)
    # print("+", subject_trajectory_resized.shape)

    subject_volume = torch.tensor([[0.5, 1.7, 0.3]])  # Default size values

    return subject_trajectory_resized, subject_volume



def camera_et_to_sim(camera_trajectory: torch.tensor, 
                     seq_len: int):
    is_batch = True
    # print("1", camera_trajectory.shape)

    if len(camera_trajectory.shape) == 3:
        camera_trajectory = camera_trajectory.unsqueeze(0)
        is_batch = False

    camera_trajectory_resized = fix_camera_traj_length(camera_trajectory, seq_len)
    # print("2", camera_trajectory_resized.shape)
    camera_trajectory_resized = et_to_6dof(camera_trajectory_resized)

    # print("3", camera_trajectory_resized.shape)
    
    if not is_batch:
        camera_trajectory_resized = camera_trajectory_resized.squeeze()
    return torch.tensor(camera_trajectory_resized)











def subject_ccdm_to_sim(subject_trajectory: None,
                      seq_len: int):
    subject_loc_rot = torch.zeros((seq_len, 6), dtype=torch.float32)
    subject_volume = torch.tensor([[0.5, 1.7, 0.3]], dtype=torch.float32)
    
    return subject_loc_rot, subject_volume



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










def subject_sim_to_et():
    pass


def camera_sim_to_et():
    pass











def subject_sim_to_ccdm():
    pass



def camera_sim_to_ccdm():
    pass






