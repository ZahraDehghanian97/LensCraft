import torch
def subject_et_to_sim():
    pass


def subject_ccdm_to_sim(subject_trajectory: None,
                      seq_len: int):
    subject_loc_rot = torch.zeros((seq_len, 6), dtype=torch.float32)
    subject_volume = torch.tensor([[0.5, 1.7, 0.3]], dtype=torch.float32)
    
    return subject_loc_rot, subject_volume



def subject_sim_to_et():
    pass



def subject_sim_to_ccdm():
    pass








def camera_et_to_sim():
    pass



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


