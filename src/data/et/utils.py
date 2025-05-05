from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d

from third_parties.DIRECTOR.visualization.common_viz import encode_text


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

def et_to_sim_cam_traj(trajectories, target_frames=30):
    resized = fix_traj_length(trajectories, target_frames)
    return et_to_6dof(resized)


def sim_to_et_subject_traj(subject_trajectory, device, seq_len=300):
    subject_positions = subject_trajectory[:, :, :3].permute(0, 2, 1)
    char_feat = torch.zeros([3, seq_len], device=device)
    char_feat[:, :, :subject_positions.shape[2]] = subject_positions.to(device)
    return char_feat


def et_to_sim_subject_traj(subject_trajectory, device):
    pass


def get_caption_feat(
    prompts: List[str],
    clip_model,
    seq_feat: bool,
    device: torch.device,
) -> Dict[str, Any]:
    caption_seq_list, caption_tokens = encode_text(prompts, clip_model, None, device)

    if seq_feat:
        padded_seqs = []
        for seq in caption_seq_list:
            padded_seq = F.pad(seq, (0, 0, 0, 77 - seq.shape[0]))
            padded_seqs.append(padded_seq)
        caption_feat = torch.stack(padded_seqs, dim=0)
        caption_feat = caption_feat.permute(0, 2, 1)
    else:
        caption_feat = caption_tokens

    return caption_feat
