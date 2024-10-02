import numpy as np
import torch
from pathlib import Path
from evo.tools.file_interface import read_kitti_poses_file


def read_single_video_data(video_id, data_dir):
    data_dir = Path(data_dir)

    traj_path = data_dir / "traj" / f"{video_id}.txt"
    trajectory = read_kitti_poses_file(traj_path)
    matrix_trajectory = torch.from_numpy(
        np.array(trajectory.poses_se3)).float()

    char_path = data_dir / "char" / f"{video_id}.npy"
    char_data = torch.from_numpy(np.load(char_path)).float()

    caption_path = data_dir / "caption" / f"{video_id}.txt"
    with open(caption_path, 'r') as f:
        caption = f.read().strip()

    intrinsics_path = data_dir / "intrinsics" / f"{video_id}.npy"
    intrinsics = np.load(intrinsics_path)

    char_segments_path = data_dir / "char_segments" / f"{video_id}.npy"
    char_segments = None
    if char_segments_path.exists():
        char_segments = torch.from_numpy(np.load(char_segments_path))

    return {
        "video_id": video_id,
        "trajectory": matrix_trajectory,
        "char_data": char_data,
        "caption": caption,
        "intrinsics": intrinsics,
        "char_segments": char_segments
    }


data_dir = "./et-data"
video_id = "2011_--RFXQ-Xlac_00001_00000"

video_data = read_single_video_data(video_id, data_dir)
print(f"Trajectory shape: {video_data['trajectory'].shape}")
print(f"Character data shape: {video_data['char_data'].shape}")
print(f"Caption: {video_data['caption']}")
print(f"Intrinsics shape: {video_data['intrinsics'].shape}")
if video_data['char_segments'] is not None:
    print(f"Character segments shape: {video_data['char_segments'].shape}")
