import numpy as np
import torch
import logging
from third_parties.DIRECTOR.visualization.common_viz import init, get_batch
from third_parties.DIRECTOR.utils.random_utils import set_random_seed
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

class ETAdapter:    
    def __init__(self, config_name: str, batch_size: int, num_frames: int, guidance_scale: float):
        self.diffuser, self.clip_model, self.dataset, self.device = init(config_name, batch_size)
        self.diffuser.to(self.device)
        self.clip_model.to(self.device)
        self.diffuser.guidance_weight = guidance_scale
        self.num_frames = num_frames
        set_random_seed(42)

    def add_subject_trajectory_to_et_batch(self, et_batch, batch):
        device = et_batch["char_feat"].device
        subject_positions = batch["subject_trajectory"][:, :, :3].permute(0, 2, 1)
        et_batch["char_feat"] = torch.zeros_like(et_batch["char_feat"], device=device)
        et_batch["char_feat"][:, :, :subject_positions.shape[2]] = subject_positions.to(device)

    def adjust_et_batch_padding(self, et_batch):
        padding_mask = torch.zeros_like(et_batch["padding_mask"])
        padding_mask[:, :self.num_frames] = 1
        
        for key in ["padding_mask", "char_padding_mask", "caption_padding_mask"]:
            et_batch[key] = padding_mask

    def resize_trajectory(self, trajectories, target_frames=30):
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

    def trajectory_to_6dof(self, trajectories):
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

    def transform_trajectories(self, trajectories, target_frames=30):
        resized = self.resize_trajectory(trajectories, target_frames)
        return torch.tensor(self.trajectory_to_6dof(resized), device=self.device)
  
    def process_batch(self, batch):
        text_prompts = batch["text_prompts"]
        self.diffuser.gen_seeds = np.arange(len(text_prompts))
        
        et_batch = get_batch(
            text_prompts, 
            '2011_F_EuMeT2wBo_00014_00001', 
            self.clip_model, 
            self.dataset, 
            self.diffuser.net.model.clip_sequential, 
            self.device
        )
        
        self.add_subject_trajectory_to_et_batch(et_batch, batch)
        self.adjust_et_batch_padding(et_batch)
        
        with torch.no_grad():
            out = self.diffuser.predict_step(et_batch, 0)
            camera_trajectory = self.transform_trajectories(
                out["gen_samples"][out["padding_mask"].to(bool)], 
                self.num_frames
            )
            
            return {
                "camera_trajectory": camera_trajectory,
                "subject_trajectory": batch["subject_trajectory"],
                "subject_volume": batch["subject_volume"],
                "padding_mask": None,
                "caption_feat": et_batch['caption_feat'],
                "raw_text": text_prompts,
                "original_camera_trajectory": batch["camera_trajectory"],
            }