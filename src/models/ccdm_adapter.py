import os
import sys
import numpy as np
import torch
import logging

from models.clip_embeddings import CLIPEmbedder

logger = logging.getLogger(__name__)

class CCDMAdapter:    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.n_T = 1000
        self.n_feature = 5
        self.n_textemb = 512
        
        self.ddpm, self.clip_embedder, self.mean, self.std = self._load_models()
        
    def _load_models(self):
        current_sys_path = sys.path.copy()
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "third_parties/Camera-control/[2024][EG]Text+keyframe"))
        from main import Transformer, DDPM
        sys.path = current_sys_path
        
        transformer = Transformer(n_feature=self.n_feature, n_textemb=self.n_textemb)
        ddpm = DDPM(nn_model=transformer, betas=(1e-4, 0.02), n_T=self.n_T, device=self.device)
        ddpm.to(self.device)
        
        checkpoint_path = self.config.ccdm_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"CCDM checkpoint not found: {checkpoint_path}")
        
        ddpm.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        ddpm.eval()
        
        clip_embedder = CLIPEmbedder(
            model_name=getattr(self.config, "clip_model_name", "openai/clip-vit-base-patch32"),
            device=self.device,
            chunk_size=getattr(self.config, "chunk_size", 100)
        )
        
        mean_std_path = os.path.join(os.path.dirname(checkpoint_path), "Mean_Std.npy")
        mean_std_data = np.load(mean_std_path, allow_pickle=True)[()]
        mean, std = mean_std_data["Mean"], mean_std_data["Std"]
        
        return ddpm, clip_embedder, mean, std
    
    def _smooth_trajectory(self, trajectory, window_size=10, iterations=4):
        smoothed = np.array(trajectory)
        for _ in range(iterations):
            for i in range(len(smoothed)):
                start = max(0, i - window_size)
                end = min(len(smoothed), i + window_size)
                smoothed[i] = np.mean(smoothed[start:end], axis=0)
        return smoothed
    
    def extract_text_prompt(self, cin_params):
        if not cin_params:
            return ""
        
        prompt_parts = []
        
        if "initial" in cin_params:
            initial = cin_params["initial"]
            
            if "cameraAngle" in initial:
                angle = initial["cameraAngle"]
                if angle == "low":
                    prompt_parts.append("The camera shoots at low angle.")
                elif angle == "high":
                    prompt_parts.append("The camera shoots at high angle.")
                elif angle == "eye":
                    prompt_parts.append("The camera shoots at eye level.")
                elif angle == "overhead":
                    prompt_parts.append("The camera shoots at overhead angle.")
                elif angle == "birdsEye":
                    prompt_parts.append("The camera shoots at bird's eye angle.")
            
            if "shotSize" in initial:
                shot_size = initial["shotSize"]
                prompt_parts.append(f"The camera shoots in {shot_size} shot.")
            
            if "subjectView" in initial:
                view = initial["subjectView"]
                if "threeQuarter" in view:
                    parts = view.replace("threeQuarter", "").split("_")
                    view_desc = f"{parts[0]} {parts[1]}" if len(parts) > 1 else view
                    prompt_parts.append(f"The camera shoots in {view_desc} view.")
                else:
                    prompt_parts.append(f"The camera shoots in {view} view.")
            
            if "subjectFraming" in initial:
                framing = initial["subjectFraming"]
                prompt_parts.append(f"The character is at the {framing} of the screen.")
        
        if "movement" in cin_params and "type" in cin_params["movement"]:
            movement_type = cin_params["movement"]["type"]
            
            if movement_type == "static":
                prompt_parts.append("The camera is static.")
            elif movement_type == "panLeft":
                prompt_parts.append("The camera pans to the left.")
            elif movement_type == "panRight":
                prompt_parts.append("The camera pans to the right.")
            elif movement_type == "tiltUp":
                prompt_parts.append("The camera tilts up.")
            elif movement_type == "tiltDown":
                prompt_parts.append("The camera tilts down.")
            elif movement_type == "dollyIn":
                prompt_parts.append("The camera pushes in to the character.")
            elif movement_type == "dollyOut":
                prompt_parts.append("The camera pulls out from the character.")
            elif movement_type == "arcLeft":
                prompt_parts.append("The camera rotates around the character to the left.")
            elif movement_type == "arcRight":
                prompt_parts.append("The camera rotates around the character to the right.")
            elif movement_type == "follow":
                prompt_parts.append("The camera follows the character.")
            elif movement_type == "track":
                prompt_parts.append("The camera tracks the character.")
            else:
                prompt_parts.append(f"The camera {movement_type}.")
                
            if "speed" in cin_params["movement"]:
                speed = cin_params["movement"]["speed"]
                if speed == "slowToFast":
                    prompt_parts.append("The camera accelerates gradually.")
                elif speed == "fastToSlow":
                    prompt_parts.append("The camera decelerates gradually.")
                elif speed == "smoothStartStop":
                    prompt_parts.append("The camera moves with smooth start and stop.")
        
        if "final" in cin_params and cin_params["final"]:
            final = cin_params["final"]
            initial = cin_params.get("initial", {})
            
            if "shotSize" in final and ("shotSize" not in initial or final["shotSize"] != initial.get("shotSize")):
                prompt_parts.append(f"The camera moves from {initial.get('shotSize', 'current')} shot to {final['shotSize']} shot.")
            
            if "subjectView" in final and ("subjectView" not in initial or final["subjectView"] != initial.get("subjectView")):
                init_view = initial.get("subjectView", "current")
                final_view = final["subjectView"]
                prompt_parts.append(f"The camera switches from {init_view} view to {final_view} view.")
        
        return " ".join(prompt_parts)
    
    def process_batch(self, batch):
        text_prompts = [
            self.extract_text_prompt(params) 
            for params in batch["cinematography_prompt_parameters"]
        ]
        
        if len(text_prompts) == 0:
            logger.warning("Empty batch, skipping")
            return None
        
        with torch.no_grad():
            text_embeddings = self.clip_embedder.extract_clip_embeddings(text_prompts, return_seq=False).to(self.device)
            
            guide_w = float(self.config.get("guidance_scale", 2.0))
            generated = self.ddpm.sample(
                n_sample=len(text_prompts),
                c=text_embeddings,
                size=(300, 5),
                device=self.device,
                guide_w=guide_w
            )
            
            generated_np = generated.detach().cpu().numpy()
            
            output_feature_dim = batch["camera_trajectory"].shape[-1]
            processed_trajectories = self._process_trajectories(generated_np, output_feature_dim)
            
            seq_length = batch["camera_trajectory"].shape[1]
            adjusted_trajectories = self._adjust_sequence_length(processed_trajectories, seq_length)
            
            trajectories_tensor = torch.tensor(
                np.array(adjusted_trajectories), 
                dtype=torch.float32, 
                device=self.device
            )
            
            return {
                'reconstructed': trajectories_tensor,
                'text_embeddings': text_embeddings,
                'text_prompts': text_prompts
            }
    
    def _process_trajectories(self, trajectories, output_feature_dim):
        processed_trajectories = []
        
        for traj in trajectories:
            denorm_traj = traj * self.std[None, :] + self.mean[None, :]
            
            smoothed_traj = self._smooth_trajectory(denorm_traj)
            
            camera_traj = np.zeros((len(smoothed_traj), output_feature_dim))
            camera_traj[:, :3] = smoothed_traj[:, :3]
            camera_traj[:, 3:5] = smoothed_traj[:, 3:5]
            
            if output_feature_dim > 5:
                camera_traj[:, 5] = 0.0
            if output_feature_dim > 6:
                camera_traj[:, 6] = 37.52
            
            processed_trajectories.append(camera_traj)
        
        return processed_trajectories
    
    def _adjust_sequence_length(self, trajectories, target_length):
        adjusted_trajectories = []
        for traj in trajectories:
            if len(traj) > target_length:
                adjusted_traj = traj[:target_length]
            elif len(traj) < target_length:
                padding = np.tile(traj[-1], (target_length - len(traj), 1))
                adjusted_traj = np.vstack([traj, padding])
            else:
                adjusted_traj = traj
            adjusted_trajectories.append(adjusted_traj)
        
        return adjusted_trajectories
