import os
import json
from typing import Dict, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.clip_embeddings import CLIPEmbedder
from data.et.dataset import ETDataset
from utils.calaculation3d import rotation_6d_to_matrix, euler_from_matrix


class ModelInference:
    def __init__(self, cfg: DictConfig):
        self.device = torch.device(
            cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")

        self.clip_embedder = CLIPEmbedder(
            model_name=cfg.clip.model_name,
            device=self.device
        )

        self.model = instantiate(cfg.training.model)
        checkpoint = torch.load(cfg.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def process_trajectory(
        self,
        trajectory: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        if padding_mask is None:
            padding_mask = torch.ones(trajectory.shape[0], dtype=torch.bool)

        valid_trajectory = trajectory[padding_mask]

        position = valid_trajectory[:, :3].cpu().numpy()
        rotation_6d = valid_trajectory[:, 3:9].cpu().numpy()

        rotation_matrices = rotation_6d_to_matrix(rotation_6d)
        euler_angles = np.array([euler_from_matrix(R)
                                for R in rotation_matrices])

        return {
            "position": position,
            "rotation_euler": euler_angles,
            "rotation_6d": rotation_6d
        }

    def generate_from_text(
        self,
        text: str,
        subject_trajectory: torch.Tensor,
        output_path: str
    ) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            caption_feat = self.clip_embedder.get_embeddings([text])

            caption_feat = caption_feat.to(self.device)
            subject_trajectory = subject_trajectory.to(self.device)

            output = self.model.generate(
                subject_trajectory=subject_trajectory,
                caption_embedding=caption_feat
            )

            trajectory_data = self.process_trajectory(
                output["generated_trajectory"][0],
                output.get("padding_mask", None)
            )

            self.save_results(
                trajectory_data,
                text,
                output_path,
                "text_to_trajectory"
            )

            return trajectory_data

    def reconstruct_trajectory(
        self,
        camera_trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor,
        output_path: str
    ) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            camera_trajectory = camera_trajectory.to(self.device)
            subject_trajectory = subject_trajectory.to(self.device)

            output = self.model(
                camera_trajectory=camera_trajectory,
                subject_trajectory=subject_trajectory
            )

            trajectory_data = self.process_trajectory(
                output["reconstructed"][0],
                output.get("padding_mask", None)
            )

            self.save_results(
                trajectory_data,
                None,
                output_path,
                "reconstruction"
            )

            return trajectory_data

    def save_results(
        self,
        trajectory_data: Dict[str, np.ndarray],
        text: Optional[str],
        output_path: str,
        prefix: str
    ):
        os.makedirs(output_path, exist_ok=True)
        timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")

        for key, data in trajectory_data.items():
            filename = f"{prefix}_{key}_{timestamp}.npy"
            filepath = os.path.join(output_path, filename)
            np.save(filepath, data)

        metadata = {
            "timestamp": timestamp,
            "type": prefix,
            "text": text,
            "files": [f"{prefix}_{key}_{timestamp}.npy" for key in trajectory_data.keys()]
        }
        metadata_path = os.path.join(
            output_path, f"{prefix}_metadata_{timestamp}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    inference = ModelInference(cfg=cfg)

    dataset = ETDataset(
        project_config_dir=cfg.project_config_dir,
        dataset_dir=cfg.dataset_dir,
        set_name=cfg.dataset.set_name,
        split=cfg.dataset.split
    )

    if cfg.sample_id:
        sample_indices = [
            dataset.original_dataset.root_filenames.index(cfg.sample_id)]
    else:
        sample_indices = range(len(dataset))

    for idx in sample_indices:
        sample = dataset[idx]
        sample_id = dataset.original_dataset.root_filenames[idx]

        subject_trajectory = sample['subject_trajectory'].unsqueeze(
            0)
        camera_trajectory = sample['camera_trajectory'].unsqueeze(
            0).transpose(1, 2)

        sample_output_dir = os.path.join(cfg.output_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)

        print(f"\nProcessing sample {sample_id}")
        print("Running trajectory reconstruction...")
        reconstructed_trajectory = inference.reconstruct_trajectory(
            camera_trajectory,
            subject_trajectory,
            output_path=sample_output_dir
        )

        if cfg.text_prompt:
            print("Generating trajectory from text...")
            text_prompt = cfg.text_prompt
            generated_trajectory = inference.generate_from_text(
                text_prompt,
                subject_trajectory,
                output_path=sample_output_dir
            )

        elif 'caption_raw' in sample and 'caption' in sample['caption_raw']:
            print("Generating trajectory from sample caption...")
            text_prompt = sample['caption_raw']['caption']
            generated_trajectory = inference.generate_from_text(
                text_prompt,
                subject_trajectory,
                output_path=sample_output_dir
            )

        print(f"Results saved to {sample_output_dir}")


if __name__ == "__main__":
    main()
