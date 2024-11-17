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


def export_kitti_poses(positions: np.ndarray, rotations_6d: np.ndarray, output_path: str):
    rotation_matrices = rotation_6d_to_matrix(rotations_6d)

    with open(output_path, 'w') as f:
        for pos, rot_mat in zip(positions, rotation_matrices):
            pose = np.eye(4)
            pose[:3, :3] = rot_mat
            pose[:3, 3] = pos

            line = ' '.join(map(str, pose[:3].flatten()))
            f.write(line + '\n')


def export_character_positions(positions: np.ndarray, output_path: str):
    np.save(output_path, positions)


def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model


class ModelInference:
    def __init__(self, cfg: DictConfig):
        self.device = torch.device(
            cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")

        self.clip_embedder = CLIPEmbedder(
            model_name=cfg.clip.model_name,
            device=self.device
        )

        self.model = instantiate(cfg.training.model)
        self.model = load_checkpoint(
            cfg.checkpoint_path, self.model, self.device)
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

        return {
            "position": position,
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

            # Export trajectory data in KITTI format and character positions
            timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
            poses_file = os.path.join(
                output_path, f"text_to_trajectory_poses_{timestamp}.txt")
            char_file = os.path.join(
                output_path, f"text_to_trajectory_char_{timestamp}.npy")

            export_kitti_poses(
                trajectory_data["position"], trajectory_data["rotation_6d"], poses_file)
            export_character_positions(
                subject_trajectory.cpu().numpy(), char_file)

            metadata = {
                "timestamp": timestamp,
                "type": "text_to_trajectory",
                "text": text,
                "poses_file": poses_file,
                "char_file": char_file
            }
            metadata_path = os.path.join(
                output_path, f"text_to_trajectory_metadata_{timestamp}.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return trajectory_data, poses_file, char_file

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

            # Export trajectory data in KITTI format and character positions
            timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
            poses_file = os.path.join(
                output_path, f"reconstruction_poses_{timestamp}.txt")
            char_file = os.path.join(
                output_path, f"reconstruction_char_{timestamp}.npy")

            export_kitti_poses(
                trajectory_data["position"], trajectory_data["rotation_6d"], poses_file)
            export_character_positions(
                subject_trajectory.cpu().numpy(), char_file)

            metadata = {
                "timestamp": timestamp,
                "type": "reconstruction",
                "poses_file": poses_file,
                "char_file": char_file
            }
            metadata_path = os.path.join(
                output_path, f"reconstruction_metadata_{timestamp}.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return trajectory_data, poses_file, char_file


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

        subject_trajectory = sample['subject_trajectory'].unsqueeze(0)
        camera_trajectory = sample['camera_trajectory'].unsqueeze(
            0).transpose(1, 2)

        sample_output_dir = os.path.join(cfg.output_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)

        print(f"\nProcessing sample {sample_id}")
        print("Running trajectory reconstruction...")
        reconstructed_trajectory, rec_poses_file, rec_char_file = inference.reconstruct_trajectory(
            camera_trajectory,
            subject_trajectory,
            output_path=sample_output_dir
        )

        if cfg.text_prompt:
            print("Generating trajectory from text...")
            text_prompt = cfg.text_prompt
            generated_trajectory, gen_poses_file, gen_char_file = inference.generate_from_text(
                text_prompt,
                subject_trajectory,
                output_path=sample_output_dir
            )

        elif 'caption_raw' in sample and 'caption' in sample['caption_raw']:
            print("Generating trajectory from sample caption...")
            text_prompt = sample['caption_raw']['caption']
            generated_trajectory, gen_poses_file, gen_char_file = inference.generate_from_text(
                text_prompt,
                subject_trajectory,
                output_path=sample_output_dir
            )

        print(f"Results saved to {sample_output_dir}")


if __name__ == "__main__":
    main()
