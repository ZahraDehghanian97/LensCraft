import json
import torch
from torch.utils.data import Dataset
from .constants import CameraMovementType, EasingType, CameraAngle, ShotType


class SimulationDataset(Dataset):
    def __init__(self, data_path: str, clip_embeddings: dict):
        self.clip_embeddings = clip_embeddings
        with open(data_path, 'r') as file:
            raw_data = json.load(file)
        self.simulation_data = [self._process_single_simulation(sim)
                                for sim in raw_data['simulations']
                                if self._is_simulation_valid(sim)]

    def __len__(self):
        return len(self.simulation_data)

    def __getitem__(self, index):
        original_item = self.simulation_data[index]
        # positions = original_item['camera_trajectory'][:, :3]
        # velocity = positions[1:] - positions[:-1]       
        # original_item['camera_trajectory'][1:, :3] = velocity
        # original_item['camera_trajectory'][0, :3] = positions[0]
        return original_item

    def _is_simulation_valid(self, simulation):
        return (len(simulation['instructions']) == 1 and
                simulation['instructions'][0]['frameCount'] == 30 and
                len(simulation['cameraFrames']) == 30)

    def _process_single_simulation(self, simulation):
        instruction = simulation['instructions'][0]
        subject = simulation['subjects'][0]

        camera_trajectory = self._extract_camera_frame_data(
            simulation['cameraFrames'])
        subject_trajectory = self._simulate_subject_trajectory(
            subject, len(camera_trajectory))

        movement_type = CameraMovementType[instruction['cameraMovement']]
        easing_type = EasingType[instruction['movementEasing']]
        camera_angle = CameraAngle[instruction.get(
            'initialCameraAngle', 'mediumAngle')]
        shot_type = ShotType[instruction.get('initialShotType', 'mediumShot')]

        return {
            'camera_trajectory': torch.tensor(camera_trajectory, dtype=torch.float32),
            'subject_trajectory': torch.tensor(subject_trajectory, dtype=torch.float32),
            'movement_clip': self.clip_embeddings['movement'][movement_type.value].to('cpu'),
            'easing_clip': self.clip_embeddings['easing'][easing_type.value].to('cpu'),
            'angle_clip': self.clip_embeddings['angle'][camera_angle.value].to('cpu'),
            'shot_clip': self.clip_embeddings['shot'][shot_type.value].to('cpu'),
            'instruction': instruction
        }

    @staticmethod
    def _extract_camera_frame_data(camera_frames):
        return [
            [
                frame['position']['x'],
                frame['position']['y'],
                frame['position']['z'],
                frame['focalLength'],
                frame['angle']['x'],
                frame['angle']['y'],
                frame['angle']['z']
            ]
            for frame in camera_frames
        ]

    @staticmethod
    def _simulate_subject_trajectory(subject, num_frames):
        subject_data = [
            subject['position']['x'], subject['position']['y'], subject['position']['z'],
            subject['size']['x'], subject['size']['y'], subject['size']['z'],
            subject['rotation']['x'], subject['rotation']['y'], subject['rotation']['z']
        ]
        return [subject_data for _ in range(num_frames)]


def batch_collate(batch):
    return {
        'camera_trajectory': torch.stack([item['camera_trajectory'] for item in batch]),
        'subject_trajectory': torch.stack([item['subject_trajectory'] for item in batch]),
        'movement_clip': torch.stack([item['movement_clip'] for item in batch]),
        'easing_clip': torch.stack([item['easing_clip'] for item in batch]),
        'angle_clip': torch.stack([item['angle_clip'] for item in batch]),
        'shot_clip': torch.stack([item['shot_clip'] for item in batch]),
        'instruction': [item['instruction'] for item in batch]
    }
