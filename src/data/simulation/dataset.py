import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List

class SimulationDataset(Dataset):
    def __init__(self, data_path: str, clip_embeddings: Dict):
        self.clip_embeddings = clip_embeddings
        
        with open(data_path, 'r') as file:
            self.raw_data_list = json.load(file)
            
        self.embedding_dim = 512

    def __len__(self):
        return len(self.raw_data_list)

    def __getitem__(self, index):
        return self._process_single_simulation(self.raw_data_list[index], index)

    def _process_single_simulation(self, simulation_data: Dict, index: int) -> Dict:
        camera_trajectory = self._extract_camera_trajectory(simulation_data['cameraFrames'])
        subject_trajectory = self._extract_subject_trajectory(simulation_data['subjectsInfo'])
        
        def get_embedding(category: str, key: str) -> torch.Tensor:
            embedding = self.clip_embeddings[index][category][key]
            return embedding if embedding is not None else torch.zeros(self.embedding_dim)

        return {
            'camera_trajectory': torch.tensor(camera_trajectory, dtype=torch.float32),
            'subject_trajectory': torch.tensor(subject_trajectory, dtype=torch.float32),
            'simulations_init_setup_embedding': get_embedding('simulation', 'init_setup'),
            'simulations_movement_embedding': get_embedding('simulation', 'movement'),
            'simulations_end_setup_embedding': get_embedding('simulation', 'end_setup'),
            'simulations_constraints_embedding': get_embedding('simulation', 'constraints'),
            'cinematography_init_setup_embedding': get_embedding('cinematography', 'init_setup'),
            'cinematography_movement_embedding': get_embedding('cinematography', 'movement'),
            'cinematography_end_setup_embedding': get_embedding('cinematography', 'end_setup'),
            'simulation_instructions': simulation_data['simulationInstructions'],
            'cinematography_prompts': simulation_data['cinematographyPrompts'],
            'embedding_masks': {
                'simulation': {
                    key: self.clip_embeddings[index]['simulation'][key] is not None
                    for key in ['init_setup', 'movement', 'end_setup', 'constraints']
                },
                'cinematography': {
                    key: self.clip_embeddings[index]['cinematography'][key] is not None
                    for key in ['init_setup', 'movement', 'end_setup']
                }
            }
        }

    def _extract_camera_trajectory(self, camera_frames: List[Dict]) -> List[List[float]]:
        return [
            [
                frame['position']['x'],
                frame['position']['y'],
                frame['position']['z'],
                frame['rotation']['x'],
                frame['rotation']['y'],
                frame['rotation']['z'],
                frame['focalLength'],
                frame['aspectRatio']
            ]
            for frame in camera_frames
        ]

    def _extract_subject_trajectory(self, subjects_info: List[Dict]) -> List[List[float]]:
        subject = subjects_info[0]
        frames = subject.get('frames', [])
        if not frames:
            static_frame = {
                'position': subject['subject'].get('attentionBox', {}).get('position', {'x': 0, 'y': 0, 'z': 0}),
                'rotation': {'x': 0, 'y': 0, 'z': 0}
            }
            frames = [static_frame] * len(self.camera_frames)

        return [
            [
                frame['position']['x'],
                frame['position']['y'],
                frame['position']['z'],
                frame['rotation']['x'],
                frame['rotation']['y'],
                frame['rotation']['z']
            ]
            for frame in frames
        ]

def collate_fn(batch):
    return {
        'camera_trajectory': torch.stack([item['camera_trajectory'] for item in batch]),
        'subject_trajectory': torch.stack([item['subject_trajectory'] for item in batch]),
        'simulations_init_setup_embedding': torch.stack([item['simulations_init_setup_embedding'] for item in batch]),
        'simulations_movement_embedding': torch.stack([item['simulations_movement_embedding'] for item in batch]),
        'simulations_end_setup_embedding': torch.stack([item['simulations_end_setup_embedding'] for item in batch]),
        'simulations_constraints_embedding': torch.stack([item['simulations_constraints_embedding'] for item in batch]),
        'cinematography_init_setup_embedding': torch.stack([item['cinematography_init_setup_embedding'] for item in batch]),
        'cinematography_movement_embedding': torch.stack([item['cinematography_movement_embedding'] for item in batch]),
        'cinematography_end_setup_embedding': torch.stack([item['cinematography_end_setup_embedding'] for item in batch]),
        'simulation_instructions': [item['simulation_instructions'] for item in batch],
        'cinematography_prompts': [item['cinematography_prompts'] for item in batch],
        'embedding_masks': {
            'simulation': {
                key: torch.tensor([item['embedding_masks']['simulation'][key] for item in batch])
                for key in ['init_setup', 'movement', 'end_setup', 'constraints']
            },
            'cinematography': {
                key: torch.tensor([item['embedding_masks']['cinematography'][key] for item in batch])
                for key in ['init_setup', 'movement', 'end_setup']
            }
        }
    }