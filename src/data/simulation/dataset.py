from typing import Dict, List
from torch.utils.data import Dataset
import torch
import json
from .constants import cinematography_struct, cinematography_struct_size, simulation_struct, simulation_struct_size
from .utils import get_parameters


class SimulationDataset(Dataset):
    def __init__(self, data_path: str, clip_embeddings: Dict):
        self.clip_embeddings = clip_embeddings

        with open(data_path, 'r') as file:
            self.raw_data_list = json.load(file)

        self.embedding_dim = 512

    def __len__(self) -> int:
        return len(self.raw_data_list)

    def __getitem__(self, index: int) -> Dict:
        return self._process_single_simulation(self.raw_data_list[index], index)

    def _process_single_simulation(self, simulation_data: Dict, index: int) -> Dict:
        camera_trajectory = self._extract_camera_trajectory(
            simulation_data['cam'])
        subject_trajectory = self._extract_subject_trajectory(simulation_data)

        instruction = self._expand_simulation_instruction(
            simulation_data['c'][0])
        prompt = self._expand_cinematography_prompt(simulation_data['c'][0])

        simulation_instruction = get_parameters(
            data=instruction,
            struct=simulation_struct,
            clip_embeddings=self.clip_embeddings
        )
        cinematography_prompt = get_parameters(
            data=prompt,
            struct=cinematography_struct,
            clip_embeddings=self.clip_embeddings
        )

        return {
            'camera_trajectory': torch.tensor(camera_trajectory, dtype=torch.float32),
            'subject_trajectory': torch.tensor(subject_trajectory, dtype=torch.float32),
            'simulation_instruction_parameters': simulation_instruction,
            'cinematography_prompt_parameters': cinematography_prompt
        }

    def _extract_camera_trajectory(self, camera_frames: List[Dict]) -> List[List[float]]:
        return [
            [*frame['p'], *frame['r'], frame['f']]
            for frame in camera_frames
        ]

    def _extract_subject_trajectory(self, simulation_data: Dict) -> List[List[float]]:
        subject = simulation_data['s'][0]
        subject_frames = simulation_data['f'][0]

        return [
            [*frame['p'], *subject['d'], *frame['r']]
            for frame in subject_frames
        ]

    def _expand_simulation_instruction(self, compact_data: Dict) -> Dict:
        return {
            'initialSetup': {
                'cameraAngle': compact_data['i']['a'],
                'shotSize': compact_data['i']['s'],
                'subjectView': compact_data['i']['v'],
                'subjectFraming': compact_data['i']['f']
            },
            'dynamic': {
                'type': 'interpolation',
                'easing': compact_data['m']['s'],
                'endSetup': compact_data.get('f', {})
            }
        }

    def _expand_cinematography_prompt(self, compact_data: Dict) -> Dict:
        return {
            'initial': {
                'cameraAngle': compact_data['i']['a'],
                'shotSize': compact_data['i']['s'],
                'subjectView': compact_data['i']['v'],
                'subjectFraming': compact_data['i']['f']
            },
            'movement': {
                'type': compact_data['m']['t'],
                'speed': compact_data['m']['s']
            },
            'final': compact_data.get('f', {})
        }


def collate_fn(batch):
    batch_size = len(batch)
    simulation_instruction_tensor = torch.full(
        (simulation_struct_size, batch_size, 512),
        -1,
        dtype=torch.float
    )
    cinematography_prompt_tensor = torch.full(
        (cinematography_struct_size, batch_size, 512),
        -1,
        dtype=torch.float
    )

    for batch_idx, item in enumerate(batch):
        for param_idx, (_, _, _, embedding) in enumerate(item['simulation_instruction_parameters']):
            if embedding is not None:
                simulation_instruction_tensor[param_idx, batch_idx] = embedding

        for param_idx, (_, _, _, embedding) in enumerate(item['cinematography_prompt_parameters']):
            if embedding is not None:
                cinematography_prompt_tensor[param_idx, batch_idx] = embedding

    return {
        'camera_trajectory': torch.stack([item['camera_trajectory'] for item in batch]),
        'subject_trajectory': torch.stack([item['subject_trajectory'] for item in batch]),
        'simulation_instruction': simulation_instruction_tensor,
        'cinematography_prompt': cinematography_prompt_tensor,
        'simulation_instruction_parameters': [item['simulation_instruction_parameters'] for item in batch],
        'cinematography_prompt_parameters': [item['cinematography_prompt_parameters'] for item in batch]
    }
