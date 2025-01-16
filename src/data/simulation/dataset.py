import json
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List


class SimulationDataset(Dataset):
    def __init__(self, data_path: str, clip_embeddings: Dict[str, Any]):
        self.clip_embeddings = clip_embeddings
        
        with open(data_path, 'r') as file:
            raw_data = json.load(file)
        
        self.simulations = [
            sim for sim in raw_data
            if self._is_simulation_valid(sim)
        ]
        
        self.simulation_data = [
            self._process_single_simulation(sim_idx)
            for sim_idx in range(len(self.simulations))
        ]

    def __len__(self) -> int:
        return len(self.simulation_data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.simulation_data[index]

    def _is_simulation_valid(self, simulation: Dict[str, Any]) -> bool:
        instructions = simulation.get("simulationInstructions", [])
        frames = simulation.get("subjectFrames", [])
        
        if len(instructions) < 1:
            return False
        if instructions[0].get("frameCount", 0) != 30:
            return False
        
        
        if not frames or len(frames[0]) != 30:
            return False
        
        return True

    def _process_single_simulation(self, sim_idx: int) -> Dict[str, Any]:
        simulation = self.simulations[sim_idx]
        
        instruction = simulation["simulationInstructions"][0]
        subject_frames = simulation["subjectFrames"][0]
        
        subject_trajectory = self._extract_subject_trajectory(subject_frames)
        
        sim_emb_entry = self.clip_embeddings["simulations"][sim_idx]["simulation"][0]
        
        caption_feat = sim_emb_entry["init_setup"]  
        if caption_feat is None:
            
            caption_feat = torch.zeros((1, 768))  
        
        
        return {
            "subject_trajectory": torch.tensor(subject_trajectory, dtype=torch.float32),
            "caption_feat": caption_feat.cpu() if hasattr(caption_feat, "cpu") else caption_feat,
            "instruction": instruction,   
        }

    @staticmethod
    def _extract_subject_trajectory(frames: List[Dict[str, Any]]) -> List[List[float]]:
        trajectory = []
        for frame in frames:
            px = frame["position"]["x"]
            py = frame["position"]["y"]
            pz = frame["position"]["z"]
            rx = frame["rotation"]["x"]
            ry = frame["rotation"]["y"]
            rz = frame["rotation"]["z"]
            trajectory.append([px, py, pz, rx, ry, rz])
        return trajectory


def batch_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "subject_trajectory": torch.stack([item["subject_trajectory"] for item in batch]),
        "caption_feat": torch.stack([item["caption_feat"] for item in batch]),
        "instruction": [item["instruction"] for item in batch],
    }
