import torch

def extract_position_of_subject_trajectory(char_feat: torch.Tensor) -> torch.Tensor:
    if char_feat.dim() == 3:
        char_feat = char_feat.squeeze(0)
    
    positions = char_feat[:, :3]
    return positions
