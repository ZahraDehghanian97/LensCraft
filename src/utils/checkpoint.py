import torch

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]
        else:
            new_key = key
        
        if 'subject_projection_loc_rot' in new_key:
            new_key = new_key.replace('subject_projection_loc_rot', 'subject_trajectory_projection')
        
        if 'subject_projection_vol' in new_key:
            new_key = new_key.replace('subject_projection_vol', 'subject_volume_projection')
        
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model