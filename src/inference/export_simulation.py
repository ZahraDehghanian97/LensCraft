from typing import Optional
import json
import os

def prepare_output_directory(output_dir: str, sample_id: Optional[str] = None) -> str:
    dir_path = os.path.join(output_dir, sample_id if sample_id else 'simulation_output')
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def generate_simulation_format(camera, subject_loc_rot=None, subject_vol=None, subject=None):
    if subject is not None and (subject_loc_rot is None or subject_vol is None):
        frames = [{
            "position": {
                "x": frame[0].item(),
                "y": frame[1].item(),
                "z": frame[2].item()
            },
            "rotation": {
                "x": frame[6].item(),
                "y": frame[7].item(),
                "z": frame[8].item()
            }
        } for frame in subject]
        
        dimensions = {
            "width": subject[0, 3].item(),
            "height": subject[0, 4].item(),
            "depth": subject[0, 5].item()
        }
    else:
        frames = [{
            "position": {
                "x": frame[0].item(),
                "y": frame[1].item(),
                "z": frame[2].item()
            },
            "rotation": {
                "x": frame[3].item(),
                "y": frame[4].item(),
                "z": frame[5].item()
            }
        } for frame in subject_loc_rot]
        
        if subject_vol.shape[1] == 1:
            dimensions = {
                "width": subject_vol[0, 0].item(),
                "height": subject_vol[1, 0].item(),
                "depth": subject_vol[2, 0].item()
            }
        else:
            dimensions = {
                "width": subject_vol[0, 0].item(),
                "height": subject_vol[0, 1].item(),
                "depth": subject_vol[0, 2].item()
            }

    res = {
        "subjects": [{
            "frames": frames,
            "dimensions": dimensions,
        }],
        "cameraFrames": [
            {
                "position": {
                    "x": camera[i, 0].item(),
                    "y": camera[i, 1].item(),
                    "z": camera[i, 2].item()
                },
                "angle": {
                    "x": camera[i, 3].item(),
                    "y": camera[i, 4].item(),
                    "z": camera[i, 5].item()
                },
                "focalLength": camera[i, 6].item(),
            }
            for i in range(camera.size(0))
        ],
        "instructions": [],
    }
    
    return res

def export_simulation(data, output_dir):
    output_path = os.path.join(output_dir, f'simulation-out.json')
    simulations = []
    
    for item in data:
        if 'subject_loc_rot' in item and 'subject_vol' in item:
            subject_loc_rot = item['subject_loc_rot']
            subject_vol = item['subject_vol']
            subject = None
        elif 'subject' in item:
            subject = item['subject']
            subject_loc_rot = None
            subject_vol = None
        else:
            raise ValueError("Cannot find subject data in expected format")
        
        rec_camera = item['rec']['reconstructed'] if isinstance(item['rec'], dict) else item['rec']
        prompt_gen_camera = item['prompt_gen']['reconstructed'] if isinstance(item['prompt_gen'], dict) else item['prompt_gen']
        hybrid_gen_camera = item['hybrid_gen']['reconstructed'] if isinstance(item['hybrid_gen'], dict) else item['hybrid_gen']
        
        print(
            item.get('simulation_instructions', None), 
            item.get('cinematography_prompts', None)
        )
        
        simulations += [
            generate_simulation_format(item['camera'], subject_loc_rot, subject_vol, subject),
            generate_simulation_format(rec_camera, subject_loc_rot, subject_vol, subject),
            generate_simulation_format(hybrid_gen_camera, subject_loc_rot, subject_vol, subject),
            generate_simulation_format(prompt_gen_camera, subject_loc_rot, subject_vol, subject)
        ]
        
    with open(output_path, 'w') as f:
        json.dump({"simulations": simulations}, f, indent=2)