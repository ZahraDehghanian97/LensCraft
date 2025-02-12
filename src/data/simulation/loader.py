from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import msgpack

def unround_floats(obj, factor=1000.0):
    if isinstance(obj, (int, np.integer)):
        return float(obj) / factor
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, list):
        return [unround_floats(x, factor) for x in obj]
    elif isinstance(obj, dict):
        return {k: unround_floats(v, factor) for k, v in obj.items()}
    else:
        return obj

def reconstruct_from_reference(refs: List[List[int]], dictionary: Dict) -> List[Dict]:    
    obj = {}
    for key_idx, val_idx in refs:
        if key_idx >= len(dictionary['keys']):
            continue
            
        path = dictionary['keys'][key_idx]
        value = dictionary['values'][key_idx][val_idx]
        
        current = obj
        parts = path.split('__')
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
        
    return obj

def _expand_subjects_info(compressed_subjects: List[Dict]) -> List[Dict]:
    expanded = []
    for subject in compressed_subjects:
        frames = []
        for frame in subject['f']:
            frames.append({
                "position": {"x": frame[0], "y": frame[1], "z": frame[2]},
                "rotation": {"x": frame[3], "y": frame[4], "z": frame[5]}
            })
        
        subject_info = {
            "subject": {
                "id": subject['i'],
                "class": subject['c'],
                "dimensions": {
                    "width": subject['d'][0],
                    "height": subject['d'][1],
                    "depth": subject['d'][2]
                }
            },
            "frames": frames
        }
        
        if 'a' in subject:
            subject_info["subject"]["attentionBox"] = {
                "position": {
                    "x": subject['a'][0],
                    "y": subject['a'][1],
                    "z": subject['a'][2]
                },
                "dimensions": {
                    "width": subject['a'][3],
                    "height": subject['a'][4],
                    "depth": subject['a'][5]
                }
            }
        
        expanded.append(subject_info)
    return expanded

def _expand_camera_frames(compressed_frames: List[List[float]]) -> List[Dict]:
    return [{
        "position": {"x": frame[0], "y": frame[1], "z": frame[2]},
        "rotation": {"x": frame[3], "y": frame[4], "z": frame[5]},
        "focalLength": frame[6],
        "aspectRatio": frame[7]
    } for frame in compressed_frames]

def load_simulation_file(file_path: Path, parameter_dictionary: Dict) -> Optional[Dict]:
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        
        data = msgpack.unpackb(raw_data, raw=False)
            
        if not isinstance(data, list) or len(data) != 4:
            return None

            
        cinematography_refs, simulation_refs, subjects_info, camera_frames = data
        
        cinematography_prompts = reconstruct_from_reference(
            cinematography_refs, 
            parameter_dictionary
        )['cinematography']
        simulation_instructions = reconstruct_from_reference(
            simulation_refs,
            parameter_dictionary
        )['simulation']
        
        return {
            "cinematographyPrompts": [cinematography_prompts],
            "simulationInstructions": [simulation_instructions],
            "subjectsInfo": _expand_subjects_info(unround_floats(subjects_info)),
            "cameraFrames": _expand_camera_frames(unround_floats(camera_frames))
        }
    except Exception as e:
        return None
