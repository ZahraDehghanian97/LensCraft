from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import msgpack
import json
import torch
from tqdm import tqdm

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

def reconstruct_from_reference(refs: List[List[int]], dictionary: Dict) -> Dict:    
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

def _decode_compressed_subject_data(compressed_subjects: List[Dict]) -> List[Dict]:
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
            "frames": frames,
            "movementType": subject['m']
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

def _convert_camera_arrays_to_objects(compressed_frames: List[List[float]]) -> List[Dict]:
    return [{
        "position": {"x": frame[0], "y": frame[1], "z": frame[2]},
        "rotation": {"x": frame[3], "y": frame[4], "z": frame[5]},
        "focalLength": frame[6],
        "aspectRatio": frame[7]
    } for frame in compressed_frames]

def parse_simulation_file_to_dict(file_path: Path, parameter_dictionary: Dict) -> Optional[Dict]:
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
            "subjectsInfo": _decode_compressed_subject_data(unround_floats(subjects_info)),
            "cameraFrames": _convert_camera_arrays_to_objects(unround_floats(camera_frames))
        }
    except Exception as e:
        return None

def validate_dataset_directory(data_path: Path) -> None:
    """Validate that the data directory exists and contains required files."""
    if not data_path.is_dir():
        raise ValueError(f"Expected directory at {data_path}")
        
    dict_path = data_path / "parameter_dictionary.msgpack"
    if not dict_path.exists():
        raise ValueError(f"parameter_dictionary.msgpack not found in {data_path}")

def load_parameter_dictionary(data_path: Path) -> Dict:
    """Load the parameter dictionary from the given data path."""
    dict_path = data_path / "parameter_dictionary.msgpack"
    with open(dict_path, 'rb') as f:
        return msgpack.unpackb(f.read(), raw=False)

def find_simulation_files(data_path: Path) -> List[Path]:
    """Find and sort all simulation files in the given directory."""
    simulation_files = sorted(data_path.glob('simulation_*.msgpack'))
    if not simulation_files:
        raise ValueError(f"No simulation files found in {data_path}")
    return simulation_files

def generate_movement_types_file(data_path: Path, simulation_files: List[Path], parameter_dictionary: Dict) -> None:
    """Generate a file mapping simulation files to their movement types."""
    movement_types_file = data_path / "movement_types.txt"
    
    if movement_types_file.exists():
        print(f"Movement types file already exists at {movement_types_file}")
        return
    
    print(f"Generating movement types file at {movement_types_file}...")
    processed = 0
    
    with open(movement_types_file, 'w') as f:
        for idx, file_path in enumerate(simulation_files):
            try:
                data = parse_simulation_file_to_dict(file_path, parameter_dictionary)
                if data and "subjectsInfo" in data and data["subjectsInfo"]:
                    movement_type = data["subjectsInfo"][0].get("movementType", "unknown")
                    f.write(f"{file_path.name}|{movement_type}\n")
                else:
                    f.write(f"{file_path.name}|unknown\n")
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{len(simulation_files)} files...")
            except Exception as e:
                f.write(f"{file_path.name}|error\n")
                print(f"Error processing {file_path}: {e}")
    
    print(f"Completed generating movement types file. Processed {processed} files.")

def load_movement_types(data_path: Path) -> Dict[str, str]:
    """Load movement types from the movement_types.txt file."""
    movement_types_file = data_path / "movement_types.txt"
    movement_types = {}
    
    with open(movement_types_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('|')
                if len(parts) == 2:
                    file_name, movement_type = parts
                    movement_types[file_name] = movement_type
    
    return movement_types

def filter_files_by_movement_types(simulation_files: List[Path], allowed_movement_types: List[str], data_path: Path) -> List[Path]:
    """Filter simulation files by allowed movement types."""
    if not allowed_movement_types:
        return simulation_files
    
    movement_types = load_movement_types(data_path)
    return [
        file_path for file_path in simulation_files 
        if movement_types.get(file_path.name, "unknown") in allowed_movement_types
    ]

def calculate_normalization_parameters(data_path: Path, parameter_dictionary: Dict) -> Dict:
    """Calculate normalization parameters from the dataset."""
    print(f"Calculating normalization parameters from dataset at {data_path}")
    
    simulation_files = find_simulation_files(data_path)
    
    print(f"Processing {len(simulation_files)} files for normalization parameters...")
    
    camera_positions = []
    camera_rotations = []
    subject_positions = []
    subject_rotations = []
    subject_dimensions = []
    
    for file_path in tqdm(simulation_files):
        try:
            data = parse_simulation_file_to_dict(file_path, parameter_dictionary)
            
            for frame in data["cameraFrames"]:
                camera_positions.append([
                    frame["position"]["x"],
                    frame["position"]["y"],
                    frame["position"]["z"]
                ])
                camera_rotations.append([
                    frame["rotation"]["x"],
                    frame["rotation"]["y"],
                    frame["rotation"]["z"]
                ])
            
            if data["subjectsInfo"]:
                subject_info = data["subjectsInfo"][0]
                subject = subject_info["subject"]
                
                subject_dimensions.append([
                    subject["dimensions"]["width"],
                    subject["dimensions"]["height"],
                    subject["dimensions"]["depth"]
                ])
                
                for frame in subject_info["frames"]:
                    subject_positions.append([
                        frame["position"]["x"],
                        frame["position"]["y"],
                        frame["position"]["z"]
                    ])
                    subject_rotations.append([
                        frame["rotation"]["x"],
                        frame["rotation"]["y"],
                        frame["rotation"]["z"]
                    ])
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    camera_positions = torch.tensor(camera_positions, dtype=torch.float32)
    camera_rotations = torch.tensor(camera_rotations, dtype=torch.float32)
    subject_positions = torch.tensor(subject_positions, dtype=torch.float32)
    subject_rotations = torch.tensor(subject_rotations, dtype=torch.float32)
    subject_dimensions = torch.tensor(subject_dimensions, dtype=torch.float32)
    
    norm_params = {
        "camera_position": {
            "mean": camera_positions.mean(dim=0).tolist(),
            "std": camera_positions.std(dim=0).tolist()
        },
        "camera_rotation": {
            "mean": camera_rotations.mean(dim=0).tolist(),
            "std": camera_rotations.std(dim=0).tolist()
        },
        "subject_position": {
            "mean": subject_positions.mean(dim=0).tolist(),
            "std": subject_positions.std(dim=0).tolist()
        },
        "subject_rotation": {
            "mean": subject_rotations.mean(dim=0).tolist(),
            "std": subject_rotations.std(dim=0).tolist()
        },
        "subject_dimensions": {
            "mean": subject_dimensions.mean(dim=0).tolist(),
            "std": subject_dimensions.std(dim=0).tolist()
        }
    }
    
    return norm_params

def load_or_calculate_normalization_parameters(data_path: Path, parameter_dictionary: Dict = None) -> Dict:
    """Load existing normalization parameters or calculate them if not available."""
    data_path = Path(data_path)
    norm_params_file = data_path / "normalization_parameters.json"
    
    if norm_params_file.exists():
        print(f"Loading existing normalization parameters from {norm_params_file}")
        with open(norm_params_file, 'r') as f:
            return json.load(f)
    
    if parameter_dictionary is None:
        parameter_dictionary = load_parameter_dictionary(data_path)
    
    norm_params = calculate_normalization_parameters(data_path, parameter_dictionary)
    
    with open(norm_params_file, 'w') as f:
        json.dump(norm_params, f, indent=4)
    
    print(f"Normalization parameters saved to {norm_params_file}")
    return norm_params

def extract_camera_trajectory(camera_frames: List[Dict]) -> torch.Tensor:
    """Extract camera trajectory from camera frames."""
    return torch.tensor([
        [
            frame["position"]["x"],
            frame["position"]["y"],
            frame["position"]["z"],
            frame["rotation"]["x"],
            frame["rotation"]["y"],
            frame["rotation"]["z"],
            # frame["focalLength"]
        ]
        for frame in camera_frames
    ], dtype=torch.float32)

def extract_subject_components(subjects_info: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract subject trajectory and volume from subjects info."""
    subject_info = subjects_info[0]
    subject = subject_info["subject"]
    
    loc_rot = torch.tensor([
        [
            frame["position"]["x"],
            frame["position"]["y"],
            frame["position"]["z"],
            frame["rotation"]["x"],
            frame["rotation"]["y"],
            frame["rotation"]["z"]
        ]
        for frame in subject_info["frames"]
    ], dtype=torch.float32)
    
    vol = torch.tensor([
        [
            subject["dimensions"]["width"],
            subject["dimensions"]["height"],
            subject["dimensions"]["depth"]
        ]
    ], dtype=torch.float32)
    
    return loc_rot, vol
