from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json

import msgpack
import numpy as np
import torch
from tqdm import tqdm

from .metadata import (
    DEFAULT_FIXED_POINT_SCALE,
    cache_metadata_matches,
    write_cache_metadata,
)


def unround_floats(obj, factor=DEFAULT_FIXED_POINT_SCALE):
    if isinstance(obj, (int, np.integer)):
        return float(obj) / factor
    if isinstance(obj, float):
        return obj
    if isinstance(obj, list):
        return [unround_floats(value, factor) for value in obj]
    if isinstance(obj, dict):
        return {key: unround_floats(value, factor) for key, value in obj.items()}
    return obj


def reconstruct_from_reference(refs: List[List[int]], dictionary: Dict) -> Dict:
    result = {}
    for key_index, value_index in refs:
        path = dictionary["keys"][key_index]
        value = dictionary["values"][key_index][value_index]

        current = result
        parts = path.split("__")
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return result


def _decode_compressed_subject_data(compressed_subjects: List[Dict]) -> List[Dict]:
    expanded = []
    for subject in compressed_subjects:
        frames = [
            {
                "position": {"x": frame[0], "y": frame[1], "z": frame[2]},
                "rotation": {"x": frame[3], "y": frame[4], "z": frame[5]},
            }
            for frame in subject["f"]
        ]

        subject_info = {
            "subject": {
                "id": subject["i"],
                "class": subject["c"],
                "dimensions": {
                    "width": subject["d"][0],
                    "height": subject["d"][1],
                    "depth": subject["d"][2],
                },
            },
            "frames": frames,
            "movementType": subject["m"],
        }

        if "a" in subject:
            subject_info["subject"]["attentionBox"] = {
                "position": {
                    "x": subject["a"][0],
                    "y": subject["a"][1],
                    "z": subject["a"][2],
                },
                "dimensions": {
                    "width": subject["a"][3],
                    "height": subject["a"][4],
                    "depth": subject["a"][5],
                },
            }

        expanded.append(subject_info)
    return expanded


def _convert_camera_arrays_to_objects(
    compressed_frames: List[List[float]],
) -> List[Dict]:
    return [
        {
            "position": {"x": frame[0], "y": frame[1], "z": frame[2]},
            "rotation": {"x": frame[3], "y": frame[4], "z": frame[5]},
            "focalLength": frame[6],
            "aspectRatio": frame[7],
        }
        for frame in compressed_frames
    ]


def parse_simulation_file_to_dict(
    file_path: Path,
    parameter_dictionary: Dict,
    fixed_point_scale: float = DEFAULT_FIXED_POINT_SCALE,
) -> Dict:
    with Path(file_path).open("rb") as file:
        data = msgpack.unpackb(file.read(), raw=False)

    cinematography_refs, simulation_refs, subjects_info, camera_frames = data
    cinematography_prompt = reconstruct_from_reference(
        cinematography_refs, parameter_dictionary
    )["cinematography"]
    simulation_instruction = reconstruct_from_reference(
        simulation_refs, parameter_dictionary
    )["simulation"]

    return {
        "cinematographyPrompts": [cinematography_prompt],
        "simulationInstructions": [simulation_instruction],
        "subjectsInfo": _decode_compressed_subject_data(
            unround_floats(subjects_info, fixed_point_scale)
        ),
        "cameraFrames": _convert_camera_arrays_to_objects(
            unround_floats(camera_frames, fixed_point_scale)
        ),
    }


def load_parameter_dictionary(data_path: Path) -> Dict:
    dict_path = Path(data_path) / "parameter_dictionary.msgpack"
    with dict_path.open("rb") as file:
        return msgpack.unpackb(file.read(), raw=False)


def find_simulation_files(data_path: Path) -> List[Path]:
    simulation_files = sorted(Path(data_path).glob("simulation_*.msgpack"))
    if not simulation_files:
        raise ValueError(f"No simulation files found in {data_path}")
    return simulation_files


def generate_movement_types_file(
    data_path: Path,
    simulation_files: Sequence[Path],
    parameter_dictionary: Dict,
    fixed_point_scale: float = DEFAULT_FIXED_POINT_SCALE,
    dataset_fingerprint: Optional[str] = None,
) -> None:
    movement_types_file = Path(data_path) / "movement_types.txt"
    metadata_file = Path(data_path) / "movement_types.meta.json"

    if movement_types_file.exists() and (
        dataset_fingerprint is None
        or cache_metadata_matches(metadata_file, dataset_fingerprint)
    ):
        print(f"Loading existing movement types from {movement_types_file}")
        return

    print(f"Generating movement types file at {movement_types_file}...")
    with movement_types_file.open("w", encoding="utf-8") as file:
        for file_path in tqdm(simulation_files, desc="Processing simulation files"):
            data = parse_simulation_file_to_dict(
                file_path, parameter_dictionary, fixed_point_scale
            )
            movement_type = data["subjectsInfo"][0]["movementType"]
            file.write(f"{file_path.name}|{movement_type}\n")

    if dataset_fingerprint is not None:
        write_cache_metadata(metadata_file, dataset_fingerprint)


def load_movement_types(data_path: Path) -> Dict[str, str]:
    movement_types = {}
    with (Path(data_path) / "movement_types.txt").open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                file_name, movement_type = parts
                movement_types[file_name] = movement_type
    return movement_types


def filter_files_by_movement_types(
    simulation_files: List[Path],
    allowed_movement_types: List[str],
    data_path: Path,
) -> List[Path]:
    if not allowed_movement_types:
        return simulation_files

    movement_types = load_movement_types(data_path)
    return [
        file_path
        for file_path in simulation_files
        if movement_types.get(file_path.name) in allowed_movement_types
    ]


def calculate_normalization_parameters_tensor(
    data_path: Path,
    parameter_dictionary: Dict,
    simulation_files: Optional[Sequence[Path]] = None,
    fixed_point_scale: float = DEFAULT_FIXED_POINT_SCALE,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if simulation_files is None:
        simulation_files = find_simulation_files(data_path)

    camera_positions = []
    subject_positions = []
    subject_dimensions = []

    print(f"Processing {len(simulation_files)} files for normalization parameters...")
    for file_path in tqdm(simulation_files):
        data = parse_simulation_file_to_dict(
            file_path, parameter_dictionary, fixed_point_scale
        )

        for frame in data["cameraFrames"]:
            camera_positions.append(
                [
                    frame["position"]["x"],
                    frame["position"]["y"],
                    frame["position"]["z"],
                ]
            )

        subject_info = data["subjectsInfo"][0]
        subject = subject_info["subject"]
        subject_dimensions.append(
            [
                subject["dimensions"]["width"],
                subject["dimensions"]["height"],
                subject["dimensions"]["depth"],
            ]
        )
        for frame in subject_info["frames"]:
            subject_positions.append(
                [
                    frame["position"]["x"],
                    frame["position"]["y"],
                    frame["position"]["z"],
                ]
            )

    camera_positions = torch.tensor(camera_positions, dtype=torch.float32)
    subject_positions = torch.tensor(subject_positions, dtype=torch.float32)
    subject_dimensions = torch.tensor(subject_dimensions, dtype=torch.float32)

    def sample_std(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] < 2:
            return torch.ones(values.shape[1], dtype=values.dtype)
        std = values.std(dim=0)
        return torch.where(std == 0, torch.ones_like(std), std)

    return {
        "camera_position": {
            "mean": camera_positions.mean(dim=0),
            "std": sample_std(camera_positions),
        },
        "subject_position": {
            "mean": subject_positions.mean(dim=0),
            "std": sample_std(subject_positions),
        },
        "subject_dimensions": {
            "mean": subject_dimensions.mean(dim=0),
            "std": sample_std(subject_dimensions),
        },
    }


def load_or_calculate_normalization_parameters(
    data_path: Path,
    parameter_dictionary: Optional[Dict] = None,
    simulation_files: Optional[Sequence[Path]] = None,
    fixed_point_scale: float = DEFAULT_FIXED_POINT_SCALE,
    dataset_fingerprint: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    data_path = Path(data_path)
    norm_params_file = data_path / "normalization_parameters.json"
    metadata_file = data_path / "normalization_parameters.meta.json"

    if norm_params_file.exists() and (
        dataset_fingerprint is None
        or cache_metadata_matches(metadata_file, dataset_fingerprint)
    ):
        print(f"Loading existing normalization parameters from {norm_params_file}")
        with norm_params_file.open("r", encoding="utf-8") as file:
            json_params = json.load(file)

        tensor_params = {
            key: {
                "mean": torch.tensor(params["mean"], dtype=torch.float32),
                "std": torch.tensor(params["std"], dtype=torch.float32),
            }
            for key, params in json_params.items()
        }
        for params in tensor_params.values():
            params["std"] = torch.where(
                params["std"] == 0,
                torch.ones_like(params["std"]),
                params["std"],
            )
        return tensor_params

    if parameter_dictionary is None:
        parameter_dictionary = load_parameter_dictionary(data_path)
    if simulation_files is None:
        simulation_files = find_simulation_files(data_path)

    print(f"Calculating normalization parameters from dataset at {data_path}")
    norm_params = calculate_normalization_parameters_tensor(
        data_path,
        parameter_dictionary,
        simulation_files,
        fixed_point_scale,
    )

    json_params = {
        key: {
            "mean": params["mean"].tolist(),
            "std": params["std"].tolist(),
        }
        for key, params in norm_params.items()
    }
    with norm_params_file.open("w", encoding="utf-8") as file:
        json.dump(json_params, file, indent=4)
        file.write("\n")

    if dataset_fingerprint is not None:
        write_cache_metadata(metadata_file, dataset_fingerprint)

    print(f"Normalization parameters saved to {norm_params_file}")
    return norm_params


def extract_camera_trajectory(camera_frames: List[Dict]) -> torch.Tensor:
    return torch.tensor(
        [
            [
                frame["position"]["x"],
                frame["position"]["y"],
                frame["position"]["z"],
                frame["rotation"]["x"],
                frame["rotation"]["y"],
                frame["rotation"]["z"],
            ]
            for frame in camera_frames
        ],
        dtype=torch.float32,
    )


def extract_subject_components(
    subjects_info: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    subject_info = subjects_info[0]
    subject = subject_info["subject"]

    location_rotation = torch.tensor(
        [
            [
                frame["position"]["x"],
                frame["position"]["y"],
                frame["position"]["z"],
                frame["rotation"]["x"],
                frame["rotation"]["y"],
                frame["rotation"]["z"],
            ]
            for frame in subject_info["frames"]
        ],
        dtype=torch.float32,
    )
    volume = torch.tensor(
        [
            [
                subject["dimensions"]["width"],
                subject["dimensions"]["height"],
                subject["dimensions"]["depth"],
            ]
        ],
        dtype=torch.float32,
    )
    return location_rotation, volume
