from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json

import msgpack
import numpy as np
import torch
from tqdm import tqdm

from .metadata import (
    DEFAULT_FIXED_POINT_SCALE,
    SIMULATION_FRAME_COUNT,
    cache_metadata_matches,
    resolve_dataset_root,
    write_cache_metadata,
)
from data.convertor.utils import resample_batch_trajectories
from utils.pytorch3d_transform import euler_angles_to_matrix, matrix_to_euler_angles


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
    keys = dictionary.get("keys")
    values = dictionary.get("values")
    if not isinstance(keys, list) or not isinstance(values, list):
        raise ValueError("Invalid parameter dictionary: expected list keys/values")

    for reference in refs:
        if not isinstance(reference, (list, tuple)) or len(reference) != 2:
            raise ValueError(f"Invalid parameter reference: {reference!r}")
        key_index, value_index = reference
        if (
            not isinstance(key_index, int)
            or isinstance(key_index, bool)
            or not isinstance(value_index, int)
            or isinstance(value_index, bool)
            or key_index < 0
            or value_index < 0
        ):
            raise ValueError(
                f"Parameter indices must be non-negative integers: {reference!r}"
            )
        try:
            path = keys[key_index]
            value = values[key_index][value_index]
        except (IndexError, TypeError) as error:
            raise ValueError(
                f"Parameter reference is outside its dictionary: {reference!r}"
            ) from error
        if not isinstance(path, str) or not path:
            raise ValueError(f"Invalid parameter path: {path!r}")

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

    if not isinstance(data, (list, tuple)) or len(data) != 4:
        raise ValueError(
            f"Invalid simulation payload in {file_path}: expected four sections"
        )
    cinematography_refs, simulation_refs, subjects_info, camera_frames = data
    if not isinstance(subjects_info, list) or len(subjects_info) != 1:
        raise ValueError(
            f"Simulation samples must contain exactly one subject; "
            f"{file_path} contains "
            f"{len(subjects_info) if isinstance(subjects_info, list) else 'invalid data'}"
        )
    reconstructed_prompt = reconstruct_from_reference(
        cinematography_refs, parameter_dictionary
    )
    reconstructed_instruction = reconstruct_from_reference(
        simulation_refs, parameter_dictionary
    )
    cinematography_prompt = reconstructed_prompt.get("cinematography", {})
    simulation_instruction = reconstructed_instruction.get("simulation", {})

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
    data_path = resolve_dataset_root(data_path)
    dict_path = Path(data_path) / "parameter_dictionary.msgpack"
    with dict_path.open("rb") as file:
        return msgpack.unpackb(file.read(), raw=False)


def find_simulation_files(data_path: Path) -> List[Path]:
    data_path = resolve_dataset_root(data_path)
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
    data_path = resolve_dataset_root(data_path)
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
    data_path = resolve_dataset_root(data_path)
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
    target_frame_count: int = SIMULATION_FRAME_COUNT,
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

        camera_trajectory = extract_camera_trajectory(data["cameraFrames"])
        subject_info = data["subjectsInfo"][0]
        subject = subject_info["subject"]
        subject_trajectory, _ = extract_subject_components(data["subjectsInfo"])
        if camera_trajectory.shape[0] != subject_trajectory.shape[0]:
            raise ValueError(
                f"Camera/subject frame mismatch in {file_path}: "
                f"{camera_trajectory.shape[0]} camera frames versus "
                f"{subject_trajectory.shape[0]} subject frames"
            )

        # Training gives every clip exactly target_frame_count temporal samples.
        # Calculate its statistics on that same normalized timeline so a
        # 500-frame source clip does not carry five times the weight of a
        # 100-frame clip merely because it was exported more densely.
        camera_trajectory, subject_trajectory = (
            resample_paired_euler_trajectories(
                camera_trajectory,
                subject_trajectory,
                target_frame_count,
            )
        )
        camera_positions.extend(camera_trajectory[:, :3].tolist())
        subject_positions.extend(subject_trajectory[:, :3].tolist())

        subject_dimensions.append(
            [
                subject["dimensions"]["width"],
                subject["dimensions"]["height"],
                subject["dimensions"]["depth"],
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
    target_frame_count: int = SIMULATION_FRAME_COUNT,
) -> Dict[str, Dict[str, torch.Tensor]]:
    data_path = resolve_dataset_root(data_path)
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
        target_frame_count,
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


def _validate_euler_trajectory(
    trajectory: torch.Tensor,
    target_frame_count: int,
    *,
    name: str = "Trajectory",
) -> None:
    if trajectory.ndim != 2 or trajectory.shape[-1] != 6:
        raise ValueError(
            f"Expected a [frames, 6] {name.lower()}, got "
            f"{tuple(trajectory.shape)}"
        )
    if trajectory.shape[0] < 1:
        raise ValueError(f"Cannot resample an empty {name.lower()}")
    if target_frame_count < 2:
        raise ValueError("target_frame_count must be at least 2")
    if not torch.isfinite(trajectory).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def _euler_trajectory_to_transforms(trajectory: torch.Tensor) -> torch.Tensor:
    frame_count = trajectory.shape[0]
    transform = torch.eye(
        4, dtype=trajectory.dtype, device=trajectory.device
    ).repeat(frame_count, 1, 1)
    transform[:, :3, :3] = euler_angles_to_matrix(trajectory[:, 3:6], "XYZ")
    transform[:, :3, 3] = trajectory[:, :3]
    return transform


def _transforms_to_euler_trajectory(transforms: torch.Tensor) -> torch.Tensor:
    rotation = matrix_to_euler_angles(transforms[:, :3, :3], "XYZ")
    if rotation.shape[0] > 1:
        wrapped_delta = torch.remainder(
            rotation[1:] - rotation[:-1] + torch.pi, 2 * torch.pi
        ) - torch.pi
        rotation = torch.cat(
            [rotation[:1], rotation[:1] + torch.cumsum(wrapped_delta, dim=0)],
            dim=0,
        )
    result = torch.cat([transforms[:, :3, 3], rotation], dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("Trajectory resampling produced NaN or infinite values")
    return result


def _resample_transforms(
    transforms: torch.Tensor, target_frame_count: int
) -> torch.Tensor:
    resampled, _ = resample_batch_trajectories(
        transforms,
        torch.tensor(
            transforms.shape[0],
            dtype=torch.long,
            device=transforms.device,
        ),
        target_frame_count,
    )
    return resampled


def resample_euler_trajectory(
    trajectory: torch.Tensor, target_frame_count: int
) -> torch.Tensor:
    """Resample an Euler-XYZ 6-DoF trajectory on normalized clip time.

    Translation uses linear interpolation. Rotation is converted to SO(3),
    interpolated with shortest-path quaternion SLERP, then converted back to
    Euler XYZ. This avoids both angle wraparound artifacts and treating a
    100--500-frame clip as if only its first 30 frames existed.
    """

    _validate_euler_trajectory(trajectory, target_frame_count)
    return _transforms_to_euler_trajectory(
        _resample_transforms(
            _euler_trajectory_to_transforms(trajectory),
            target_frame_count,
        )
    )


def resample_paired_euler_trajectories(
    camera_trajectory: torch.Tensor,
    subject_trajectory: torch.Tensor,
    target_frame_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample camera and subject on one timeline without breaking their relation.

    Dataset clips are normally downsampled from 100--500 frames to 30. In that
    case, shared nearest source-frame indices are selected instead of
    interpolating the two world-space paths independently. Every returned pair
    therefore existed in the validated export, preserving visibility,
    static-distance, and monotonic Dolly constraints.

    The generic upsampling path interpolates the subject's world pose together
    with the camera pose expressed in the subject's local frame. This preserves
    a fixed subject-relative camera pose while the subject translates or turns.
    """

    _validate_euler_trajectory(
        camera_trajectory,
        target_frame_count,
        name="Camera trajectory",
    )
    _validate_euler_trajectory(
        subject_trajectory,
        target_frame_count,
        name="Subject trajectory",
    )
    if camera_trajectory.shape[0] != subject_trajectory.shape[0]:
        raise ValueError(
            "Camera/subject trajectories must have the same frame count: "
            f"{camera_trajectory.shape[0]} != {subject_trajectory.shape[0]}"
        )
    if camera_trajectory.device != subject_trajectory.device:
        raise ValueError("Camera/subject trajectories must be on the same device")
    if camera_trajectory.dtype != subject_trajectory.dtype:
        raise ValueError("Camera/subject trajectories must have the same dtype")

    frame_count = camera_trajectory.shape[0]
    if frame_count >= target_frame_count:
        if frame_count == target_frame_count:
            indices = torch.arange(frame_count, device=camera_trajectory.device)
        else:
            # Build the tiny index vector on CPU so this also works on devices
            # without float64 support (notably MPS), then move only the indices.
            indices = torch.linspace(
                0,
                frame_count - 1,
                target_frame_count,
                dtype=torch.float64,
            ).round().to(device=camera_trajectory.device, dtype=torch.long)
        return (
            camera_trajectory.index_select(0, indices).clone(),
            subject_trajectory.index_select(0, indices).clone(),
        )

    camera_world = _euler_trajectory_to_transforms(camera_trajectory)
    subject_world = _euler_trajectory_to_transforms(subject_trajectory)

    subject_rotation_inverse = subject_world[:, :3, :3].transpose(-1, -2)
    camera_relative = torch.eye(
        4,
        dtype=camera_trajectory.dtype,
        device=camera_trajectory.device,
    ).repeat(frame_count, 1, 1)
    camera_relative[:, :3, :3] = (
        subject_rotation_inverse @ camera_world[:, :3, :3]
    )
    camera_relative[:, :3, 3] = (
        subject_rotation_inverse
        @ (camera_world[:, :3, 3] - subject_world[:, :3, 3]).unsqueeze(-1)
    ).squeeze(-1)

    subject_resampled = _resample_transforms(subject_world, target_frame_count)
    relative_resampled = _resample_transforms(camera_relative, target_frame_count)
    camera_resampled = torch.eye(
        4,
        dtype=camera_trajectory.dtype,
        device=camera_trajectory.device,
    ).repeat(target_frame_count, 1, 1)
    camera_resampled[:, :3, :3] = (
        subject_resampled[:, :3, :3] @ relative_resampled[:, :3, :3]
    )
    camera_resampled[:, :3, 3] = (
        subject_resampled[:, :3, :3]
        @ relative_resampled[:, :3, 3].unsqueeze(-1)
    ).squeeze(-1) + subject_resampled[:, :3, 3]

    return (
        _transforms_to_euler_trajectory(camera_resampled),
        _transforms_to_euler_trajectory(subject_resampled),
    )
