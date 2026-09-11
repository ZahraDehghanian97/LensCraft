"""Dataset and inference adapters for the qualitative comparison workspace.

The portable JSON format contains camera-to-world OpenCV poses in a Y-up world.
Reading it and creating the demonstration requires only NumPy; training and
dataset dependencies are imported only when that functionality is requested.
"""

from __future__ import annotations

import copy
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

DATASETS = ("simulation", "et", "ccdm")
MODELS = ("lens_craft", "ccdm", "et", "gendop")
MODES = ("prompt_generation", "reconstruction", "key_framing+prompt", "key_framing")
SCHEMA = "lenscraft.qualitative.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _array(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _poses(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 3 or result.shape[1:] != (4, 4) or len(result) == 0:
        raise ValueError(f"{name} must contain at least one pose with shape [T, 4, 4].")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite coordinates.")
    if not np.allclose(result[:, 3], [0, 0, 0, 1], atol=1e-4):
        raise ValueError(f"{name} does not contain homogeneous camera-to-world poses.")
    return result


@dataclass
class ComparisonSample:
    sample_id: str
    dataset: str
    prompt: str
    trajectories: dict[str, np.ndarray]
    subject: np.ndarray | None = None
    volume: np.ndarray | None = None
    keyframes: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _context: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not self.trajectories:
            raise ValueError("A comparison must contain at least one trajectory.")
        self.trajectories = {str(k): _poses(v, str(k)) for k, v in self.trajectories.items()}
        if self.subject is not None:
            self.subject = _poses(self.subject, "subject")
        if self.volume is not None:
            self.volume = np.asarray(self.volume, dtype=np.float32).reshape(-1)[:3]
            if self.volume.shape != (3,) or not np.isfinite(self.volume).all() or (self.volume <= 0).any():
                raise ValueError("Subject volume must contain three positive finite dimensions.")
        self.keyframes = _validate_keyframes(self.keyframes, self.frame_count)
        self.metadata.setdefault("camera_convention", "opencv")
        self.metadata.setdefault("up_axis", "y")

    @property
    def frame_count(self) -> int:
        return len(self.trajectories.get("GT", next(iter(self.trajectories.values()))))


def _validate_keyframes(indices, frame_count):
    result = []
    for index in indices:
        if isinstance(index, bool) or int(index) != index or not 0 <= int(index) < frame_count:
            raise ValueError(f"Keyframe {index!r} must be an integer between 0 and {frame_count - 1}.")
        result.append(int(index))
    return sorted(set(result))


def _json_value(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def save_result(sample: ComparisonSample | Iterable[ComparisonSample], path: str | Path) -> Path:
    """Write a portable comparison with no checkpoint or normalization dependency."""
    samples = [sample] if isinstance(sample, ComparisonSample) else list(sample)
    if not samples:
        raise ValueError("There are no samples to save.")
    payload = {"schema": SCHEMA, "samples": [
        {name: _json_value(getattr(item, name)) for name in (
            "sample_id", "dataset", "prompt", "trajectories", "subject", "volume", "keyframes", "metadata"
        )} for item in samples
    ]}
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _select_item(value, index, batch_size, *, batched_ndim=None):
    if value is None:
        return None
    array = np.asarray(value)
    if batched_ndim is not None:
        return array[index] if array.ndim == batched_ndim else array
    if array.ndim > 0 and len(array) == batch_size:
        return value[index]
    return value


def _sim_standard(camera, subject=None, volume=None, *, normalized=False):
    import torch
    from data.simulation.convertor import SIMConvertor

    camera = torch.as_tensor(camera, dtype=torch.float32)
    subject = None if subject is None else torch.as_tensor(subject, dtype=torch.float32)
    volume = None if volume is None else torch.as_tensor(volume, dtype=torch.float32)
    if normalized:
        from dotenv import load_dotenv
        from data.simulation.dataset import SimulationDataset
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        try:
            camera, subject, volume = SimulationDataset.normalize_item(camera, subject, volume, False)
        except (ValueError, FileNotFoundError) as error:
            raise ValueError("This legacy inference file stores normalized coordinates. Configure "
                             "SIMULATION_DATA_PATH to the training dataset, then save a portable comparison.") from error
    output = SIMConvertor().to_standard(camera, subject, volume)
    return tuple(_array(value) for value in output)


def load_result(path: str | Path, index: int = 0) -> ComparisonSample:
    """Load a portable comparison or the existing inference_result.json format."""
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A comparison file must contain a JSON object.")
    if "samples" in payload:
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported comparison schema: {payload.get('schema')!r}")
        if not 0 <= index < len(payload["samples"]):
            raise IndexError(f"Sample index {index} is outside this file's {len(payload['samples'])} samples.")
        sample = ComparisonSample(**payload["samples"][index])
        sample.metadata.setdefault("source_file", str(path))
        return sample
    if "trajectories" not in payload:
        raise ValueError("Expected a comparison bundle or inference_result.json with trajectories.")
    values = payload["trajectories"]
    if not isinstance(values, dict) or not values:
        raise ValueError("The result file contains no trajectories.")
    first = np.asarray(next(iter(values.values())))
    matrix_format = first.shape[-2:] == (4, 4)
    batched = first.ndim == (4 if matrix_format else 3)
    batch_size = len(first) if batched else 1
    if not 0 <= index < batch_size:
        raise IndexError(f"Sample index {index} is outside this file's {batch_size} samples.")
    batch = payload.get("batch_data", {})
    model = payload.get("model_type", "lens_craft")
    padding = _select_item(batch.get("padding_mask"), index, batch_size, batched_ndim=2)
    trajectories = {}
    for name, value in values.items():
        item = np.asarray(value)[index] if batched else np.asarray(value)
        if not matrix_format:
            item = _sim_standard(item, normalized=(name == "GT" or model == "lens_craft"))[0]
        if padding is not None and len(item) == len(padding):
            item = item[~np.asarray(padding, dtype=bool)]
        key = "GT" if name == "GT" else f"{model}:{name}"
        trajectories[key] = item
    raw_subject = batch.get("subject_trajectory")
    subject_matrix_format = raw_subject is not None and np.asarray(raw_subject).shape[-2:] == (4, 4)
    subject = _select_item(raw_subject, index, batch_size, batched_ndim=4 if subject_matrix_format else 3)
    volume = _select_item(batch.get("subject_volume"), index, batch_size)
    if subject is not None:
        if subject.shape[-2:] != (4, 4):
            gt = np.asarray(values.get("GT", next(iter(values.values()))))
            gt = gt[index] if batched else gt
            _, subject, volume = _sim_standard(gt, subject, volume, normalized=True)
        if padding is not None and len(subject) == len(padding):
            subject = subject[~np.asarray(padding, dtype=bool)]
    mask = _select_item(batch.get("key_framing_padding_mask"), index, batch_size, batched_ndim=2)
    keyframes = []
    if mask is not None and model == "lens_craft":
        visible = ~np.asarray(mask, dtype=bool)
        if padding is not None and len(visible) == len(padding):
            visible = visible[~np.asarray(padding, dtype=bool)]
        keyframes = np.flatnonzero(visible).tolist()
    prompt = _select_item(batch.get("text_prompts", payload.get("prompt", "")), index, batch_size)
    sample_id = _select_item(batch.get("item_ids"), index, batch_size)
    runs = {}
    for name in trajectories:
        if name == "GT":
            continue
        mode = name.split(":", 1)[-1]
        uses_keyframes = model == "lens_craft" and mode in ("key_framing", "key_framing+prompt")
        runs[name] = {"model": model, "mode": mode, "keyframes": keyframes if uses_keyframes else []}
    return ComparisonSample(str(sample_id or f"{path.stem}:{index}"), payload.get("dataset_type", "unknown"),
        str(prompt or ""), trajectories, subject, volume, keyframes,
        {"source_file": str(path), "legacy_result": True, "keyframe_source": "GT",
         "sample_index": index, "raw_prompt": _select_item(batch.get("raw_prompt"), index, batch_size), "runs": runs})


def _split_indices(length, split, seed, val_size, test_size):
    if split == "all":
        return list(range(length))
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, test, or all.")
    import torch
    train_count = int((1 - val_size - test_size) * length)
    val_count = int(val_size * length)
    permutation = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    ranges = {"train": (0, train_count), "val": (train_count, train_count + val_count),
              "test": (train_count + val_count, length)}
    start, stop = ranges[split]
    return permutation[start:stop]


@contextmanager
def _generation_seed(seed, device):
    """Give each model the same reproducible seed without altering caller RNGs."""
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class VisualizationRepository:
    """Cache native datasets and models during interactive sample browsing."""

    def __init__(self, config_dir=None, overrides=(), device="auto", seed=42):
        self.config_dir = Path(config_dir or PROJECT_ROOT / "config").resolve()
        self.overrides = tuple(overrides)
        self.device = device
        self.seed = int(seed)
        if not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
        self._datasets = {}
        self._models = {}
        self._inference_datasets = {}

    def _config(self, dataset, model="lens_craft", data_path=None):
        from dotenv import load_dotenv
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        dataset_group = "default" if dataset == "simulation" else dataset
        model_group = "default" if model == "lens_craft" else model
        # General overrides apply to every model; model-specific ones should be
        # configured through each baseline's environment variables/config file.
        overrides = [f"data/dataset={dataset_group}", f"training/model={model_group}"]
        overrides.extend(self.overrides)
        with initialize_config_dir(version_base=None, config_dir=str(self.config_dir)):
            config = compose(config_name="inference", overrides=overrides)
        if data_path is not None:
            path = Path(data_path).expanduser().resolve()
            if dataset == "simulation":
                config.data.dataset.config.data_path = str(path.parent if path.is_file() else path)
            elif dataset == "ccdm":
                config.data.dataset.config.data_path = str(path / "data.npy" if path.is_dir() else path)
            else:
                config.data.dataset.config.dataset_dir = str(path)
        return config

    def _dataset(self, dataset, split, data_path=None):
        if dataset == "lens_craft":
            dataset = "simulation"
        if dataset not in DATASETS:
            raise ValueError(f"Dataset must be one of {', '.join(DATASETS)}. GenDoP is an inference model in this repository.")
        if split not in ("train", "val", "test", "all"):
            raise ValueError("split must be train, val, test, or all.")
        cache_key = (dataset, split, str(data_path))
        if cache_key in self._datasets:
            return self._datasets[cache_key]
        config = self._config(dataset, data_path=data_path)
        dataset_config = config.data.dataset.config
        if dataset == "simulation":
            from data.simulation.loader import find_simulation_files, load_parameter_dictionary
            from data.simulation.metadata import resolve_dataset_root, load_dataset_manifest, get_fixed_point_scale
            path = resolve_dataset_root(Path(dataset_config.data_path).expanduser())
            paths = find_simulation_files(path)
            native = {"path": path, "files": paths, "dictionary": load_parameter_dictionary(path),
                      "scale": get_fixed_point_scale(load_dataset_manifest(path))}
            ids = [p.stem for p in paths]
        elif dataset == "ccdm":
            path = Path(dataset_config.data_path).expanduser()
            # Native CCDM uses a dictionary saved by NumPy, as its training loader does.
            native = np.load(path, allow_pickle=True).item()
            ids = [str(i) for i in range(len(native["cam"]))]
        else:
            from data.et.load import load_et_dataset
            if split == "all":
                raise ValueError("E.T. has native splits; choose train, val, or test.")
            native = load_et_dataset(dataset_config.project_config_dir, dataset_config.dataset_dir,
                                     dataset_config.set_name, split)
            ids = [Path(name).stem for name in native.root_filenames]
        indices = list(range(len(ids))) if dataset == "et" else _split_indices(
            len(ids), split, self.seed, float(config.data.val_size), float(config.data.test_size))
        result = {"dataset": dataset, "split": split, "config": config, "native": native,
                  "indices": indices, "ids": ids, "data_path": data_path}
        self._datasets[cache_key] = result
        return result

    def list_samples(self, dataset="simulation", split="test", data_path=None):
        handle = self._dataset(dataset, split, data_path)
        return [{"index": i, "sample_id": handle["ids"][native_index]}
                for i, native_index in enumerate(handle["indices"])]

    def load_sample(self, dataset="simulation", split="test", index=0, sample_id=None, data_path=None):
        handle = self._dataset(dataset, split, data_path)
        indices = handle["indices"]
        if sample_id is None and data_path is not None and Path(data_path).is_file() and dataset == "simulation":
            sample_id = Path(data_path).stem
        if sample_id is not None:
            identifiers = {str(sample_id), Path(str(sample_id)).name, Path(str(sample_id)).stem}
            matches = [i for i, native_index in enumerate(indices)
                       if handle["ids"][native_index] in identifiers]
            if not matches:
                raise ValueError(f"Sample {sample_id!r} is not in the {split} split. Use split=all to select any simulation/CCDM item.")
            index = matches[0]
        if not 0 <= int(index) < len(indices):
            raise IndexError(f"Sample index {index} is outside the {split} split ({len(indices)} samples).")
        native_index = indices[int(index)]
        native, config = handle["native"], handle["config"]
        dataset = handle["dataset"]
        metadata = {"split": split, "index": int(index), "dataset_index": native_index,
                    "split_seed": self.seed, "raw_prompt": None,
                    "split_policy": "native" if dataset == "et" else "seeded_random"}
        if dataset == "simulation":
            from data.simulation.loader import parse_simulation_file_to_dict, extract_camera_trajectory, extract_subject_components, resample_paired_euler_trajectories
            from data.simulation.caption import extract_text_prompt
            raw = parse_simulation_file_to_dict(native["files"][native_index], native["dictionary"], native["scale"])
            camera = extract_camera_trajectory(raw["cameraFrames"])
            subject, volume = extract_subject_components(raw["subjectsInfo"])
            metadata["original_frame_count"] = len(camera)
            camera, subject = resample_paired_euler_trajectories(camera, subject, int(config.training.model.data_format.seq_length))
            camera, subject, volume = _sim_standard(camera, subject, volume)
            prompt = extract_text_prompt(raw["cinematographyPrompts"][0], raw["subjectsInfo"][0]["movementType"])
            metadata.update(source_file=str(native["files"][native_index]), raw_prompt=raw["cinematographyPrompts"][0],
                            raw_instruction=raw["simulationInstructions"][0])
        elif dataset == "ccdm":
            import torch
            from data.ccdm.convertor import CCDMConvertor
            trajectory = torch.as_tensor(native["cam"][native_index][:300], dtype=torch.float32)
            camera, subject, volume = map(_array, CCDMConvertor().to_standard(trajectory))
            info = native["info"][native_index]
            prompt = info if isinstance(info, str) else " ".join(info)
            metadata["subject_geometry"] = "default proxy at the CCDM subject origin"
        else:
            from data.et.dataset import ETDataset
            from data.et.convertor import ETConvertor
            item = native[native_index]
            camera, subject, _ = ETDataset.normalize_item(item["traj_feat"].T, item["char_feat"].T, None, False)
            camera, subject, volume = map(_array, ETConvertor().to_standard(camera, subject))
            valid = _array(item["padding_mask"]).astype(bool)
            camera, subject = camera[valid], subject[valid]
            prompt = item["caption_raw"]["caption"]
            metadata["subject_geometry"] = "default proxy dimensions"
        return ComparisonSample(handle["ids"][native_index], dataset, str(prompt), {"GT": camera}, subject, volume,
                                metadata=metadata, _context=(self, handle, native_index))

    def _batch(self, handle, native_index):
        import hydra
        dataset = handle["dataset"]
        key = (dataset, handle["split"], str(handle["data_path"]))
        if key not in self._inference_datasets:
            from omegaconf import OmegaConf
            dataset_config = OmegaConf.create(OmegaConf.to_container(handle["config"].data.dataset.config, resolve=True))
            if not dataset_config.get("normalize", True):
                raise ValueError("Model inference requires data.dataset.config.normalize=true, as in the training pipeline.")
            if dataset == "et":
                dataset_config.split = handle["split"]
            self._inference_datasets[key] = hydra.utils.instantiate(dataset_config)
        native = self._inference_datasets[key]
        if dataset == "simulation":
            from data.simulation.dataset import collate_fn
            # Training configs can filter movement types, changing the native
            # dataset's indices. Always preserve the preview's selected file.
            selected_path = handle["native"]["files"][native_index].resolve()
            matching_indices = [i for i, path in enumerate(native.simulation_files)
                                if Path(path).resolve() == selected_path]
            if not matching_indices:
                raise ValueError(
                    f"Selected sample {selected_path.stem!r} is excluded by the inference dataset "
                    "filters. Clear data.dataset.config.allowed_movement_types or select an included sample.")
            native_index = matching_indices[0]
        elif dataset == "et":
            from data.et.dataset import collate_fn
        else:
            from data.ccdm.dataset import collate_fn
        batch = collate_fn([native[native_index]])
        batch["random_prompt_index"] = [0]
        return batch

    def generate(self, sample, models=("lens_craft",), mode="prompt_generation", keyframes=None, *, seed=None, device=None):
        import torch
        from data.sim_format import MEMORY_TEACHER_FORCING_BY_MODE, to_simulation_format
        from data.simulation.utils import structured_conditioning_from_batch
        from inferencing.process import inference_batch
        from models.factory import load_model
        from utils.device import move_batch_to_device

        if mode not in MODES:
            raise ValueError(f"Mode must be one of {', '.join(MODES)}.")
        models = ["lens_craft" if model in ("ours", "lenscraft") else model for model in models]
        if any(model not in MODELS for model in models):
            raise ValueError(f"Models must be chosen from {', '.join(MODELS)}.")
        if sample._context is None:
            raise ValueError("Inference requires a native dataset sample. Load it using its dataset, split and sample ID.")
        repository, handle, native_index = sample._context
        if repository is not self:
            return repository.generate(sample, models, mode, keyframes, seed=seed, device=device)
        seed = self.seed if seed is None else int(seed)
        if not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
        requested_device = device or self.device
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(requested_device)
        batch = move_batch_to_device(self._batch(handle, native_index), torch_device)
        output = copy.copy(sample)
        output.trajectories = dict(sample.trajectories)
        output.metadata = copy.deepcopy(sample.metadata)
        output.metadata.update(seed=seed, mode=mode, keyframe_source="GT")
        # Existing mode columns keep their original conditioning. A later
        # text-only run must neither erase it nor claim to consume keyframes.
        output.keyframes = list(sample.keyframes)
        errors = {}
        generated_methods = []
        for model_type in dict.fromkeys(models):
            try:
                cfg = self._config(sample.dataset, model_type, handle["data_path"])
                model_key = (model_type, str(torch_device))
                if model_key not in self._models:
                    with _generation_seed(seed, torch_device):
                        self._models[model_key] = load_model(cfg, model_type, torch_device)
                model = self._models[model_key]
                seq_length = int(cfg.training.model.data_format.seq_length)
                constraints = {}
                with _generation_seed(seed, torch_device), torch.inference_mode():
                    if model_type == "lens_craft":
                        sim_camera, sim_subject, sim_volume, padding = to_simulation_format(batch, sample.dataset, target_len=seq_length)
                        if padding is None:
                            padding = torch.zeros(sim_camera.shape[:2], dtype=torch.bool, device=torch_device)
                        source_mask = padding.clone()
                        if mode in ("key_framing", "key_framing+prompt"):
                            selected = _validate_keyframes(keyframes if keyframes is not None else [0, sample.frame_count - 1], sample.frame_count)
                            if not selected:
                                raise ValueError("Select at least one keyframe for keyframe generation.")
                            timeline_indices = {}
                            for i in selected:
                                model_index = round(i * (seq_length - 1) / max(sample.frame_count - 1, 1))
                                timeline_indices.setdefault(model_index, i)
                            source_indices = sorted(timeline_indices)
                            source_mask[:] = True
                            source_mask[:, source_indices] = False
                            source_mask |= padding
                            if source_mask.all():
                                raise ValueError("The selected keyframes are all padded; choose valid frames.")
                            camera_standard = _sim_standard(sim_camera, normalized=True)[0][0]
                            valid_indices = [i for i in source_indices if not bool(padding[0, i])]
                            selected = [timeline_indices[i] for i in valid_indices]
                            constraints = {"keyframe_poses": camera_standard[valid_indices].tolist(),
                                           "keyframe_model_indices": valid_indices,
                                           "keyframe_sequence_length": seq_length}
                        caption = structured_conditioning_from_batch(batch) if sample.dataset in ("simulation", "et") else None
                        if mode in ("prompt_generation", "key_framing+prompt") and caption is None:
                            raise ValueError("LensCraft prompt conditioning needs structured cinematography annotations; this dataset sample has none.")
                        generated = model.generate_camera_trajectory(subject_trajectory=sim_subject, subject_volume=sim_volume,
                            camera_trajectory=sim_camera, src_key_mask=source_mask, padding_mask=padding,
                            memory_teacher_forcing_ratio=MEMORY_TEACHER_FORCING_BY_MODE[mode], caption_embedding=caption)["reconstructed"]
                        poses = _sim_standard(generated, normalized=True)[0][0]
                        poses = poses[~_array(padding[0]).astype(bool)]
                        name = f"lens_craft:{mode}"
                    else:
                        inference_model = model
                        if model_type == "et":
                            # E.T. has its own advancing seed stream; resetting
                            # global RNGs alone does not reproduce its output.
                            inference_model = SimpleNamespace(generate_using_text=lambda *args, **kwargs:
                                model.generate_using_text(*args, generation_seeds=[seed], **kwargs))
                        result, _, _, _, baseline_padding, _ = inference_batch(
                            inference_model, batch, torch_device, sample.dataset, model_type, seq_length)
                        # Baseline adapters return raw simulation-world coordinates.
                        poses = _sim_standard(result["prompt_generation"], normalized=False)[0][0]
                        # E.T. preserves native temporal padding; CCDM/GenDoP
                        # generate a complete output sequence and have no mask.
                        if model_type == "et" and baseline_padding is not None and len(poses) == baseline_padding.shape[1]:
                            poses = poses[~_array(baseline_padding[0]).astype(bool)]
                        name = f"{model_type}:prompt_generation"
                    output.trajectories[name] = _poses(poses, name)
                    generated_methods.append(name)
                    if constraints:
                        output.keyframes = selected
                        output.metadata.update(constraints)
                    output.metadata.setdefault("runs", {})[name] = {"seed": seed, "model": model_type,
                        "mode": mode if model_type == "lens_craft" else "prompt_generation",
                        "checkpoint": str(cfg.training.model.inference.get("checkpoint_path", "adapter configuration")),
                        "keyframes": list(selected) if constraints else [], **constraints}
            except Exception as error:
                errors[model_type] = f"{type(error).__name__}: {error}"
        output.metadata["errors"] = errors
        output.metadata["last_run"] = {"models": models, "mode": mode, "seed": seed,
                                        "generated_methods": generated_methods, "errors": errors}
        return output


def load_dataset_sample(dataset="simulation", split="test", index=0, sample_id=None, overrides=(), seed=42, data_path=None):
    return VisualizationRepository(overrides=overrides, seed=seed).load_sample(dataset, split, index, sample_id, data_path)


def list_samples(dataset="simulation", split="test", overrides=(), seed=42, data_path=None):
    return VisualizationRepository(overrides=overrides, seed=seed).list_samples(dataset, split, data_path)


def run_models(sample, models=("lens_craft",), mode="prompt_generation", keyframes=None, device="auto", seed=42, overrides=()):
    repository = sample._context[0] if sample._context is not None else VisualizationRepository(overrides=overrides)
    return repository.generate(sample, models, mode, keyframes, seed=seed, device=device)


def make_demo() -> ComparisonSample:
    """Deterministic illustrative paths, explicitly marked as synthetic examples."""
    count = 60
    time = np.linspace(0, 1, count)
    subject = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    subject[:, :3, 3] = np.column_stack((2 * time - 1, np.full(count, 0.85), 0.25 * np.sin(time * np.pi)))
    angle = -0.6 + time * 1.6
    position = subject[:, :3, 3] + np.column_stack((4 * np.sin(angle), 1.1 + 0.35 * time, 4 * np.cos(angle)))

    def look_at(positions):
        forward = subject[:, :3, 3] - positions
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        down = np.cross(forward, right)
        poses = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
        poses[:, :3, :3] = np.stack((right, down, forward), axis=-1)
        poses[:, :3, 3] = positions
        return poses

    variants = {"GT": look_at(position), "Demo A": look_at(position + np.column_stack((0.08 * np.sin(3 * np.pi * time), 0.05 * np.sin(np.pi * time), 0.06 * time))),
                "Demo B": look_at(position + np.column_stack((0.7 * time, 0.3 * np.sin(time * np.pi), -0.5 * time)))}
    return ComparisonSample("demo-orbit", "demo", "Orbit around a walking subject while gradually rising.", variants,
        subject, np.array([0.5, 1.7, 0.35]), [0, 20, 40, 59],
        {"demo": True, "notice": "Synthetic illustration — these paths are not model predictions.",
         "keyframe_source": "GT", "keyframes_illustrative": True})
