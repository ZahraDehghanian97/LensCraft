"""Coordinate transforms owned by frozen metrics, independent of generators."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import warnings

from omegaconf import OmegaConf
import torch

from data.simulation.dataset import SimulationDataset


def copy_normalization(parameters):
    """Validate/copy scales without changing the active simulation transform."""
    result = {}
    for group in ("camera_position", "subject_position", "subject_dimensions"):
        result[group] = {}
        for statistic in ("mean", "std"):
            value = torch.as_tensor(parameters[group][statistic], dtype=torch.float32).detach().cpu().clone()
            if value.shape != (3,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid metric normalization {group}.{statistic}")
            if statistic == "std" and (value <= 0).any():
                raise ValueError(f"Metric normalization {group}.std must be positive")
            result[group][statistic] = value
    return result


class MetricNormalization:
    """Re-express the same physical trajectory in a judge's training scales.

    ``parameters=None`` is only meaningful with ``normalized=False``. Unknown
    historical scales fail explicitly when features are requested, rather than
    borrowing a generator's training transform and changing a frozen score.
    """
    def __init__(self, parameters=None, *, normalized=True, provenance=None):
        self.parameters = copy_normalization(parameters) if parameters is not None else None
        self.normalized = normalized
        self.provenance = dict(provenance or {})
        self.provenance["normalized"] = normalized
        self.provenance["available"] = self.parameters is not None or not normalized
        serialized = ({group: {key: value.tolist() for key, value in values.items()}
                       for group, values in self.parameters.items()}
                      if self.parameters is not None else None)
        self.provenance["statistics_sha256"] = hashlib.sha256(
            json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() if serialized is not None else None

    def convert(self, camera, subject=None, volume=None, *, source_normalization=None,
                camera_is_normalized=True, subject_is_normalized=True):
        if self.normalized and self.parameters is None:
            raise ValueError(
                "Frozen metric normalization is unavailable. Use a judge checkpoint "
                "containing its training normalization, its saved training config with "
                "a concrete original simulation data_path, or set "
                "SIMULATION_NORMALIZATION_DATA_PATH to its original reference. "
                "Generator normalization cannot substitute for judge training scales."
            )
        source = source_normalization
        if source is None:
            # Never call get_normalization_parameters here: it can fit a new
            # inventory and overwrite the generator's active transform.
            source = SimulationDataset._normalization_parameters
        needs_source = camera_is_normalized or (subject_is_normalized and
                                               (subject is not None or volume is not None))
        if needs_source and source is None:
            raise ValueError("Normalized metric inputs require their source normalization parameters")
        if source is not None:
            source = copy_normalization(source)

        def transform(tensor, group, is_normalized, positions_only=False):
            if tensor is None:
                return None
            result = tensor.clone()
            value = result[..., :3] if positions_only else result
            if is_normalized:
                value = value * source[group]["std"].to(value) + source[group]["mean"].to(value)
            if self.normalized:
                value = (value - self.parameters[group]["mean"].to(value)) / self.parameters[group]["std"].to(value)
            if positions_only:
                result[..., :3] = value
                return result
            return value

        return (transform(camera, "camera_position", camera_is_normalized, True),
                transform(subject, "subject_position", subject_is_normalized, True),
                transform(volume, "subject_dimensions", subject_is_normalized))


def _saved_simulation_config(config_path):
    if config_path is None:
        return None
    saved = OmegaConf.load(config_path)
    for key in ("data.dataset.config", "data.dataset.simulation_config"):
        dataset = OmegaConf.select(saved, key)
        if dataset is not None and "SimulationDataset" in str(dataset.get("_target_", "")):
            return dataset
    return None


def load_metric_normalization(checkpoint, *, saved_config_path=None, frame_count=30):
    """Resolve a judge's own training scales while preserving active globals.

    An unresolved environment interpolation in an old config is not evidence of
    its historical training inventory. Only a concrete saved path or an explicit
    normalization reference can supply missing legacy statistics.
    """
    parameters = checkpoint.get("normalization")
    if parameters is not None:
        return MetricNormalization(parameters, provenance={"source": "judge_checkpoint"})
    context = checkpoint.get("normalization_context", {})
    if context.get("normalized") is False:
        return MetricNormalization(normalized=False, provenance={"source": "judge_checkpoint"})

    dataset = _saved_simulation_config(saved_config_path)
    if dataset is not None and dataset.get("normalize", True) is False:
        return MetricNormalization(normalized=False, provenance={"source": "saved_training_config"})
    reference = os.environ.get("SIMULATION_NORMALIZATION_DATA_PATH")
    source = "explicit_legacy_reference" if reference else "saved_training_inventory"
    if not reference and dataset is not None:
        raw = OmegaConf.to_container(dataset, resolve=False)
        data_path = raw.get("data_path")
        if isinstance(data_path, str) and "${" not in data_path:
            # Hydra's config location does not identify the original working
            # directory. A relative path must not resolve against today's
            # evaluation checkout and silently select a different inventory.
            concrete = Path(data_path).expanduser()
            if concrete.is_absolute():
                reference = str(concrete)
    if not reference:
        warnings.warn(
            "Frozen metric checkpoint has no saved normalization and no verified original "
            "normalization reference. Trajectory scoring is unavailable until its "
            "training scales are supplied; generator scales will not be used. "
            "For relative or environment-dependent saved data paths, set "
            "SIMULATION_NORMALIZATION_DATA_PATH to the original absolute reference.",
            RuntimeWarning, stacklevel=2,
        )
        return MetricNormalization(provenance={"source": "unavailable"})

    if dataset is not None:
        for key in ("normalization_frame_count", "target_frame_count"):
            try:
                configured = dataset.get(key)
            except (ValueError, KeyError):
                continue
            if configured is not None:
                frame_count = int(configured)
                break
    # Older releases relied on the full training inventory. Reading that
    # reference must never install it as the generator's current transform.
    snapshot = {name: getattr(SimulationDataset, name) for name in (
        "_normalization_parameters", "_normalization_parameters_override",
        "_normalization_parameters_frame_count",
    ) if hasattr(SimulationDataset, name)}
    try:
        parameters = SimulationDataset.get_normalization_parameters(
            data_path=reference, target_frame_count=frame_count,
        )
        result = MetricNormalization(parameters, provenance={
            "source": source, "reference_path": str(Path(reference).expanduser().absolute()),
            "frame_count": frame_count, "historical_training_scales_verified": False,
        })
    finally:
        for name, value in snapshot.items():
            setattr(SimulationDataset, name, value)
    warnings.warn(
        "Frozen metric normalization was reconstructed from its historical reference; "
        "the legacy checkpoint did not persist its training scales, so exact historical "
        "identity cannot be verified.", RuntimeWarning, stacklevel=2,
    )
    return result


def find_training_config(checkpoint_path):
    """Find Hydra's saved config beside a native CLaTr training run."""
    parent = Path(checkpoint_path).resolve().parent
    for directory in (parent, parent.parent):
        candidate = directory / ".hydra" / "config.yaml"
        if candidate.is_file():
            return str(candidate)
    return None
