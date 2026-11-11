"""Install a checkpoint's coordinate scales before preparing its inputs."""

import os
import warnings

from data.simulation.dataset import SimulationDataset


def prepare_evaluation_normalization(data_module, model, model_type):
    """Restore checkpoint scales for evaluation or continued training.

    Training data modules fit their training partition by default. Existing
    weights must use the fitted transform saved with the generator instead. Historical
    checkpoints did not save that transform, so their existing full-inventory
    or explicitly configured reference protocol remains available with a warning.
    """
    if model_type != "lens_craft":
        return None

    config = getattr(data_module, "dataset_config", None)
    if config is None:
        config = data_module.simulation_config
    parameters = getattr(model, "normalization_parameters", None)
    if parameters is not None:
        if "SimulationDataset" in config["_target_"] and not config.get("normalize", True):
            raise ValueError(
                "This LensCraft checkpoint contains fitted normalization parameters; "
                "checkpoint inputs require normalize=true in the simulation dataset config."
            )
        data_module.load_state_dict({"normalization": parameters})
        return {"source": "generator_checkpoint", "training_statistics_restored": True}

    if "SimulationDataset" in config["_target_"] and not config.get("normalize", True):
        warnings.warn(
            "This LensCraft checkpoint has no saved normalization parameters; "
            "using the explicitly configured unnormalized inputs.",
            RuntimeWarning, stacklevel=2,
        )
        return {"source": "legacy_unnormalized", "training_statistics_restored": False}

    reference = os.environ.get("SIMULATION_NORMALIZATION_DATA_PATH")
    is_simulation = "SimulationDataset" in config["_target_"]
    data_path = config.get("data_path") if is_simulation else os.environ.get("SIMULATION_DATA_PATH")
    if not reference and not data_path:
        raise ValueError(
            "This LensCraft checkpoint has no saved normalization parameters. "
            "Set SIMULATION_NORMALIZATION_DATA_PATH to the original normalization "
            "reference before using it."
        )
    warnings.warn(
        "This LensCraft checkpoint has no saved normalization parameters; using "
        + ("the explicit SIMULATION_NORMALIZATION_DATA_PATH reference. " if reference
           else "the historical full simulation inventory transform. ")
        + "Its original training scales cannot be verified; a different "
        "inventory would recalibrate the checkpoint. Input preparation will not "
        "fit the new training-split transform for this checkpoint.",
        RuntimeWarning, stacklevel=2,
    )
    frame_count = int(config.get("normalization_frame_count")
                      or config.get("target_frame_count") or 30)
    parameters = SimulationDataset.get_normalization_parameters(
        data_path=reference or data_path, target_frame_count=frame_count,
    )
    data_module.load_state_dict({"normalization": parameters})
    return {
        "source": "legacy_reference" if reference else "legacy_full_inventory",
        "reference_path": str(reference or data_path),
        "training_statistics_restored": False,
    }
