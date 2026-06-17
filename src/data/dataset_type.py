from typing import Literal

DatasetType = Literal["ccdm", "et", "simulation"]


def resolve_dataset_type(target: str) -> DatasetType:
    """Map a dataset Hydra ``_target_`` string to its dataset-type tag."""
    if "CCDMDataset" in target:
        return "ccdm"
    if "ETDataset" in target:
        return "et"
    return "simulation"
