import math
import os
import hydra
import lightning as L
from torch import Generator
from torch.utils.data import random_split, DataLoader, Subset
from data.simulation.dataset import SimulationDataset, collate_fn
from data.simulation.loader import generate_movement_types_file, load_movement_types
from data.et.dataset import collate_fn as et_collate_fn
from data.ccdm.dataset import collate_fn as ccdm_collate_fn


def compact_simulation_collate_fn(batch):
    """Keep model inputs while dropping metadata before worker transfer/pinning."""
    result = collate_fn(batch)
    for key in (
        "raw_prompt",
        "raw_instruction",
        "text_prompts",
        "simulation_instruction_parameters",
        "cinematography_prompt_parameters",
        "original_frame_count",
        "camera_intrinsics",
        "simulation_normalized",
    ):
        result.pop(key, None)
    return result


class CameraTrajectoryDataModule(L.LightningDataModule):
    def __init__(self, dataset_config, batch_size, num_workers=None, val_size=0.1, test_size=0.1, split_seed=42, compact_batches=False, test_movement_types=None, test_fraction=1.0):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.split_seed = split_seed
        self.test_fraction = float(test_fraction)
        if not math.isfinite(self.test_fraction) or not 0 < self.test_fraction <= 1:
            raise ValueError("test_fraction must be finite and in (0, 1]")
        if isinstance(test_movement_types, str):
            raise ValueError("test_movement_types must be a list of movement names")
        self.test_movement_types = (
            None if test_movement_types is None else list(test_movement_types)
        )
        if self.test_movement_types is not None and dataset_config.get("allowed_movement_types"):
            raise ValueError(
                "test_movement_types requires the full dataset inventory; remove "
                "data.dataset.config.allowed_movement_types to preserve the held-out split"
            )
        
        self.num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
        
        if 'ETDataset' in self.dataset_config['_target_']:
            self.dataset_mode = 'et'
            self.collate_fn = et_collate_fn
        elif 'CCDMDataset' in self.dataset_config['_target_']:
            self.dataset_mode = 'ccdm'
            self.collate_fn = ccdm_collate_fn
        else:
            self.dataset_mode = 'simulation'
            self.collate_fn = collate_fn

        self.train_val_collate_fn = (
            compact_simulation_collate_fn
            if compact_batches and self.dataset_mode == 'simulation'
            else self.collate_fn
        )

    def setup(self, stage=None):
        full_dataset = hydra.utils.instantiate(self.dataset_config)
        
        if hasattr(full_dataset, 'preprocess'):
            full_dataset.preprocess()

        train_size = int((1 - self.val_size - self.test_size) * len(full_dataset))
        val_size = int(self.val_size * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            # Model initialization and repeated setup calls must not change splits.
            generator=Generator().manual_seed(self.split_seed),
        )
        self.original_test_sample_count = len(self.test_dataset)
        if self.test_fraction < 1:
            sample_count = max(1, math.ceil(self.original_test_sample_count * self.test_fraction))
            self.test_dataset = Subset(full_dataset, self.test_dataset.indices[:sample_count])
        self.fractional_test_sample_count = len(self.test_dataset)
        if self.test_movement_types is not None:
            self._filter_test_movement_types(full_dataset)

    def _filter_test_movement_types(self, full_dataset):
        """Intersect movement metadata with the original held-out indices only."""
        if not isinstance(full_dataset, SimulationDataset):
            raise ValueError("test_movement_types is supported only for SimulationDataset")
        if full_dataset.allowed_movement_types:
            raise ValueError("test_movement_types requires an unfiltered SimulationDataset")

        generate_movement_types_file(
            full_dataset.data_path,
            full_dataset.simulation_files,
            full_dataset.parameter_dictionary,
            full_dataset.fixed_point_scale,
            full_dataset.dataset_fingerprint,
        )
        movement_types = load_movement_types(full_dataset.data_path)
        allowed = set(self.test_movement_types)
        indices = [
            index for index in self.test_dataset.indices
            if movement_types[full_dataset.simulation_files[index].name] in allowed
        ]
        if not indices:
            raise ValueError(
                "No held-out samples match test_movement_types="
                f"{self.test_movement_types!r}"
            )
        self.test_dataset = Subset(full_dataset, indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_val_collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_val_collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
