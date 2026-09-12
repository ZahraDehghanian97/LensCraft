"""Evaluation cohorts must remain subsets of the original training holdout."""

from pathlib import Path

import pytest
import torch

from data import datamodule
from data.datamodule import CameraTrajectoryDataModule
from data.simulation.dataset import SimulationDataset


@pytest.fixture
def inventory(monkeypatch, tmp_path):
    dataset = object.__new__(SimulationDataset)
    dataset.simulation_files = [Path(f"simulation_{i:03d}.msgpack") for i in range(100)]
    dataset.allowed_movement_types = []
    dataset.data_path = tmp_path
    dataset.parameter_dictionary = {}
    dataset.fixed_point_scale = 1000
    dataset.dataset_fingerprint = "test-inventory"
    movements = {
        path.name: "static" if i % 2 == 0 else "linear"
        for i, path in enumerate(dataset.simulation_files)
    }
    (tmp_path / "movement_types.txt").write_text(
        "".join(f"{name}|{movement}\n" for name, movement in movements.items())
    )
    monkeypatch.setattr(datamodule.hydra.utils, "instantiate", lambda _: dataset)
    monkeypatch.setattr(datamodule, "generate_movement_types_file", lambda *args: None)

    def no_sample_reads(*args):
        raise AssertionError("Cohort filtering must use metadata, not load trajectories")

    monkeypatch.setattr(SimulationDataset, "__getitem__", no_sample_reads)
    return dataset


def make_module(**kwargs):
    return CameraTrajectoryDataModule(
        dataset_config={"_target_": "data.simulation.dataset.SimulationDataset"},
        batch_size=8, num_workers=0, val_size=0.2, test_size=0.2, **kwargs,
    )


def test_cohorts_preserve_original_holdout_and_indices(inventory):
    original = make_module()
    original.setup()
    cohorts = []
    for movement in ("static", "linear"):
        torch.rand(17)  # Other model initialization must not change split membership.
        filtered = make_module(test_movement_types=[movement])
        filtered.setup()
        expected = [
            index for index in original.test_dataset.indices
            if (index % 2 == 0) == (movement == "static")
        ]
        assert filtered.train_dataset.indices == original.train_dataset.indices
        assert filtered.val_dataset.indices == original.val_dataset.indices
        assert filtered.test_dataset.dataset is inventory
        assert filtered.test_dataset.indices == expected
        filtered.setup()
        assert filtered.test_dataset.indices == expected
        cohorts.append(set(expected))
    assert cohorts[0].isdisjoint(cohorts[1])
    assert cohorts[0] | cohorts[1] == set(original.test_dataset.indices)
    assert (cohorts[0] | cohorts[1]).isdisjoint(original.train_dataset.indices)
    assert (cohorts[0] | cohorts[1]).isdisjoint(original.val_dataset.indices)


def test_rejects_prefiltered_inventory(inventory):
    with pytest.raises(ValueError, match="full dataset inventory"):
        CameraTrajectoryDataModule(
            dataset_config={
                "_target_": "data.simulation.dataset.SimulationDataset",
                "allowed_movement_types": ["static"],
            },
            batch_size=8, test_movement_types=["static"],
        )
    inventory.allowed_movement_types = ["static"]
    with pytest.raises(ValueError, match="unfiltered SimulationDataset"):
        make_module(test_movement_types=["static"]).setup()


@pytest.mark.parametrize("movements", [[], ["missing"]])
def test_rejects_empty_cohort(inventory, movements):
    with pytest.raises(ValueError, match="No held-out samples"):
        make_module(test_movement_types=movements).setup()


def test_rejects_unsupported_dataset(monkeypatch):
    monkeypatch.setattr(datamodule.hydra.utils, "instantiate", lambda _: list(range(100)))
    with pytest.raises(ValueError, match="only for SimulationDataset"):
        make_module(test_movement_types=["static"]).setup()


@pytest.mark.parametrize("fraction, expected_count", [(1.0, 20), (0.1, 2), (0.075, 2), (0.001, 1)])
def test_fraction_selects_original_holdout_prefix(inventory, fraction, expected_count):
    original = make_module()
    original.setup()
    torch.rand(17)
    fractional = make_module(test_fraction=fraction)
    fractional.setup()
    assert fractional.test_dataset.dataset is inventory
    assert fractional.test_dataset.indices == original.test_dataset.indices[:expected_count]
    assert fractional.train_dataset.indices == original.train_dataset.indices
    assert fractional.val_dataset.indices == original.val_dataset.indices
    assert fractional.original_test_sample_count == 20
    assert fractional.fractional_test_sample_count == expected_count
    fractional.setup()
    assert fractional.test_dataset.indices == original.test_dataset.indices[:expected_count]


def test_fraction_precedes_movement_cohorts(inventory):
    original = make_module()
    original.setup()
    prefix = original.test_dataset.indices[:2]
    # Guarantee both movements in this tiny 10% fixture without loading samples.
    (inventory.data_path / "movement_types.txt").write_text(
        "".join(
            f"{path.name}|{'static' if index == prefix[0] else 'linear'}\n"
            for index, path in enumerate(inventory.simulation_files)
        )
    )
    cohorts = []
    for movement in ("static", "linear"):
        filtered = make_module(test_fraction=0.1, test_movement_types=[movement])
        filtered.setup()
        assert filtered.original_test_sample_count == 20
        assert filtered.fractional_test_sample_count == 2
        assert filtered.train_dataset.indices == original.train_dataset.indices
        assert filtered.val_dataset.indices == original.val_dataset.indices
        cohorts.append(set(filtered.test_dataset.indices))
    assert cohorts == [{prefix[0]}, {prefix[1]}]
    assert cohorts[0] | cohorts[1] == set(prefix)
    assert (cohorts[0] | cohorts[1]).isdisjoint(original.train_dataset.indices)
    assert (cohorts[0] | cohorts[1]).isdisjoint(original.val_dataset.indices)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_test_fraction(fraction):
    with pytest.raises(ValueError, match="test_fraction must be finite"):
        make_module(test_fraction=fraction)
