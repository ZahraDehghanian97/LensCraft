"""Holdout clips cannot influence fitted scales or change between stages."""

import json

import msgpack
import pytest
import torch

from data.datamodule import CameraTrajectoryDataModule, compact_simulation_collate_fn
from data.multi_dataset_module import MultiDatasetModule
from data.simulation import loader
from data.simulation.dataset import SimulationDataset


@pytest.fixture(autouse=True)
def isolated_normalization(monkeypatch):
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH", raising=False)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_cache", {})
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_override", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_frame_count", None)


def write_export(path, count=20, offset=0.):
    path.mkdir()
    dictionary = {"keys": ["cinematography__movement__type"], "values": [["static"]]}
    (path / "parameter_dictionary.msgpack").write_bytes(msgpack.packb(dictionary, use_bin_type=True))
    (path / "manifest.json").write_text(json.dumps({"config": {"subjectCount": 1, "instructionCount": 1}}))
    for index in range(count):
        base = float(index) + offset
        camera = [[base + frame, 2., 3., 0., 0., 0., 35., 16 / 9] for frame in range(4)]
        subject = [[base / 2, 1., 0., 0., 0., 0.] for _ in range(4)]
        payload = [[[0, 0]], [], [{"i": "subject", "c": "human", "d": [base + 1., 1.7, .3],
                                    "m": "static", "f": subject}], camera]
        (path / f"simulation_{index:03d}.msgpack").write_bytes(msgpack.packb(payload, use_bin_type=True))
    return path


@pytest.fixture
def export(tmp_path):
    return write_export(tmp_path / "simulation")


def configuration(path, **extra):
    return {"_target_": "data.simulation.dataset.SimulationDataset", "data_path": str(path),
            "target_frame_count": 4, **extra}


def module(path, **extra):
    return CameraTrajectoryDataModule(configuration(path), batch_size=4, num_workers=0,
                                      val_size=.2, test_size=.2, **extra)


def explicit_parameters(mean):
    return {group: {"mean": torch.full((3,), float(mean)), "std": torch.ones(3)}
            for group in ("camera_position", "subject_position", "subject_dimensions")}


def assert_parameters_equal(actual, expected):
    for group in expected:
        for statistic in expected[group]:
            torch.testing.assert_close(actual[group][statistic], expected[group][statistic])


def test_data_module_fits_only_training_and_reuses_dataset_across_stages(export, monkeypatch):
    fitted_files = []
    calculate = loader.calculate_normalization_parameters_tensor

    def record_fit(path, dictionary, files, *args):
        fitted_files.append(list(files))
        return calculate(path, dictionary, files, *args)

    monkeypatch.setattr(loader, "calculate_normalization_parameters_tensor", record_fit)
    data = module(export)
    rng_state = torch.random.get_rng_state()
    data.setup("fit")
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    dataset = data.train_dataset.dataset
    train_indices = data.train_dataset.indices
    assert len(fitted_files) == 1
    assert {path.name for path in fitted_files[0]} == {
        dataset.simulation_files[index].name for index in train_indices
    }
    assert data.val_dataset.dataset is dataset and data.test_dataset.dataset is dataset
    expected_mean = torch.tensor(train_indices, dtype=torch.float32).mean() + 1.5
    torch.testing.assert_close(dataset.normalization_parameters["camera_position"]["mean"][0], expected_mean)
    assert not torch.isclose(expected_mean, torch.tensor(11.))  # Whole-export mean.
    heldout_index = data.test_dataset.indices[0]
    raw_x = torch.arange(4, dtype=torch.float32) + heldout_index
    parameters = dataset.normalization_parameters["camera_position"]
    torch.testing.assert_close(data.test_dataset[0]["camera_trajectory"][:, 0],
                               (raw_x - parameters["mean"][0]) / parameters["std"][0])

    torch.rand(500)
    data.setup("validate")
    data.setup("test")
    assert data.train_dataset.dataset is dataset
    assert data.train_dataset.indices == train_indices
    assert len(fitted_files) == 1
    compact = compact_simulation_collate_fn([dataset[train_indices[0]]])
    assert compact["camera_intrinsics"].shape == (1, 4, 2)
    assert compact["simulation_normalized"] is True


def test_deferred_normalization_rejects_early_sample_reads_and_empty_training(export):
    dataset = SimulationDataset(str(export), defer_normalization=True, target_frame_count=4)
    assert not (export / "normalization_parameters.json").exists()
    with pytest.raises(RuntimeError, match="normalization is deferred"):
        dataset[0]
    with pytest.raises(ValueError, match="at least one training sample"):
        dataset.initialize_training_normalization([])


def test_normalization_cache_distinguishes_training_membership(export):
    first = SimulationDataset(str(export), defer_normalization=True, target_frame_count=4)
    first.initialize_training_normalization([0, 1])
    first_fingerprint = json.loads((export / "normalization_parameters.meta.json").read_text())["datasetFingerprint"]
    second = SimulationDataset(str(export), defer_normalization=True, target_frame_count=4)
    second.initialize_training_normalization([18, 19])
    second_fingerprint = json.loads((export / "normalization_parameters.meta.json").read_text())["datasetFingerprint"]
    assert first_fingerprint != second_fingerprint
    assert first.normalization_parameters["camera_position"]["mean"][0].item() == 2.
    assert second.normalization_parameters["camera_position"]["mean"][0].item() == 20.

    # The same membership in another order shares its scales, without inheriting
    # the most recently fitted split from the process-wide conversion cache.
    reordered = SimulationDataset(str(export), defer_normalization=True, target_frame_count=4)
    reordered.initialize_training_normalization([1, 0])
    assert_parameters_equal(reordered.normalization_parameters, first.normalization_parameters)


def test_independent_reference_is_honored_after_training_split(export, tmp_path, monkeypatch):
    reference = write_export(tmp_path / "reference", count=3, offset=100.)
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(reference))
    data = module(export)
    data.setup("fit")
    assert data.train_dataset.dataset.normalization_parameters["camera_position"]["mean"][0].item() == 102.5
    assert (reference / "normalization_parameters.json").exists()
    assert not (export / "normalization_parameters.json").exists()

    # Evaluation-only inventories have no local training clips to fit. A
    # separately declared reference still provides valid, independent scales.
    evaluation = CameraTrajectoryDataModule(configuration(export), batch_size=4, num_workers=0,
                                            val_size=0., test_size=1.)
    evaluation.setup("test")
    assert len(evaluation.train_dataset) == 0
    assert_parameters_equal(evaluation.test_dataset.dataset.normalization_parameters,
                            data.train_dataset.dataset.normalization_parameters)


def test_explicit_checkpoint_constructor_parameters_take_precedence_over_reference(export, monkeypatch):
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", "/missing/reference")
    expected = explicit_parameters(123.)
    dataset = SimulationDataset(str(export), normalization_parameters=expected, target_frame_count=4)
    assert_parameters_equal(dataset.normalization_parameters, expected)
    assert_parameters_equal(SimulationDataset.get_normalization_parameters(target_frame_count=4), expected)
    assert not (export / "normalization_parameters.json").exists()


@pytest.mark.parametrize("training_frame_count", [4, 100, 300])
def test_implicit_static_conversion_keeps_restored_training_scales(monkeypatch, tmp_path, training_frame_count):
    missing_reference = tmp_path / "missing-reference"
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(missing_reference))
    parameters = explicit_parameters(123.)
    installed = SimulationDataset.set_current_normalization_parameters(parameters, training_frame_count)
    camera = torch.tensor([[1., 2., 3., .1, .2, .3], [4., 5., 6., .4, .5, .6]])
    subject = camera + 1
    dimensions = torch.tensor([[.5, 1.7, .3]])

    # Conversion has no sampling-resolution argument. It must use exactly the
    # installed transform, including when a reference path is unavailable here.
    normalized = SimulationDataset.normalize_item(camera, subject, dimensions)
    torch.testing.assert_close(normalized[0][:, :3], camera[:, :3] - 123.)
    torch.testing.assert_close(normalized[0][:, 3:], camera[:, 3:])
    torch.testing.assert_close(normalized[1][:, :3], subject[:, :3] - 123.)
    torch.testing.assert_close(normalized[2], dimensions - 123.)
    restored = SimulationDataset.normalize_item(*normalized, normalize=False)
    for actual, expected in zip(restored, (camera, subject, dimensions)):
        torch.testing.assert_close(actual, expected)
    assert SimulationDataset.get_normalization_parameters() is installed

    # A caller that explicitly requests another normalization timeline still
    # needs that reference's statistics rather than silently reusing saved ones.
    with pytest.raises(ValueError, match="No simulation files"):
        SimulationDataset.get_normalization_parameters(target_frame_count=30)


@pytest.mark.parametrize("before_setup", [False, True])
def test_datamodule_restores_checkpoint_normalization_before_or_after_setup(export, before_setup):
    expected = explicit_parameters(123.)
    data = module(export)
    if before_setup:
        data.load_state_dict({"normalization": expected})
    data.setup("test")
    if not before_setup:
        data.load_state_dict({"normalization": expected})
    assert_parameters_equal(data.state_dict()["normalization"], expected)
    assert_parameters_equal(data.val_dataset.dataset.normalization_parameters, expected)
    assert_parameters_equal(data.test_dataset.dataset.normalization_parameters, expected)
    if before_setup:
        assert not (export / "normalization_parameters.json").exists()


def test_mixed_splits_use_independent_rng_and_fit_training_only(export, monkeypatch):
    from data import multi_dataset_module as mixed

    instantiate = mixed.hydra.utils.instantiate
    calls = []

    def instantiate_with_ccdm(config):
        calls.append(config["_target_"])
        if config["_target_"] == "test.CCDMDataset":
            torch.rand(100)  # Initialization must not perturb either split.
            return list(range(20))
        return instantiate(config)

    monkeypatch.setattr(mixed.hydra.utils, "instantiate", instantiate_with_ccdm)

    def make():
        return MultiDatasetModule(configuration(export), {"_target_": "test.CCDMDataset"},
                                  batch_size=4, num_workers=0, val_size=.2, test_size=.2)

    first = make()
    first.setup("fit")
    sim_membership = [first.sim_train.indices, first.sim_val.indices, first.sim_test.indices]
    ccdm_membership = [first.ccdm_train.indices, first.ccdm_val.indices, first.ccdm_test.indices]
    assert sim_membership != ccdm_membership
    for splits in (sim_membership, ccdm_membership):
        assert set(splits[0]).isdisjoint(splits[1])
        assert set(splits[0]).isdisjoint(splits[2])
        assert set(splits[1]).isdisjoint(splits[2])
        assert set().union(*map(set, splits)) == set(range(20))
    expected = torch.tensor(first.sim_train.indices, dtype=torch.float32).mean() + 1.5
    torch.testing.assert_close(first.sim_train.dataset.normalization_parameters["camera_position"]["mean"][0], expected)
    torch.rand(50)
    first.setup("test")
    assert len(calls) == 2
    second = make()
    second.setup("test")
    assert [second.sim_train.indices, second.sim_val.indices, second.sim_test.indices] == sim_membership
    assert [second.ccdm_train.indices, second.ccdm_val.indices, second.ccdm_test.indices] == ccdm_membership
    assert "multiprocessing_context" not in first._get_dataloader_kwargs()

    # Restoring after setup must update the shared data instance as well as the
    # conversion helpers; otherwise evaluation normalizes with different scales.
    saved = {"normalization": explicit_parameters(456.)}
    second.load_state_dict(saved)
    assert_parameters_equal(second.state_dict()["normalization"], saved["normalization"])
    assert_parameters_equal(second.sim_test.dataset.normalization_parameters, saved["normalization"])
