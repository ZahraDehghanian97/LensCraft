"""Checkpoint scales must reach data preparation before any fit/setup can run."""
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import msgpack
import pytest
import torch

from data.multi_dataset_module import MultiDatasetModule
from data.simulation.dataset import SimulationDataset
from utils.normalization import prepare_evaluation_normalization
import train as entrypoint


@pytest.mark.parametrize("mode, with_normalization", [
    ("initialize_from_checkpoint", False), ("initialize_from_checkpoint", True),
    ("resume_checkpoint", False), ("resume_checkpoint", True), ("fresh", False),
])
def test_training_entry_restores_checkpoint_scales_before_fit(
    tmp_path, monkeypatch, mode, with_normalization,
):
    checkpoint_path = tmp_path / "weights.ckpt"
    parameters = {
        group: {"mean": [1.0, 2.0, 3.0], "std": [4.0, 5.0, 6.0]}
        for group in ("camera_position", "subject_position", "subject_dimensions")
    }
    legacy_parameters = {
        group: {"mean": [10.0, 20.0, 30.0], "std": [4.0, 5.0, 6.0]}
        for group in parameters
    }
    checkpoint = {"state_dict": {"model.weight": torch.ones(1)}}
    if with_normalization:
        checkpoint["normalization"] = parameters
    torch.save(checkpoint, checkpoint_path)
    directory = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        overrides = ["checkpoint_monitor=val_loss"]
        if mode != "fresh":
            overrides.append(f"{mode}={checkpoint_path}")
        cfg = compose(config_name="config", overrides=overrides)

    events = []
    loaded_paths = []
    real_load = torch.load

    def load(*args, **kwargs):
        loaded_paths.append(args[0])
        return real_load(*args, **kwargs)

    class DataModule:
        def __init__(self, **kwargs):
            self.state = None

        def load_state_dict(self, state):
            events.append("scales")
            self.state = state

    class Trainer:
        callback_metrics = {"val_loss": 0.25}

        def fit(self, module, datamodule, ckpt_path):
            events.append("fit")
            expected = parameters if with_normalization else legacy_parameters
            assert datamodule.state == ({"normalization": expected} if mode != "fresh" else None)
            assert ckpt_path == (str(checkpoint_path) if mode == "resume_checkpoint" else None)

    def instantiate(config, **kwargs):
        return Trainer() if config._target_ == "lightning.Trainer" else object()

    def initialize(path, model, device, *, checkpoint):
        events.append("weights")
        assert path == str(checkpoint_path)
        assert checkpoint["state_dict"]["model.weight"].item() == 1
        return model

    def legacy_normalization(datamodule, model, model_type):
        assert model_type == "lens_craft"
        events.append("legacy_normalization")
        datamodule.load_state_dict({"normalization": legacy_parameters})
        return {"source": "legacy_full_inventory", "training_statistics_restored": False}

    monkeypatch.setattr(entrypoint, "CameraTrajectoryDataModule", DataModule)
    monkeypatch.setattr(entrypoint, "instantiate", instantiate)
    monkeypatch.setattr(entrypoint, "load_checkpoint", initialize)
    monkeypatch.setattr(entrypoint, "prepare_evaluation_normalization", legacy_normalization)
    monkeypatch.setattr(torch, "load", load)
    assert entrypoint.main.__wrapped__(cfg) == 0.25
    assert loaded_paths == ([str(checkpoint_path)] if mode != "fresh" else [])
    if mode != "fresh":
        assert events.index("scales") < events.index("fit")
    assert ("legacy_normalization" in events) == (mode != "fresh" and not with_normalization)
    assert ("weights" in events) == (mode == "initialize_from_checkpoint")


@pytest.mark.parametrize("explicit_reference", [False, True])
def test_legacy_mixed_continuation_installs_full_inventory_or_reference_scales(
    tmp_path, monkeypatch, explicit_reference,
):
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH", raising=False)
    monkeypatch.delenv("SIMULATION_DATA_PATH", raising=False)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_cache", {})
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_override", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_frame_count", None)

    def export(path, offset):
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"config": {"subjectCount": 1,
                                                                  "instructionCount": 1}}))
        (path / "parameter_dictionary.msgpack").write_bytes(msgpack.packb({
            "keys": ["cinematography__movement__type"], "values": [["static"]],
        }, use_bin_type=True))
        for index in range(10):
            camera = [[offset + index + frame, 2., 3., 0., 0., 0., 35., 16 / 9]
                      for frame in range(4)]
            subject = [[0., 1., 0., 0., 0., 0.] for _ in range(4)]
            payload = [[[0, 0]], [], [{"i": "subject", "c": "human", "d": [.5, 1.7, .3],
                                       "m": "static", "f": subject}], camera]
            (path / f"simulation_{index:03d}.msgpack").write_bytes(
                msgpack.packb(payload, use_bin_type=True))
        return path

    inventory = export(tmp_path / "inventory", 0.)
    if explicit_reference:
        reference = export(tmp_path / "original_reference", 100.)
        monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(reference))
    module = MultiDatasetModule(
        simulation_config={"_target_": "data.simulation.dataset.SimulationDataset",
                           "data_path": str(inventory), "target_frame_count": 4},
        ccdm_config={}, batch_size=2, num_workers=0,
    )
    with pytest.warns(RuntimeWarning, match="original training scales cannot be verified"):
        provenance = prepare_evaluation_normalization(module, SimpleNamespace(), "lens_craft")
    parameters = module.state_dict()["normalization"]
    assert parameters["camera_position"]["mean"][0].item() == (106. if explicit_reference else 6.)
    assert provenance["source"] == ("legacy_reference" if explicit_reference else "legacy_full_inventory")
    assert not provenance["training_statistics_restored"]


def test_mixed_checkpoint_scales_are_authoritative_without_source_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_override", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_frame_count", None)
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(tmp_path / "missing_reference"))
    module = MultiDatasetModule(
        simulation_config={"_target_": "data.simulation.dataset.SimulationDataset",
                           "data_path": str(tmp_path / "missing_inventory")},
        ccdm_config={}, batch_size=2, num_workers=0,
    )
    parameters = {
        group: {"mean": [1., 2., 3.], "std": [4., 5., 6.]}
        for group in ("camera_position", "subject_position", "subject_dimensions")
    }
    provenance = prepare_evaluation_normalization(
        module, SimpleNamespace(normalization_parameters=parameters), "lens_craft")
    torch.testing.assert_close(module.state_dict()["normalization"]["camera_position"]["mean"],
                               torch.tensor([1., 2., 3.]))
    assert provenance == {"source": "generator_checkpoint", "training_statistics_restored": True}
