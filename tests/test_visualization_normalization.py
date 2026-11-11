"""Qualitative comparisons use each generator's persisted coordinate scales."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from data.simulation import loader
from data.simulation.dataset import SimulationDataset
from test_evaluation_normalization import export, scales
from visualization.data import ComparisonSample, VisualizationRepository, _sim_standard


@pytest.fixture(autouse=True)
def isolated_normalization(monkeypatch):
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH", raising=False)
    monkeypatch.delenv("SIMULATION_DATA_PATH", raising=False)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_cache", {})
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_override", None)


def handle_for(path):
    return {
        "dataset": "simulation", "split": "all", "data_path": str(path),
        "native": {"files": sorted(path.glob("simulation_*.msgpack"))},
        "config": OmegaConf.create({"data": {"dataset": {"config": {
            "_target_": "data.simulation.dataset.SimulationDataset", "data_path": str(path),
            "target_frame_count": 4, "normalization_frame_count": 4,
        }}}}),
    }


def test_visualization_reads_saved_scales_without_fitting_inventory(tmp_path, monkeypatch):
    path = export(tmp_path / "evaluation", offset=1000.)
    repository = VisualizationRepository(device="cpu")
    parameters = scales(500.)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("visualization must not fit evaluation camera statistics")

    monkeypatch.setattr(loader, "calculate_normalization_parameters_tensor", forbidden_fit)
    batch = repository._batch(handle_for(path), 0, normalization_parameters=parameters)
    torch.testing.assert_close(batch["camera_trajectory"][0, 0, :3],
                               (torch.tensor([1000., 2., 3.]) - 500.) / 10.)
    assert not (path / "normalization_parameters.json").exists()


def test_visualization_cache_separates_scales_and_reactivates_cached_conversion(tmp_path):
    path = export(tmp_path / "evaluation", offset=1000.)
    handle = handle_for(path)
    repository = VisualizationRepository(device="cpu")
    first_scales, second_scales = scales(500.), scales(100.)
    first = repository._batch(handle, 0, normalization_parameters=first_scales)
    second = repository._batch(handle, 0, normalization_parameters=second_scales)
    assert len(repository._inference_datasets) == 2
    assert not torch.equal(first["camera_trajectory"], second["camera_trajectory"])
    equivalent = {group: {name: value.clone() for name, value in values.items()}
                  for group, values in first_scales.items()}
    cached = repository._batch(handle, 0, normalization_parameters=equivalent)
    assert len(repository._inference_datasets) == 2
    torch.testing.assert_close(cached["camera_trajectory"], first["camera_trajectory"])
    torch.testing.assert_close(SimulationDataset._normalization_parameters["camera_position"]["mean"],
                               first_scales["camera_position"]["mean"])
    # Both checkpoints' inputs represent the same physical selected sample.
    first_world = _sim_standard(first["camera_trajectory"], normalized=True,
                                normalization_parameters=first_scales)[0]
    second_world = _sim_standard(second["camera_trajectory"], normalized=True,
                                 normalization_parameters=second_scales)[0]
    np.testing.assert_allclose(first_world, second_world, atol=1e-4)


def test_visualization_builds_batch_after_loading_each_generator_and_denormalizes_its_output(tmp_path, monkeypatch):
    path = export(tmp_path / "evaluation", offset=1000.)
    handle = handle_for(path)
    repository = VisualizationRepository(device="cpu")
    configuration = OmegaConf.create({"training": {"model": {
        "data_format": {"seq_length": 4}, "inference": {"checkpoint_path": "saved.pt"},
    }}})
    monkeypatch.setattr(repository, "_config", lambda *args, **kwargs: configuration)
    expected_world = torch.zeros(1, 4, 6)
    expected_world[..., :3] = torch.tensor([1000., 2., 3.])
    sample = ComparisonSample("simulation_000", "simulation", "static", {"GT": _sim_standard(expected_world)[0][0]},
                              _context=(repository, handle, 0))
    events = []

    class Generator:
        def __init__(self, parameters):
            self.normalization_parameters = parameters

        def generate_camera_trajectory(self, **kwargs):
            events.append("generation")
            expected = (torch.tensor([1000., 2., 3.]) - self.normalization_parameters["camera_position"]["mean"]) / 10.
            torch.testing.assert_close(kwargs["camera_trajectory"][0, 0, :3], expected)
            return {"reconstructed": kwargs["camera_trajectory"]}

    model = Generator(scales(500.))
    def load_model(*args):
        events.append("model")
        return model
    monkeypatch.setattr("models.factory.load_model", load_model)
    original_batch = repository._batch
    def build_batch(*args, **kwargs):
        assert events[-1] == "model"
        assert kwargs["normalization_parameters"] is model.normalization_parameters
        events.append("batch")
        return original_batch(*args, **kwargs)
    monkeypatch.setattr(repository, "_batch", build_batch)
    output = repository.generate(sample, mode="reconstruction")
    assert events == ["model", "batch", "generation"]
    assert output.metadata["errors"] == {}
    np.testing.assert_allclose(output.trajectories["lens_craft:reconstruction"][0, :3, 3], [1000., 2., 3.], atol=1e-4)
