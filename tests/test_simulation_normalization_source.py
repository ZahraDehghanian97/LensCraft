"""A shared statistics source must never consume a new dataset's holdout."""

from pathlib import Path

import pytest

from data.simulation import dataset as simulation


@pytest.fixture
def normalization_sources(monkeypatch, tmp_path):
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH", raising=False)
    monkeypatch.setattr(simulation.SimulationDataset, "_normalization_parameters", None)
    monkeypatch.setattr(simulation.SimulationDataset, "_normalization_parameters_cache", {})
    monkeypatch.setattr(simulation.SimulationDataset, "_normalization_parameters_override", None)
    monkeypatch.setattr(simulation.SimulationDataset, "_normalization_parameters_frame_count", None)
    paths = {name: tmp_path / name for name in ("new", "reference", "other")}
    files = {name: [path / "simulation_000.msgpack"] for name, path in paths.items()}
    calls = []
    scans = []
    monkeypatch.setattr(simulation, "resolve_dataset_root", lambda path: Path(path))

    def find_files(path):
        scans.append(path)
        return files.get(path.name, [])

    monkeypatch.setattr(simulation, "find_simulation_files", find_files)
    monkeypatch.setattr(simulation, "load_dataset_manifest", lambda path: {"source": path.name})
    monkeypatch.setattr(simulation, "get_fixed_point_scale", lambda manifest: 2000)
    monkeypatch.setattr(
        simulation, "compute_dataset_fingerprint",
        lambda path, manifest, inventory: f"{manifest['source']}:{inventory[0].name}",
    )

    def load_parameters(path, **kwargs):
        calls.append((path, kwargs))
        return {"source": str(path), "frames": kwargs["target_frame_count"]}

    monkeypatch.setattr(simulation, "load_or_calculate_normalization_parameters", load_parameters)
    return paths, files, calls, scans


def test_default_preserves_dataset_metadata_and_fast_conversion_cache(normalization_sources):
    paths, files, calls, scans = normalization_sources
    dictionary = {"datasetId": "new"}
    parameters = simulation.SimulationDataset.get_normalization_parameters(
        paths["new"], parameter_dictionary=dictionary, simulation_files=files["new"],
        fixed_point_scale=1000, dataset_fingerprint="new-fingerprint",
    )
    assert calls == [(paths["new"], {
        "parameter_dictionary": dictionary, "simulation_files": files["new"],
        "fixed_point_scale": 1000, "dataset_fingerprint": "new-fingerprint:target_frames=30",
        "target_frame_count": 30,
    })]
    assert simulation.SimulationDataset.get_normalization_parameters() is parameters
    assert scans == []


def test_override_discards_new_dataset_metadata_and_is_shared_by_conversion(
    monkeypatch, normalization_sources,
):
    paths, files, calls, scans = normalization_sources
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(paths["reference"]))
    parameters = simulation.SimulationDataset.get_normalization_parameters(
        paths["new"], parameter_dictionary={"datasetId": "new"},
        simulation_files=files["new"], fixed_point_scale=1000, dataset_fingerprint="new-fingerprint",
    )
    assert calls == [(paths["reference"], {
        "parameter_dictionary": None, "simulation_files": files["reference"],
        "fixed_point_scale": 2000,
        "dataset_fingerprint": "reference:simulation_000.msgpack:target_frames=30",
        "target_frame_count": 30,
    })]
    for _ in range(3):
        assert simulation.SimulationDataset.get_normalization_parameters() is parameters
    assert scans == [paths["reference"]]
    # A separately initialized evaluation dataset shares the same cached stats.
    assert simulation.SimulationDataset.get_normalization_parameters(paths["other"]) is parameters
    assert len(calls) == 1


def test_changing_or_removing_override_cannot_reuse_previous_current_statistics(
    monkeypatch, normalization_sources,
):
    paths, _, calls, _ = normalization_sources
    monkeypatch.setenv("SIMULATION_DATA_PATH", str(paths["new"]))
    original = simulation.SimulationDataset.get_normalization_parameters()
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(paths["reference"]))
    reference = simulation.SimulationDataset.get_normalization_parameters()
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(paths["other"]))
    other = simulation.SimulationDataset.get_normalization_parameters()
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH")
    assert simulation.SimulationDataset.get_normalization_parameters() is original
    assert reference["source"] == str(paths["reference"])
    assert other["source"] == str(paths["other"])
    assert [path for path, _ in calls] == [paths[name] for name in ("new", "reference", "other")]


def test_override_cache_keeps_normalization_frame_counts_separate(monkeypatch, normalization_sources):
    paths, _, calls, _ = normalization_sources
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(paths["reference"]))
    native = simulation.SimulationDataset.get_normalization_parameters(target_frame_count=300)
    model = simulation.SimulationDataset.get_normalization_parameters(target_frame_count=30)
    assert native["frames"] == 300 and model["frames"] == 30
    assert len(calls) == 2
    assert simulation.SimulationDataset.get_normalization_parameters(target_frame_count=300) is native
    assert len(calls) == 2


def test_default_cache_keeps_explicit_frame_counts_separate(monkeypatch, normalization_sources):
    paths, _, calls, _ = normalization_sources
    monkeypatch.setenv("SIMULATION_DATA_PATH", str(paths["new"]))
    native = simulation.SimulationDataset.get_normalization_parameters(target_frame_count=300)
    assert simulation.SimulationDataset.get_normalization_parameters() is native
    model = simulation.SimulationDataset.get_normalization_parameters(target_frame_count=30)
    assert native["frames"] == 300 and model["frames"] == 30
    assert len(calls) == 2
    assert simulation.SimulationDataset.get_normalization_parameters(target_frame_count=300) is native
    assert len(calls) == 2


def test_missing_override_fails_instead_of_falling_back(monkeypatch, normalization_sources):
    paths, _, calls, _ = normalization_sources
    simulation.SimulationDataset.get_normalization_parameters(paths["new"])
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(paths["new"].parent / "missing"))
    with pytest.raises(ValueError, match="Normalization reference has no simulation files"):
        simulation.SimulationDataset.get_normalization_parameters()
    assert len(calls) == 1
