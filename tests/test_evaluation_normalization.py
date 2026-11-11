"""Generator scales must be installed before an evaluation dataset is prepared."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import msgpack
import pytest
import torch
from omegaconf import OmegaConf

from data.datamodule import CameraTrajectoryDataModule
from data.simulation import loader
from data.simulation.dataset import SimulationDataset
from utils.normalization import prepare_evaluation_normalization


@pytest.fixture(autouse=True)
def isolated_normalization(monkeypatch):
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH", raising=False)
    monkeypatch.delenv("SIMULATION_DATA_PATH", raising=False)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_cache", {})
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_override", None)


def scales(mean):
    return {group: {"mean": torch.full((3,), float(mean)), "std": torch.full((3,), 10.)}
            for group in ("camera_position", "subject_position", "subject_dimensions")}


def export(path, offset=0.):
    path.mkdir()
    (path / "parameter_dictionary.msgpack").write_bytes(msgpack.packb(
        {"keys": ["cinematography__movement__type"], "values": [["static"]]}, use_bin_type=True))
    (path / "manifest.json").write_text(json.dumps({"config": {"subjectCount": 1, "instructionCount": 1}}))
    for index in range(10):
        camera = [[offset + index + frame, 2., 3., 0., 0., 0., 35., 16 / 9] for frame in range(4)]
        subject = [[0., 1., 0., 0., 0., 0.] for _ in range(4)]
        payload = [[[0, 0]], [], [{"i": "subject", "c": "human", "d": [.5, 1.7, .3],
                                    "m": "static", "f": subject}], camera]
        (path / f"simulation_{index:03d}.msgpack").write_bytes(msgpack.packb(payload, use_bin_type=True))
    return path


def datamodule(path, **extra):
    return CameraTrajectoryDataModule(
        {"_target_": "data.simulation.dataset.SimulationDataset", "data_path": str(path),
         "target_frame_count": 4, **extra},
        batch_size=2, num_workers=0, val_size=.2, test_size=.2,
    )


def test_saved_scales_are_used_without_fitting_evaluation_cameras(tmp_path, monkeypatch):
    path = export(tmp_path / "evaluation", offset=1000.)
    module = datamodule(path)
    parameters = scales(500.)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("evaluation cameras must not fit checkpoint scales")

    monkeypatch.setattr(loader, "calculate_normalization_parameters_tensor", forbidden_fit)
    provenance = prepare_evaluation_normalization(
        module, SimpleNamespace(normalization_parameters=parameters), "lens_craft")
    module.setup("test")
    dataset = module.test_dataset.dataset
    torch.testing.assert_close(dataset.normalization_parameters["camera_position"]["mean"],
                               parameters["camera_position"]["mean"])
    sample_index = module.test_dataset.indices[0]
    sample = module.test_dataset[0]
    torch.testing.assert_close(sample["camera_trajectory"][0, :3],
                               (torch.tensor([1000. + sample_index, 2., 3.]) - 500.) / 10.)
    assert not (path / "normalization_parameters.json").exists()
    assert provenance == {"source": "generator_checkpoint", "training_statistics_restored": True}


def test_checkpoint_scales_override_a_different_reference_without_reading_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(tmp_path / "missing_reference"))
    module = datamodule(export(tmp_path / "evaluation"))
    parameters = scales(123.)
    prepare_evaluation_normalization(module, SimpleNamespace(normalization_parameters=parameters), "lens_craft")
    module.setup("test")
    torch.testing.assert_close(module.state_dict()["normalization"]["camera_position"]["mean"],
                               parameters["camera_position"]["mean"])


def test_legacy_checkpoint_warns_and_keeps_full_inventory_transform(tmp_path):
    path = export(tmp_path / "legacy")
    module = datamodule(path)
    with pytest.warns(RuntimeWarning, match="historical full simulation inventory"):
        provenance = prepare_evaluation_normalization(module, SimpleNamespace(), "lens_craft")
    module.setup("test")
    # Ten clips with mean x=0..9 and four frames offset by 0..3.
    assert module.state_dict()["normalization"]["camera_position"]["mean"][0].item() == 6.
    assert provenance["source"] == "legacy_full_inventory"
    assert not provenance["training_statistics_restored"]


def test_legacy_checkpoint_honors_explicit_reference(tmp_path, monkeypatch):
    path = export(tmp_path / "evaluation", offset=1000.)
    reference = export(tmp_path / "training_reference", offset=100.)
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(reference))
    module = datamodule(path)
    with pytest.warns(RuntimeWarning, match="explicit SIMULATION_NORMALIZATION_DATA_PATH"):
        provenance = prepare_evaluation_normalization(module, SimpleNamespace(), "lens_craft")
    module.setup("test")
    assert module.state_dict()["normalization"]["camera_position"]["mean"][0].item() == 106.
    assert not (path / "normalization_parameters.json").exists()
    assert provenance["source"] == "legacy_reference"


def test_saved_scales_cannot_be_silently_disabled_and_baselines_are_separate(tmp_path):
    module = datamodule(tmp_path, normalize=False)
    model = SimpleNamespace(normalization_parameters=scales(500.))
    with pytest.raises(ValueError, match="normalize=true"):
        prepare_evaluation_normalization(module, model, "lens_craft")
    assert prepare_evaluation_normalization(module, model, "ccdm") is None
    assert module.state_dict() == {}


def entrypoint(name):
    spec = importlib.util.spec_from_file_location(f"_lenscraft_{name}_entrypoint", Path(__file__).parents[1] / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["test", "inference", "efficiency"])
def test_entrypoints_restore_model_scales_before_dataset_setup(name, monkeypatch):
    module = entrypoint(name)
    model = SimpleNamespace(normalization_parameters=scales(400.))
    events = []

    class Prepared(Exception):
        pass

    class DataModule:
        def __init__(self, dataset_config, **kwargs):
            self.dataset_config = dataset_config

        def load_state_dict(self, state):
            assert state["normalization"] is model.normalization_parameters
            events.append("normalization")

        def setup(self):
            assert events == ["model", "normalization"]
            raise Prepared

    def load_model(*args, **kwargs):
        events.append("model")
        return model

    cfg = OmegaConf.create({
        "data": {"dataset": {"config": {"_target_": "data.simulation.dataset.SimulationDataset"},
                             "info": {"seq_length": 4}},
                 "batch_size": 2, "num_workers": 0, "val_size": .2, "test_size": .2},
        "training": {"model": {"data_format": {"type": "simulation"}}},
    })
    monkeypatch.setattr(module, "CameraTrajectoryDataModule", DataModule)
    monkeypatch.setattr(module, "_resolve_device", lambda cfg: torch.device("cpu"))
    if name == "test":
        monkeypatch.setattr(module, "_load_eval_models", lambda *args: (load_model(), object()))
    else:
        monkeypatch.setattr(module, "_load_model" if name == "inference" else "load_model", load_model)
    with pytest.raises(Prepared):
        module.main.__wrapped__(cfg)


def test_cache_invalidates_lenscraft_generation_and_reference_changes(tmp_path, monkeypatch):
    from testing import trajectory_cache as cache

    reference = export(tmp_path / "reference")
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", str(reference))
    config = {"training": {"model": {"data_format": {"type": "simulation"}}}}
    original = cache.build_trajectory_cache_key(config)
    monkeypatch.setattr(cache, "LENSCRAFT_GENERATION_VERSION", cache.LENSCRAFT_GENERATION_VERSION + 1)
    assert cache.build_trajectory_cache_key(config) != original
    changed_version = cache.build_trajectory_cache_key(config)
    (reference / "manifest.json").write_text('{"modified": true}')
    assert cache.build_trajectory_cache_key(config) != changed_version


def script(name):
    spec = importlib.util.spec_from_file_location(f"_lenscraft_{name}_script", Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_caption_preservation_invalidates_ccdm_generation_cache_only(monkeypatch):
    from testing import trajectory_cache as cache

    ccdm = {"training": {"model": {"data_format": {"type": "ccdm"}}}}
    ours = {"training": {"model": {"data_format": {"type": "simulation"}}}}
    before_ccdm = cache.build_trajectory_cache_key(ccdm)
    before_ours = cache.build_trajectory_cache_key(ours)
    monkeypatch.setattr(cache, 'CCDM_GENERATION_VERSION', cache.CCDM_GENERATION_VERSION + 1)
    assert cache.build_trajectory_cache_key(ccdm) != before_ccdm
    assert cache.build_trajectory_cache_key(ours) == before_ours


def test_paper_timing_restores_checkpoint_scales_before_reading_batch(monkeypatch):
    from data import datamodule as data
    from models import factory

    runner = script("run_paper_supplement")
    model = SimpleNamespace(normalization_parameters=scales(700.))
    events = []

    class Prepared(Exception):
        pass

    class DataModule:
        def __init__(self, dataset_config, **kwargs):
            self.dataset_config = dataset_config

        def load_state_dict(self, state):
            assert state["normalization"] is model.normalization_parameters
            events.append("normalization")

        def setup(self):
            assert events == ["model", "normalization"]
            raise Prepared

    def load_model(*args):
        events.append("model")
        return model

    monkeypatch.setattr(data, "CameraTrajectoryDataModule", DataModule)
    monkeypatch.setattr(factory, "load_model", load_model)
    cfg = OmegaConf.create({"device": "cpu", "test_fraction": 1.,
        "data": {"dataset": {"config": {"_target_": "data.simulation.dataset.SimulationDataset"}},
                 "val_size": .2, "test_size": .2},
        "training": {"model": {"data_format": {"type": "simulation"}}}})
    with pytest.raises(Prepared):
        runner._timing_inputs(cfg, SimpleNamespace(eff_batch_size=2))


def test_robustness_cohort_restores_selected_scales_before_setup(tmp_path, monkeypatch):
    from data import datamodule as data

    runner = script("evaluate_et_robustness")
    parameters = scales(900.)

    class Prepared(Exception):
        pass

    class DataModule:
        def __init__(self, **kwargs):
            self.restored = None

        def load_state_dict(self, state):
            self.restored = state["normalization"]

        def setup(self):
            assert self.restored is parameters
            raise Prepared

    monkeypatch.setattr(data, "CameraTrajectoryDataModule", DataModule)
    cfg = OmegaConf.create({"data": {"dataset": {"config": {
        "_target_": "data.simulation.dataset.SimulationDataset", "normalize": True}},
        "batch_size": 2, "val_size": .2, "test_size": .2}})
    args = SimpleNamespace(batch_size=2)
    with pytest.raises(Prepared):
        runner.evaluate_cohort(args, cfg, object(), "simulation", tmp_path / "cohort", parameters)


def test_robustness_main_uses_saved_scales_without_reading_training_export(tmp_path, monkeypatch):
    from models import factory

    runner = script("evaluate_et_robustness")
    parameters = scales(800.)
    model = SimpleNamespace(normalization_parameters=parameters, keyframe_pose_conditioning=False)
    cfg = OmegaConf.create({
        "data": {"dataset": {"config": {"_target_": "data.simulation.dataset.SimulationDataset",
                                         "data_path": str(tmp_path / "unavailable_training_export")}}},
        "training": {"model": {"inference": {"checkpoint_path": "unused", "config": None}}}})
    train_config = tmp_path / "training.yaml"
    OmegaConf.save(cfg, train_config)
    et_config = tmp_path / "evaluation.yaml"
    et_config.write_text("{}")
    args = SimpleNamespace(self_test=False, repo_root=tmp_path, train_config=train_config,
        et_config=et_config, checkpoint=tmp_path / "checkpoint.ckpt", output=tmp_path / "output",
        simulation_config=None, device="cpu")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda argv: args)
    monkeypatch.setattr(runner, "bind_environment", lambda args: None)
    monkeypatch.setattr(runner, "sha256", lambda path: "mocked-file-hash")
    monkeypatch.setattr(factory, "load_model", lambda *args: model)
    observed = []

    def evaluate(args, cfg, model, dataset_type, output, normalization):
        observed.append(normalization)
        return {"sample_count": 0}

    monkeypatch.setattr(runner, "evaluate_cohort", evaluate)
    runner.main([])
    torch.testing.assert_close(observed[0]["camera_position"]["mean"], parameters["camera_position"]["mean"])
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["normalization"]["source"] == "generator_checkpoint"
    assert not (tmp_path / "unavailable_training_export").exists()
