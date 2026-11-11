"""Frozen judges score physical motion independently of generator scales."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from data.simulation.dataset import SimulationDataset
from testing.metrics.normalization import MetricNormalization, load_metric_normalization
from testing.metrics.semantic_evaluator import FrozenSemanticEvaluator
from testing.metrics.native_clatr_extractor import NativeCLaTrFeatureExtractor
from testing.metrics import clatr_extractor as et_clatr
from testing.process import _metric_normalization_kwargs
from training.clatr.lightning_module import LightningCLaTr
from train_clatr import _prepare_resume_normalization


def scales(offset, scale):
    return {group: {"mean": torch.full((3,), float(offset + index)),
                    "std": torch.full((3,), float(scale))}
            for index, group in enumerate(("camera_position", "subject_position", "subject_dimensions"))}


def normalize(values, parameters):
    return SimulationDataset._normalize_item_with_parameters(*values, parameters)


def physical_sample():
    camera = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6)
    subject = camera + 2
    volume = torch.full((2, 1, 3), 4.)
    return camera, subject, volume


def test_semantic_features_invariant_to_generator_scales(monkeypatch):
    judge = FrozenSemanticEvaluator(latent_dim=8, nhead=2, num_encoder_layers=1,
                                     dim_feedforward=16, seq_length=3)
    judge.metric_normalization = MetricNormalization(scales(16, 4))
    physical = physical_sample()
    first, second = scales(0, 2), scales(32, 8)
    initial = scales(-10, 16)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", initial)
    mask = torch.tensor([[False, False, True], [False, False, False]])
    before = {name: value.clone() for name, value in judge.state_dict().items()}
    outputs = [judge.embed_trajectory(*normalize(physical, source), src_key_mask=mask,
                                     source_normalization=source)
               for source in (first, second)]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    physical_output = judge.embed_trajectory(*physical, src_key_mask=mask,
                                             camera_is_normalized=False,
                                             subject_is_normalized=False)
    torch.testing.assert_close(outputs[0], physical_output, rtol=0, atol=0)
    for name, value in judge.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert SimulationDataset._normalization_parameters is initial


def test_native_clatr_features_invariant_to_generator_scales(monkeypatch):
    # Exercise the public adapter without loading weights, CLIP, or third party
    # network assets. The stand-in records the exact input consumed by a judge.
    extractor = NativeCLaTrFeatureExtractor.__new__(NativeCLaTrFeatureExtractor)
    torch.nn.Module.__init__(extractor)
    extractor.device = torch.device("cpu")
    extractor.traj_num_feats = 6
    extractor.metric_normalization = MetricNormalization(scales(16, 4))
    extractor.lit = SimpleNamespace(model=SimpleNamespace(
        encode=lambda data, **_: data["x"].sum(dim=1)))
    physical = physical_sample()
    first, second = scales(0, 2), scales(32, 8)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", first)
    a = extractor.encode_trajectory(normalize(physical, first)[0])
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", second)
    b = extractor.encode_trajectory(normalize(physical, second)[0])
    raw = extractor.encode_trajectory(physical[0], camera_is_normalized=False)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a, raw, rtol=0, atol=0)
    assert SimulationDataset._normalization_parameters is second


def test_et_clatr_receives_same_physical_motion_for_raw_and_normalized_cameras(monkeypatch):
    extractor = et_clatr.CLaTrFeatureExtractor.__new__(et_clatr.CLaTrFeatureExtractor)
    torch.nn.Module.__init__(extractor)
    extractor.device, extractor.num_cams = torch.device("cpu"), 3
    extractor.metric_normalization = MetricNormalization(normalized=False)
    extractor.clatr = SimpleNamespace(encode=lambda data, **_: data["x"].sum(dim=1))
    physical, source = physical_sample(), scales(0, 2)
    camera, subject, volume = normalize(physical, source)
    def convert(source_name, target_name, camera_value, subject_value, volume_value, padding, **kwargs):
        assert (source_name, target_name) == ("simulation", "et")
        assert kwargs["need_denormal"] is False
        for actual, expected in zip((camera_value, subject_value, volume_value), physical):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return camera_value, subject_value, volume_value, padding
    monkeypatch.setattr(et_clatr, "convert_to_target", convert)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", source)
    padding = torch.zeros((2, 3), dtype=torch.bool)
    normalized = extractor.encode_trajectory(camera, subject, volume, padding)
    mixed = extractor.encode_trajectory(physical[0], subject, volume, padding,
                                         camera_is_normalized=False)
    torch.testing.assert_close(normalized, mixed, rtol=0, atol=0)


def test_metric_calls_describe_reference_view_and_raw_baseline_camera():
    judge = SimpleNamespace(metric_normalization=MetricNormalization(scales(16, 4)))
    reference = {"camera_trajectory": torch.empty(1, 30, 6), "simulation_normalized": False}
    batch = {"simulation_normalized": True, "simulation_reference": reference}
    kwargs = _metric_normalization_kwargs(judge, batch, target_len=30)
    assert kwargs["camera_is_normalized"] is False and kwargs["subject_is_normalized"] is False
    kwargs = _metric_normalization_kwargs(judge, batch, target_len=300,
                                         camera_is_normalized=False)
    assert kwargs["camera_is_normalized"] is False and kwargs["subject_is_normalized"] is True
    assert _metric_normalization_kwargs(SimpleNamespace(), batch) == {}


def test_checkpoint_scales_take_precedence_without_touching_active_source(monkeypatch):
    source = scales(0, 2)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", source)
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", "/missing/reference")
    adapter = load_metric_normalization({"normalization": scales(16, 4)})
    assert adapter.provenance["source"] == "judge_checkpoint"
    assert adapter.provenance["statistics_sha256"]
    assert SimulationDataset._normalization_parameters is source


def test_direct_semantic_checkpoint_loader_restores_its_own_transform():
    model = FrozenSemanticEvaluator(latent_dim=8, nhead=2, num_encoder_layers=1,
                                    dim_feedforward=16, seq_length=3)
    model.load_metric_checkpoint({"state_dict": model.state_dict(), "normalization": scales(16, 4)})
    assert model.metric_normalization.provenance["source"] == "judge_checkpoint"
    torch.testing.assert_close(model.metric_normalization.parameters["camera_position"]["mean"],
                               torch.full((3,), 16.))


def test_legacy_reference_reads_own_scales_and_restores_all_current_state(monkeypatch):
    source, target = scales(0, 2), scales(16, 4)
    monkeypatch.setenv("SIMULATION_NORMALIZATION_DATA_PATH", "/historical/reference")
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", source)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_override", ("source", 17))
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_frame_count", 17)
    def independent_reference(**kwargs):
        assert kwargs == {"data_path": "/historical/reference", "target_frame_count": 30}
        SimulationDataset._normalization_parameters = target
        SimulationDataset._normalization_parameters_override = ("judge", 30)
        SimulationDataset._normalization_parameters_frame_count = 30
        return target
    monkeypatch.setattr(SimulationDataset, "get_normalization_parameters", independent_reference)
    with pytest.warns(RuntimeWarning, match="historical reference"):
        adapter = load_metric_normalization({})
    assert adapter.provenance["source"] == "explicit_legacy_reference"
    assert SimulationDataset._normalization_parameters is source
    assert SimulationDataset._normalization_parameters_override == ("source", 17)
    assert SimulationDataset._normalization_parameters_frame_count == 17


def test_unverified_current_inventory_cannot_supply_historical_judge_scales(tmp_path, monkeypatch):
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH", raising=False)
    monkeypatch.setenv("SIMULATION_DATA_PATH", "/evaluation_only/inventory")
    config = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"data": {"dataset": {"config": {
        "_target_": "data.simulation.dataset.SimulationDataset",
        "data_path": "${oc.env:SIMULATION_DATA_PATH}",
    }}}}), config)
    with pytest.warns(RuntimeWarning, match="scoring is unavailable"):
        adapter = load_metric_normalization({}, saved_config_path=config)
    with pytest.raises(ValueError, match="Generator normalization cannot substitute"):
        adapter.convert(physical_sample()[0], source_normalization=scales(0, 2))
    assert not adapter.provenance["available"]


def test_clatr_checkpoint_hooks_persist_training_scales_without_replacing_source(monkeypatch):
    source, target = scales(0, 2), scales(16, 4)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", source)
    data = SimpleNamespace(state_dict=lambda: {"normalization": target}, dataset_mode="simulation",
                           _full_dataset=SimpleNamespace(normalize=True, normalization_frame_count=30))
    module = SimpleNamespace(_trainer=SimpleNamespace(datamodule=data), normalization_parameters=None,
                             normalization_context={})
    checkpoint = {"state_dict": {"_clip_embedder.x": 1, "model.x": 2}}
    LightningCLaTr.on_save_checkpoint(module, checkpoint)
    target["camera_position"]["mean"].fill_(99)
    assert checkpoint["normalization"]["camera_position"]["mean"].tolist() == [16.] * 3
    assert checkpoint["normalization_context"] == {"format": "simulation", "normalized": True, "frame_count": 30}
    assert checkpoint["state_dict"] == {"model.x": 2}
    frozen = SimpleNamespace(_trainer=None)
    LightningCLaTr.on_load_checkpoint(frozen, checkpoint)
    assert frozen.normalization_context == checkpoint["normalization_context"]
    assert SimulationDataset._normalization_parameters is source


def test_clatr_resume_restores_saved_scales_before_setup(tmp_path):
    from data.datamodule import CameraTrajectoryDataModule
    checkpoint = tmp_path / "judge.ckpt"
    torch.save({"normalization": scales(16, 4)}, checkpoint)
    # The dataset does not exist: restoring must happen without instantiation
    # or fitting, before Lightning starts setup/fit.
    data = CameraTrajectoryDataModule({"_target_": "data.simulation.dataset.SimulationDataset",
                                       "data_path": str(tmp_path / "missing"), "normalize": True},
                                      batch_size=1, num_workers=0)
    _prepare_resume_normalization(data, checkpoint)
    assert data._full_dataset is None
    torch.testing.assert_close(data.state_dict()["normalization"]["camera_position"]["mean"],
                               torch.full((3,), 16.))


@pytest.mark.parametrize("checkpoint_context,normalize", [({}, True), ({"normalized": False}, True),
                                                         ({"normalized": True}, False)])
def test_clatr_resume_rejects_unknown_or_different_training_coordinates(
    tmp_path, monkeypatch, checkpoint_context, normalize,
):
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH", raising=False)
    payload = {"normalization_context": checkpoint_context}
    if checkpoint_context.get("normalized") is True:
        payload["normalization"] = scales(16, 4)
    checkpoint = tmp_path / "judge.ckpt"
    torch.save(payload, checkpoint)
    data = SimpleNamespace(dataset_mode="simulation", dataset_config={"normalize": normalize},
                           load_state_dict=lambda _: pytest.fail("Unverified coordinates must not be installed"))
    if not checkpoint_context:
        with pytest.warns(RuntimeWarning, match="scoring is unavailable"):
            with pytest.raises(ValueError, match="Cannot resume CLaTr"):
                _prepare_resume_normalization(data, checkpoint)
    else:
        with pytest.raises(ValueError, match="original normalize setting"):
            _prepare_resume_normalization(data, checkpoint)


def test_clatr_resume_accepts_explicitly_unnormalized_checkpoint(tmp_path):
    checkpoint = tmp_path / "judge.ckpt"
    torch.save({"normalization_context": {"format": "simulation", "normalized": False}}, checkpoint)
    data = SimpleNamespace(dataset_mode="simulation", dataset_config={"normalize": False},
                           load_state_dict=lambda _: pytest.fail("Raw checkpoints have no coordinate scales"))
    _prepare_resume_normalization(data, checkpoint)
