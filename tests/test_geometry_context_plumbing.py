"""Geometry conditions must follow the same observations as model poses."""
from types import SimpleNamespace

import pytest
import torch

from data.sim_format import geometry_context_from_batch, to_simulation_format
from generation import generate_lenscraft_trajectory
from training.base_trainer import BaseTrainer, MaskConfig, NoiseConfig, TeacherForcingConfig


def observation(frames=4, batch_size=2):
    return {
        "camera_trajectory": torch.zeros(batch_size, frames, 6),
        "subject_trajectory": torch.zeros(batch_size, frames, 6),
        "subject_volume": torch.ones(batch_size, 1, 3),
        "padding_mask": torch.zeros(batch_size, frames, dtype=torch.bool),
    }


def test_reference_optics_and_availability_follow_visible_reference_only():
    batch = observation()
    batch["camera_intrinsics"] = torch.tensor([24., 1.5]).expand(2, 4, 2)
    reference = observation(frames=6)
    reference["camera_intrinsics"] = torch.tensor([70., 2.]).expand(2, 6, 2)
    reference["subject_orientation_available"] = torch.zeros(2, dtype=torch.bool)
    reference["subject_dimensions_available"] = torch.zeros(2, dtype=torch.bool)
    # A metric's complete physical subject must not become a model condition.
    reference["geometry_reference"] = dict(observation(frames=6),
        camera_intrinsics=torch.tensor([99., 3.]).expand(2, 6, 2))
    batch["simulation_reference"] = reference
    view = to_simulation_format(batch, "simulation", target_len=6)
    context = geometry_context_from_batch(batch, target_len=6)
    assert view[0].shape[1] == context["camera_intrinsics"].shape[1] == 6
    torch.testing.assert_close(context["camera_intrinsics"], reference["camera_intrinsics"])
    assert not context["subject_orientation_available"].any()
    assert not context["subject_dimensions_available"].any()


def test_optics_resampling_uses_valid_prefix_and_retains_missing_measurements():
    batch = observation()
    batch["padding_mask"][0, 2:] = True
    batch["padding_mask"][1] = True
    batch["camera_intrinsics"] = torch.tensor([
        [[24., 1.5], [70., 2.], [999., 8.], [999., 8.]],
        [[35., 1.5]] * 4,
    ])
    context = geometry_context_from_batch(batch, target_len=3)
    torch.testing.assert_close(context["camera_intrinsics"][0],
        torch.tensor([[24., 1.5], [47., 1.75], [70., 2.]]))
    assert context["camera_intrinsics"][1].isnan().all()
    batch["camera_intrinsics"][0, 1, 0] = float("nan")
    sampled = geometry_context_from_batch(batch, target_len=3)["camera_intrinsics"][0]
    assert sampled[0, 0] == 24.  # Exact measured frame remains available.
    assert sampled[1:, 0].isnan().all()
    assert torch.isfinite(sampled[:, 1]).all()


def test_temporal_orientation_availability_is_resampled_conservatively():
    batch = observation(frames=2)
    batch["subject_orientation_available"] = torch.tensor([[True, False], [True, True]])
    context = geometry_context_from_batch(batch, target_len=3)
    assert context["subject_orientation_available"].tolist() == [[True, False, False], [True, True, True]]


@pytest.mark.parametrize("dataset_type", ["et", "ccdm", "gendop"])
def test_native_unknown_subject_and_converter_defaults_never_become_measurements(dataset_type):
    batch = observation()
    batch["intrinsics"] = torch.ones(2, 4, 4)  # Native ET projection units.
    context = geometry_context_from_batch(batch, dataset_type, target_len=3)
    assert "camera_intrinsics" not in context
    assert not context["subject_orientation_available"].any()
    assert not context["subject_dimensions_available"].any()
    assert geometry_context_from_batch(observation()) == {}


@pytest.mark.parametrize("bad", [torch.tensor([-1., 1.5]), torch.tensor([float("inf"), 1.5])])
def test_invalid_measured_optics_are_rejected(bad):
    batch = observation()
    batch["camera_intrinsics"] = bad.expand(2, 4, 2)
    with pytest.raises(ValueError, match="positive and finite"):
        geometry_context_from_batch(batch)


def test_generation_and_training_forward_exact_geometry_context():
    class Recorder(torch.nn.Module):
        def forward(self, **kwargs):
            self.kwargs = kwargs
            return {"reconstructed": kwargs["src"]}

        def generate_camera_trajectory(self, **kwargs):
            self.kwargs = kwargs
            return {"reconstructed": kwargs["camera_trajectory"]}

    batch = observation()
    batch["camera_intrinsics"] = torch.tensor([35., 16. / 9.]).expand(2, 4, 2)
    batch["subject_orientation_available"] = torch.tensor([True, False])
    batch["subject_dimensions_available"] = torch.tensor([False, True])
    context = geometry_context_from_batch(batch)
    model = Recorder()
    view = tuple(batch[key] for key in ("camera_trajectory", "subject_trajectory", "subject_volume", "padding_mask"))
    generate_lenscraft_trajectory(model, view, batch["padding_mask"], None, 0., geometry_context=context)
    for key, value in context.items():
        assert model.kwargs[key] is value
    trainer = BaseTrainer(model, None, None, None, NoiseConfig(0., 0.), MaskConfig(0., 0.),
                          TeacherForcingConfig(0., 0., 0., 0.), compile_enabled=False)
    trainer._trainer = SimpleNamespace(current_epoch=0, max_epochs=1)
    trainer._forward_step(*view[:3], None, view[3], conditioning_mode="key_framing",
                          keyframe_count=1, geometry_context=context)
    for key, value in context.items():
        assert model.kwargs[key] is value


def test_training_checkpoint_round_trip_preserves_instance_scales_and_migrates_legacy_weights(monkeypatch):
    from models.camera_trajectory_model import LensCraft
    from data.simulation.dataset import SimulationDataset

    parameters = {group: {"mean": torch.arange(3).float() + index, "std": torch.ones(3) * (index + 1)}
                  for index, group in enumerate(("camera_position", "subject_position", "subject_dimensions"))}
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters", None)
    monkeypatch.setattr(SimulationDataset, "_normalization_parameters_override", None)
    model = LensCraft(latent_dim=16, nhead=2, num_encoder_layers=1,
                      num_decoder_layers=1, dim_feedforward=32, dropout_rate=0., seq_length=4)
    trainer = BaseTrainer(model, None, None, None, NoiseConfig(0., 0.), MaskConfig(0., 0.),
                          TeacherForcingConfig(0., 0., 0., 0.), compile_enabled=False)
    class DataModule:
        def state_dict(self):
            return {"normalization": parameters}
        def load_state_dict(self, state):
            self.restored = state
    datamodule = DataModule()
    trainer._trainer = SimpleNamespace(datamodule=datamodule)
    checkpoint = {"state_dict": {key: value for key, value in trainer.state_dict().items()
                                 if not key.startswith("model.geometric_context.")}}
    trainer.on_save_checkpoint(checkpoint)
    for group in parameters:
        assert checkpoint["normalization"][group]["mean"] is not parameters[group]["mean"]
    with pytest.warns(RuntimeWarning, match="predates geometric context"):
        trainer.on_load_checkpoint(checkpoint)
    trainer.load_state_dict(checkpoint["state_dict"], strict=True)
    assert all(torch.count_nonzero(parameter) == 0 for parameter in model.geometric_context.parameters())
    for group in parameters:
        torch.testing.assert_close(SimulationDataset._normalization_parameters[group]["mean"], parameters[group]["mean"])
        torch.testing.assert_close(datamodule.restored["normalization"][group]["std"], parameters[group]["std"])


def test_padding_does_not_turn_unobserved_optics_into_a_validation_error():
    batch = observation()
    batch["camera_intrinsics"] = torch.tensor([35., 1.5]).expand(2, 4, 2).clone()
    batch["padding_mask"][0, 2:] = True
    batch["camera_intrinsics"][0, 2:] = torch.tensor([float("inf"), -1.])
    context = geometry_context_from_batch(batch)
    assert torch.isfinite(context["camera_intrinsics"][0, :2]).all()
    assert context["camera_intrinsics"][0, 2:].isnan().all()


def test_pre_geometry_optimizer_resume_gives_actionable_initialization_path():
    from models.camera_trajectory_model import LensCraft

    model = LensCraft(latent_dim=16, nhead=2, num_encoder_layers=1,
                      num_decoder_layers=1, dim_feedforward=32, dropout_rate=0., seq_length=4)
    trainer = BaseTrainer(model, None, None, None, NoiseConfig(0., 0.), MaskConfig(0., 0.),
                          TeacherForcingConfig(0., 0., 0., 0.), compile_enabled=False)
    state = {key: value for key, value in trainer.state_dict().items()
             if "geometric_context." not in key}
    checkpoint = {"state_dict": state, "optimizer_states": [{"param_groups": []}]}
    with pytest.raises(ValueError, match="initialize_from_checkpoint=<path> and resume_checkpoint=null"):
        trainer.on_load_checkpoint(checkpoint)
    assert checkpoint["state_dict"] is state
