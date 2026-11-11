"""Editing restores current categorical weights and rejects superseded generators."""
from copy import deepcopy

from omegaconf import OmegaConf
import pytest
import torch

from models.camera_trajectory_model import LensCraft
from models.categorical_latent import from_ids
from testing.edit_checkpoint import load_edit_generator, read_edit_checkpoint
from utils.load_lens_craft import load_lens_craft_model


def normalization():
    return {group: {"mean": [0., 0., 0.], "std": [1., 1., 1.]}
            for group in ("camera_position", "subject_position", "subject_dimensions")}


@pytest.fixture
def experiment(tmp_path):
    torch.manual_seed(12)
    module = OmegaConf.create(dict(
        _target_="models.camera_trajectory_model.LensCraft", nhead=2,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32,
        dropout_rate=0., seq_length=5, latent_dim=16,
    ))
    base = LensCraft(**{key: value for key, value in module.items() if key != "_target_"}).eval()
    inference = OmegaConf.create(dict(checkpoint_path=str(tmp_path / "source.pt"), config=None))
    return module, inference, base, normalization()


def load(experiment, payload):
    module, inference, _, parameters = experiment
    torch.save(payload, inference.checkpoint_path)
    return load_edit_generator(module, inference, "cpu", normalization_parameters=parameters)


@pytest.mark.parametrize("layout", ["model", "state_dict", "compiled_state_dict"])
def test_current_standalone_and_lightning_checkpoints_restore_and_edit(experiment, layout):
    _, inference, base, _ = experiment
    prefix = "model._orig_mod." if layout == "compiled_state_dict" else "model."
    state = {((prefix + key) if layout != "model" else key): value
             for key, value in base.state_dict().items()}
    state_key = "state_dict" if layout == "compiled_state_dict" else layout
    model, provenance = load(experiment, {state_key: state})
    assert type(model) is LensCraft and not model.training
    assert provenance["source_format"] == state_key
    assert provenance["semantic_shape"] == [10, "batch", 21]
    for name, value in model.state_dict().items():
        assert torch.equal(value, base.state_dict()[name])
    caption, presence = from_ids(torch.zeros((1, 10), dtype=torch.long))
    edits = torch.zeros((1, 10), dtype=torch.bool)
    edits[:, 4] = True
    output = model.generate_camera_trajectory(
        camera_trajectory=torch.linspace(-.3, .5, 30).reshape(1, 5, 6),
        caption_embedding=caption, caption_presence_mask=presence, caption_edit_mask=edits,
    )
    assert output["reconstructed"].shape == (1, 5, 6)
    assert torch.isfinite(output["reconstructed"]).all()
    assert set(read_edit_checkpoint(inference.checkpoint_path)[state_key]) == set(state)


def test_preloaded_payload_avoids_second_read(experiment, monkeypatch):
    module, inference, base, parameters = experiment
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Checkpoint read twice"))
    model, _ = load_edit_generator(module, inference, "cpu", normalization_parameters=parameters,
                                   checkpoint_payload={"model": base.state_dict()})
    assert type(model) is LensCraft


@pytest.mark.parametrize("layout", ["model", "state_dict", "model_state_dict"])
def test_legacy_clip_id_and_wrapper_weights_fail_with_useful_error(experiment, layout):
    old = {"encoder.query_tokens": torch.ones(46, 1, 512)}
    with pytest.raises(ValueError, match="not a ten-token categorical"):
        load(experiment, {layout: old})


@pytest.mark.parametrize("mutation", ["unexpected", "missing"])
def test_current_schema_does_not_silently_drop_missing_or_extra_weights(experiment, mutation):
    state = deepcopy(experiment[2].state_dict())
    if mutation == "unexpected":
        state["unused_adapter.weight"] = torch.ones(1)
    else:
        state.pop("subject_volume_projection.weight")
    with pytest.raises(RuntimeError):
        load(experiment, {"model": state})


def test_saved_categorical_config_migrates_only_redundant_options_without_resolver(experiment, tmp_path):
    module, inference, base, _ = experiment
    saved = OmegaConf.create({"training": {"model": {"module": OmegaConf.to_container(module)}}})
    obsolete = dict(conditioning_representation="categorical", use_merged_memory=False,
                    denormalize_memory=False, caption_presence_conditioning=False,
                    camera_memory_norms="${forbidden_clip_resolver:}")
    for key, value in obsolete.items():
        saved.training.model.module[key] = value
    inference.config = str(tmp_path / "saved.yaml")
    OmegaConf.save(saved, inference.config)
    model, _ = load(experiment, {"model": base.state_dict()})
    assert not hasattr(model, "camera_memory_norms")
    assert "conditioning_representation" not in module
    restored = load_lens_craft_model(module, inference, "cpu")
    assert torch.equal(restored.encoder.semantic_projection.weight, base.encoder.semantic_projection.weight)


def test_explicit_old_representation_is_rejected_even_with_categorical_weights(experiment):
    experiment[0].conditioning_representation = "clip"
    with pytest.raises(ValueError, match="not a ten-token categorical"):
        load(experiment, {"model": experiment[2].state_dict()})


def test_optional_saved_normalization_is_verified(experiment):
    payload = {"model": experiment[2].state_dict(), "normalization": normalization()}
    _, provenance = load(experiment, payload)
    assert provenance["normalization_verified"]
    payload["normalization"]["camera_position"]["mean"][0] = 1.
    with pytest.raises(ValueError, match="normalization mismatch"):
        load(experiment, payload)
