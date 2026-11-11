"""LensCraft integration contracts for the bounded categorical representation."""
import math

import pytest
import torch

from models.camera_trajectory_model import LensCraft
from models.categorical_latent import (
    FIELD_WIDTHS, categorical_presence, from_ids, masked_softmax,
    valid_category_mask, validate_semantic,
)


def tiny_model(**options):
    torch.manual_seed(42)
    arguments = dict(input_dim=6, subject_dim=6, latent_dim=16, nhead=2,
                     num_encoder_layers=1, num_decoder_layers=1,
                     dim_feedforward=32, dropout_rate=0.0, seq_length=5)
    arguments.update(options)
    return LensCraft(**arguments).cpu().eval()


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(7)
    camera = torch.randn(2, 5, 6, generator=generator) / 5
    camera[..., :3] += 10
    subject = torch.randn(2, 5, 6, generator=generator) / 5
    volume = torch.rand(2, 1, 3, generator=generator) + 0.5
    return camera, subject, volume


def caption(missing=()):
    ids = torch.tensor([[width - 1 for width in FIELD_WIDTHS], [0] * 10])
    for batch, field in missing:
        ids[batch, field] = -1
    return from_ids(ids)


def assert_same_predictions(left, right):
    for name in ("embeddings", "reconstructed", "reconstructed_raw_matrix",
                 "reconstructed_rot_matrix"):
        if name in left and name in right:
            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)




def test_default_model_is_the_single_categorical_architecture():
    model = LensCraft().cpu()
    assert model.subject_trajectory_projection.out_features == 256
    assert len(model.encoder.transformer_encoder.layers) == 4
    assert len(model.decoder.transformer_decoder.layers) == 4
    for layer in model.encoder.transformer_encoder.layers:
        assert layer.self_attn.embed_dim == 256
        assert layer.self_attn.num_heads == 8
        assert layer.linear1.out_features == 1024
    for layer in model.decoder.transformer_decoder.layers:
        assert layer.self_attn.embed_dim == 256
        assert layer.self_attn.num_heads == 8
        assert layer.linear1.out_features == 1024
        assert layer.multihead_attn.kdim == layer.multihead_attn.vdim == 21
    assert model.num_query_tokens == 10
    assert not hasattr(model, "conditioning_representation")
    assert not hasattr(model, "embedding_merger")
    assert not any(name.startswith("embedding_merger.") for name in model.state_dict())


def test_categorical_checkpoint_roundtrip_keeps_exact_state_and_predictions(tmp_path, inputs):
    model = tiny_model()
    checkpoint = tmp_path / "categorical_state.pt"
    torch.save(model.state_dict(), checkpoint)
    loaded = tiny_model()
    result = loaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert set(model.state_dict()) == set(loaded.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value), name
    assert_same_predictions(model(*inputs), loaded(*inputs))


def test_categorical_forward_exposes_only_93_semantics_and_allocates_no_merger(inputs):
    model = tiny_model()
    seen = []
    hook = model.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["memory"].detach().clone()),
        with_kwargs=True,
    )
    try:
        output = model(*inputs, memory_teacher_forcing_ratio=0.0)
    finally:
        hook.remove()
    probabilities, logits = output["embeddings"], output["embedding_logits"]
    validate_semantic(probabilities)
    assert probabilities.shape == logits.shape == (10, 2, 21)
    valid = valid_category_mask()[:, None, :].expand_as(probabilities)
    assert valid_category_mask().sum().item() == 93
    assert (~valid_category_mask()).sum().item() == 117
    assert torch.equal(probabilities[~valid], torch.zeros_like(probabilities[~valid]))
    assert torch.isneginf(logits[~valid]).all()
    torch.testing.assert_close(probabilities, logits.softmax(-1))
    torch.testing.assert_close(seen[0], probabilities, atol=0, rtol=0)
    assert model.num_query_tokens == model.memory_tokens_count == 10
    assert model.semantic_dim == 93 and model.semantic_token_width == 21
    assert model.decoder.transformer_decoder.layers[0].multihead_attn.kdim == 21
    assert model.decoder.transformer_decoder.layers[0].multihead_attn.vdim == 21
    assert not hasattr(model, "embedding_merger")
    assert not any(name.startswith("embedding_merger.") for name in model.state_dict())
    assert not any(name.startswith("embedding_merger.") for name, _ in model.named_parameters())
    assert not hasattr(model, "structured_slot_encoding")
    assert not hasattr(model, "missing_caption_encoding")




def test_caption_only_forward_and_pure_prompt_generation_have_no_camera_dependency(inputs):
    model = tiny_model()
    camera, subject, volume = inputs
    values, presence = caption()
    first = model(camera, subject, volume, caption_embedding=values,
                  caption_presence_mask=presence, memory_teacher_forcing_ratio=1.0)
    changed = model(camera.flip(1) * -31, subject, volume, caption_embedding=values,
                    caption_presence_mask=presence, memory_teacher_forcing_ratio=1.0)
    generated = model.generate_camera_trajectory(
        caption_embedding=values, subject_trajectory=subject, subject_volume=volume,
        caption_presence_mask=presence)
    for name in ("reconstructed", "reconstructed_raw_matrix", "reconstructed_rot_matrix"):
        torch.testing.assert_close(first[name], changed[name], atol=0, rtol=0)
        torch.testing.assert_close(first[name], generated[name], atol=0, rtol=0)
    # Caption-only memory must not even perform arithmetic on an unused,
    # potentially invalid camera semantic tensor.
    memory = model.prepare_embedding_memory_for_decoder(
        camera_embedding=torch.full_like(values, float("nan")),
        caption_embedding=values, teacher_forcing_ratio=1.0)
    assert torch.equal(memory, values)


def test_unknown_zero_fields_are_dontcare_and_an_all_unknown_prompt_is_finite(inputs):
    model = tiny_model()
    _, subject, volume = inputs
    values, presence = from_ids(torch.full((2, 10), -1, dtype=torch.long))
    assert not presence.any()
    poisoned = torch.full_like(values, float("nan"))
    zero = model.generate_camera_trajectory(caption_embedding=values,
        subject_trajectory=subject, subject_volume=volume)
    ignored = model.generate_camera_trajectory(caption_embedding=poisoned,
        caption_presence_mask=presence, subject_trajectory=subject, subject_volume=volume)
    assert_same_predictions(zero, ignored)
    assert torch.isnan(poisoned).all()
    for name in ("reconstructed", "reconstructed_raw_matrix", "reconstructed_rot_matrix"):
        assert torch.isfinite(zero[name]).all()
    attention, padding = model.decoder.prepare_attention_memory(values)
    assert torch.equal(attention, torch.zeros_like(attention))
    assert torch.equal((~padding).sum(1), torch.ones(2, dtype=torch.long))


@pytest.mark.parametrize("ratio", [0.25, 1e-8, 1 - 1e-8])
def test_partial_blends_keep_unit_field_mass_and_retain_available_information(ratio):
    model = tiny_model()
    generator = torch.Generator().manual_seed(7)
    camera = masked_softmax(torch.randn(10, 2, 21, generator=generator))
    camera[1, 0] = 0
    camera[3, 0] = 0
    values, presence = caption(missing=((1, 2), (0, 3)))
    memory = model.prepare_embedding_memory_for_decoder(
        camera_embedding=camera, caption_embedding=values,
        caption_presence_mask=presence, teacher_forcing_ratio=ratio)
    validate_semantic(memory)
    torch.testing.assert_close(memory[0], (1 - ratio) * camera[0] + ratio * values[0])
    torch.testing.assert_close(memory[1, 0], values[1, 0])
    torch.testing.assert_close(memory[2, 1], camera[2, 1])
    assert torch.equal(memory[3, 0], torch.zeros(21))
    expected_presence = categorical_presence(camera) | presence
    torch.testing.assert_close(memory.sum(-1).T, expected_presence.float())


def test_field_editing_changes_only_selected_semantics_and_ignores_other_fillers(inputs):
    model = tiny_model(keyframe_pose_conditioning=True)
    camera, subject, volume = inputs
    camera_semantics = model.embed_trajectory(camera, subject, volume)
    values, _ = caption()
    edits = torch.zeros(2, 10, dtype=torch.bool)
    edits[0, 4] = True
    edits[1, 0] = True
    poisoned = values.masked_fill(~edits.T[..., None], float("nan"))
    original_camera = camera_semantics.clone()
    expected = torch.where(edits.T[..., None], values, camera_semantics)
    memory = model.prepare_embedding_memory_for_decoder(
        camera_embedding=camera_semantics, caption_embedding=poisoned,
        caption_edit_mask=edits, caption_presence_mask=torch.ones_like(edits),
        teacher_forcing_ratio=1.0)
    assert torch.equal(memory, expected)
    assert torch.equal(camera_semantics, original_camera)
    assert torch.isnan(poisoned.masked_select(~edits.T[..., None])).all()
    seen = []
    hook = model.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(dict(kwargs)), with_kwargs=True)
    try:
        model(camera, subject, volume, caption_embedding=poisoned,
              caption_edit_mask=edits, caption_presence_mask=torch.ones_like(edits))
    finally:
        hook.remove()
    torch.testing.assert_close(seen[0]["memory"], expected, atol=0, rtol=0)
    assert seen[0].get("known_camera_poses") is None
    assert seen[0].get("known_pose_mask") is None


def test_dropping_every_memory_field_stays_finite(inputs):
    model = tiny_model()
    values, presence = caption()
    seen = []
    hook = model.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["memory"].detach().clone()),
        with_kwargs=True)
    try:
        output = model(*inputs, caption_embedding=values, caption_presence_mask=presence,
                       memory_teacher_forcing_ratio=0.5, mask_memory_prob=1.0)
    finally:
        hook.remove()
    assert torch.equal(seen[0], torch.zeros_like(seen[0]))
    for name in ("reconstructed", "reconstructed_raw_matrix", "reconstructed_rot_matrix"):
        assert torch.isfinite(output[name]).all()


@pytest.mark.parametrize("pose_conditioning", [False, True])
def test_explicit_keyframes_are_exact_and_hidden_nan_fillers_preserve_gradients(inputs, pose_conditioning):
    model = tiny_model(keyframe_pose_conditioning=pose_conditioning)
    camera, subject, volume = inputs
    camera = camera.clone()
    camera[0, 0, 3] += 2 * math.pi
    known = torch.tensor([[True, False, False, True, False],
                          [False, True, False, False, True]])
    zero = camera.masked_fill(~known[..., None], 0).requires_grad_()
    first = model(zero, subject, volume, known_camera_mask=known)
    first["reconstructed_raw_matrix"].square().mean().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    missing = camera.masked_fill(~known[..., None], float("nan")).requires_grad_()
    second = model(missing, subject, volume, known_camera_mask=known)
    assert_same_predictions(first, second)
    assert torch.equal(second["reconstructed"][known], camera[known])
    expected_matrix = model._euler6_to_mat12(camera)[known]
    for name in ("reconstructed_raw_matrix", "reconstructed_rot_matrix"):
        assert torch.equal(second[name][known], expected_matrix)
    second["reconstructed_raw_matrix"].square().mean().backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, gradients[name], atol=0, rtol=0)
    assert torch.isfinite(missing.grad).all()
    assert torch.equal(missing.grad[~known], torch.zeros_like(missing.grad[~known]))
    assert model.encoder.semantic_projection.weight.grad.abs().sum() > 0


def test_keyframe_flag_alone_never_exposes_all_source_poses_to_decoder(inputs):
    model = tiny_model(keyframe_pose_conditioning=True)
    seen = []
    hook = model.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(dict(kwargs)), with_kwargs=True)
    try:
        flagged = model(*inputs, memory_teacher_forcing_ratio=0.0)
    finally:
        hook.remove()
    assert seen[0].get("known_camera_poses") is None
    assert seen[0].get("known_pose_mask") is None
    assert not flagged["known_camera_mask"].any()
    assert not flagged["known_pose_mask"].any()
    model.keyframe_pose_conditioning = False
    ordinary = model(*inputs, memory_teacher_forcing_ratio=0.0)
    assert_same_predictions(flagged, ordinary)


def test_embed_trajectory_masks_subject_frames_independently_of_hidden_camera(inputs):
    model = tiny_model()
    camera, subject, volume = inputs
    hidden_camera = torch.ones(2, 5, dtype=torch.bool)
    missing_camera = torch.full_like(camera, float("nan"))
    visible_subject = torch.zeros(2, 5, dtype=torch.bool)
    implicit = model.embed_trajectory(missing_camera, subject, volume, hidden_camera)
    explicit = model.embed_trajectory(missing_camera, subject, volume, hidden_camera,
                                      subject_key_padding_mask=visible_subject)
    torch.testing.assert_close(implicit, explicit, atol=0, rtol=0)
    changed = model.embed_trajectory(missing_camera, subject.flip(1) * 17, volume, hidden_camera)
    assert not torch.equal(implicit, changed)
    subject_mask = torch.tensor([[False, False, True, True, True],
                                 [False, True, False, True, True]])
    zeros = subject.masked_fill(subject_mask[..., None], 0)
    poisoned = subject.masked_fill(subject_mask[..., None], float("nan"))
    first = model.embed_trajectory(missing_camera, zeros, volume, hidden_camera,
                                   subject_key_padding_mask=subject_mask)
    second = model.embed_trajectory(missing_camera, poisoned, volume, hidden_camera,
                                    subject_key_padding_mask=subject_mask)
    assert torch.isfinite(second).all()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
