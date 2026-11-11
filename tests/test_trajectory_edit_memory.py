"""Trajectory edits replace selected semantic slots without pinning source poses."""

import pytest
import torch

from models.categorical_latent import from_ids, masked_softmax, validate_semantic
from models.camera_trajectory_model import LensCraft


def tiny_model(**options):
    torch.manual_seed(17)
    return LensCraft(
        nhead=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout_rate=0., seq_length=5, latent_dim=16,
        **options,
    ).cpu().eval()


@pytest.fixture
def model():
    return tiny_model()


@pytest.fixture
def camera():
    return torch.linspace(-.3, .7, 60).reshape(2, 5, 6)


def memory_inputs(model):
    camera = masked_softmax(torch.linspace(-2., 2., 10 * 2 * 21).reshape(10, 2, 21))
    caption, _ = from_ids(torch.ones((2, 10), dtype=torch.long))
    edits = torch.zeros(2, 10, dtype=torch.bool)
    edits[0, [0, 4, 9]] = True
    edits[1, [1, 7, 8]] = True
    return camera, caption, edits


def generate(model, camera, **kwargs):
    return model.generate_camera_trajectory(
        camera_trajectory=camera,
        subject_trajectory=torch.zeros_like(camera),
        subject_volume=torch.ones(camera.shape[0], 1, 3),
        **kwargs,
    )


def test_per_example_edits_replace_only_selected_categorical_distributions(model):
    camera, caption, edits = memory_inputs(model)
    expected = torch.where(
        edits.T[..., None], caption, camera,
    )
    before = (camera.clone(), caption.clone(), edits.clone())

    for ratio in (0., .37, 1.):
        memory = model.prepare_embedding_memory_for_decoder(
            camera, caption, teacher_forcing_ratio=ratio, caption_edit_mask=edits,
        )
        torch.testing.assert_close(memory, expected, rtol=0, atol=0)

    for original, unchanged in zip((camera, caption, edits), before):
        assert torch.equal(original, unchanged)


@pytest.mark.parametrize("all_caption", [False, True])
def test_empty_and_full_edit_masks_match_existing_memory_boundaries(model, all_caption):
    camera, caption, _ = memory_inputs(model)
    edits = torch.full((2, model.num_query_tokens), all_caption, dtype=torch.bool)
    memory = model.prepare_embedding_memory_for_decoder(
        camera, caption, caption_edit_mask=edits,
    )
    expected = model.prepare_embedding_memory_for_decoder(
        None if all_caption else camera,
        caption if all_caption else None,
    )
    torch.testing.assert_close(memory, expected, rtol=0, atol=0)


def test_absent_nan_caption_fillers_never_enter_unedited_camera_fields():
    model = tiny_model()
    camera, caption, edits = memory_inputs(model)
    caption = caption.masked_fill(~edits.T[..., None], float("nan"))
    memory = model.prepare_embedding_memory_for_decoder(
        camera, caption, caption_edit_mask=edits, caption_presence_mask=edits,
    )
    expected = torch.where(edits.T[..., None], caption, camera)
    assert torch.isfinite(memory).all()
    torch.testing.assert_close(memory, expected, rtol=0, atol=0)


def test_semantic_padding_remains_zero_after_field_edit(model):
    camera, caption, edits = memory_inputs(model)
    memory = model.prepare_embedding_memory_for_decoder(camera, caption, caption_edit_mask=edits)
    validate_semantic(memory)
    assert memory.shape == (10, 2, 21)


@pytest.mark.parametrize("all_caption", [False, True])
def test_real_decoder_matches_reconstruction_and_caption_only_boundaries(model, camera, all_caption):
    _, caption, _ = memory_inputs(model)
    edits = torch.full((2, model.num_query_tokens), all_caption, dtype=torch.bool)
    edited = generate(model, camera, caption_embedding=caption, caption_edit_mask=edits)
    expected = generate(
        model, camera, caption_embedding=caption if all_caption else None,
        memory_teacher_forcing_ratio=1. if all_caption else 0.,
    )
    for key in ("reconstructed", "reconstructed_raw_matrix", "reconstructed_rot_matrix"):
        torch.testing.assert_close(edited[key], expected[key], rtol=0, atol=0)
    assert not edited["known_camera_mask"].any()


@pytest.mark.parametrize("api", ["forward", "generate"])
def test_editing_bypasses_direct_pose_queries_and_ignores_ratio(camera, api):
    model = tiny_model(keyframe_pose_conditioning=True)
    _, caption, edits = memory_inputs(model)
    captures = []

    def capture(module, args, kwargs):
        captures.append(kwargs)

    handle = model.decoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        kwargs = dict(caption_embedding=caption, caption_edit_mask=edits)
        if api == "generate":
            first = generate(model, camera, memory_teacher_forcing_ratio=0., **kwargs)
            second = generate(model, camera, memory_teacher_forcing_ratio=1., **kwargs)
        else:
            kwargs.update(src=camera, subject_trajectory=torch.zeros_like(camera), subject_volume=torch.ones(2, 1, 3))
            first = model(memory_teacher_forcing_ratio=0., **kwargs)
            second = model(memory_teacher_forcing_ratio=1., **kwargs)
    finally:
        handle.remove()

    assert len(captures) == 2
    assert all("known_camera_poses" not in item and "known_pose_mask" not in item for item in captures)
    assert not first["known_pose_mask"].any()
    assert not first["known_camera_mask"].any()
    assert not torch.equal(first["reconstructed"], camera)
    torch.testing.assert_close(first["reconstructed"], second["reconstructed"], rtol=0, atol=0)


def test_editing_preserves_temporal_padding_sanitization(model, camera):
    _, caption, edits = memory_inputs(model)
    padding = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    clean = generate(model, camera, caption_embedding=caption, caption_edit_mask=edits, padding_mask=padding)
    hidden_nan = generate(
        model, camera.masked_fill(padding[..., None], float("nan")),
        caption_embedding=caption, caption_edit_mask=edits, padding_mask=padding,
    )
    for key in ("embeddings", "reconstructed", "reconstructed_rot_matrix"):
        assert torch.isfinite(hidden_nan[key]).all()
        torch.testing.assert_close(hidden_nan[key], clean[key], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["float_mask", "wrong_batch", "wrong_fields", "pooled_caption", "wrong_camera", "absent_edit", "bad_presence"])
def test_invalid_edit_memory_arguments_fail_clearly(model, invalid):
    camera, caption, edits = memory_inputs(model)
    presence = torch.ones_like(edits)
    if invalid == "float_mask":
        edits = edits.float()
    elif invalid == "wrong_batch":
        edits = edits[:1]
    elif invalid == "wrong_fields":
        edits = edits[:, :-1]
    elif invalid == "pooled_caption":
        caption = caption[:1]
    elif invalid == "wrong_camera":
        camera = camera[:-1]
    elif invalid == "absent_edit":
        presence[0, 0] = False
    elif invalid == "bad_presence":
        presence = presence.float()
    with pytest.raises(ValueError):
        model.prepare_embedding_memory_for_decoder(
            camera, caption, caption_edit_mask=edits, caption_presence_mask=presence,
        )


@pytest.mark.parametrize("missing", ["camera", "caption"])
def test_editing_requires_both_camera_and_caption_memory(model, missing):
    camera, caption, edits = memory_inputs(model)
    with pytest.raises(ValueError, match="caption_edit_mask"):
        model.prepare_embedding_memory_for_decoder(
            None if missing == "camera" else camera,
            None if missing == "caption" else caption,
            caption_edit_mask=edits,
        )


def test_generation_requires_camera_and_rejects_fixed_keyframes(model, camera):
    _, caption, edits = memory_inputs(model)
    with pytest.raises(ValueError, match="requires camera_trajectory"):
        model.generate_camera_trajectory(caption_embedding=caption, caption_edit_mask=edits)
    with pytest.raises(ValueError, match="known_camera_mask"):
        generate(model, camera, caption_embedding=caption, caption_edit_mask=edits,
                 known_camera_mask=torch.ones(2, 5, dtype=torch.bool))


def test_editing_adds_no_state_dict_entries_and_preserves_strict_checkpoint_loading(model, camera):
    checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
    restored = tiny_model()
    restored.load_state_dict(checkpoint, strict=True)
    _, caption, edits = memory_inputs(model)
    generate(restored, camera, caption_embedding=caption, caption_edit_mask=edits)
    assert restored.state_dict().keys() == checkpoint.keys()
    for name, value in restored.state_dict().items():
        assert torch.equal(value, checkpoint[name])
