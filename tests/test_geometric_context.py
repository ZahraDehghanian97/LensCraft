"""Geometry observations must affect generation independently of semantic labels."""
from copy import deepcopy

from omegaconf import OmegaConf
import pytest
import torch

from models.camera_trajectory_model import LensCraft
from models.categorical_latent import from_ids
from utils.load_lens_craft import load_categorical_generator


@pytest.fixture
def example():
    torch.manual_seed(17)
    module = dict(_target_='models.camera_trajectory_model.LensCraft', latent_dim=16,
                  nhead=2, num_encoder_layers=1, num_decoder_layers=1,
                  dim_feedforward=32, seq_length=5, dropout_rate=0.)
    model = LensCraft(**{key: value for key, value in module.items() if key != '_target_'}).eval()
    camera = torch.randn(2, 5, 6) / 5
    subject = torch.randn(2, 5, 6) / 5
    volume = torch.ones(2, 1, 3)
    caption, presence = from_ids(torch.zeros(2, 10, dtype=torch.long))
    return model, module, camera, subject, volume, caption, presence


def generate(example, **context):
    model, _, _, subject, volume, caption, presence = example
    return model.generate_camera_trajectory(
        caption_embedding=caption, caption_presence_mask=presence,
        subject_trajectory=subject, subject_volume=volume, **context)


@pytest.mark.parametrize('component,changed', [(0, 70.), (1, 9 / 16)])
def test_lens_and_aspect_change_prompt_generation_and_encoder_context(example, component, changed):
    model, _, camera, subject, volume, _, _ = example
    optics = torch.tensor([24., 16 / 9]).expand(2, 5, 2).clone()
    different = optics.clone()
    different[..., component] = changed
    first = generate(example, camera_intrinsics=optics)
    second = generate(example, camera_intrinsics=different)
    assert not torch.allclose(first['reconstructed_raw_matrix'], second['reconstructed_raw_matrix'])
    one = model.embed_trajectory(camera, subject, volume, camera_intrinsics=optics)
    two = model.embed_trajectory(camera, subject, volume, camera_intrinsics=different)
    assert not torch.allclose(one, two)
    assert one.shape == two.shape == (10, 2, 21)


def test_forward_and_direct_prompt_api_use_identical_geometric_context(example):
    model, _, camera, subject, volume, caption, presence = example
    context = dict(camera_intrinsics=torch.tensor([[24., 1.], [70., 16 / 9]]),
                   subject_orientation_available=torch.tensor([True, False]),
                   subject_dimensions_available=torch.tensor([False, True]))
    direct = generate(example, **context)
    forward = model(camera, subject, volume, caption_embedding=caption,
                    caption_presence_mask=presence, memory_teacher_forcing_ratio=1., **context)
    for name in ('reconstructed', 'reconstructed_raw_matrix', 'reconstructed_rot_matrix'):
        torch.testing.assert_close(direct[name], forward[name], atol=0, rtol=0)


def test_unknown_subject_placeholders_are_ignored_and_distinct_from_measurements(example):
    model, _, _, subject, volume, caption, presence = example
    unknown = dict(subject_orientation_available=torch.zeros(2, dtype=torch.bool),
                   subject_dimensions_available=torch.zeros(2, dtype=torch.bool))
    first = generate(example, **unknown)
    poisoned_subject = subject.clone()
    poisoned_subject[..., 3:] = float('nan')
    second = model.generate_camera_trajectory(
        caption_embedding=caption, caption_presence_mask=presence,
        subject_trajectory=poisoned_subject, subject_volume=torch.full_like(volume, float('nan')),
        **unknown)
    torch.testing.assert_close(first['reconstructed_raw_matrix'], second['reconstructed_raw_matrix'], atol=0, rtol=0)
    measured = generate(example)
    assert not torch.allclose(first['reconstructed_raw_matrix'], measured['reconstructed_raw_matrix'])


def test_unobserved_optics_do_not_fabricate_a_default_lens(example):
    absent = generate(example)
    unknown = generate(example, camera_intrinsics=torch.full((2, 5, 2), float('nan')))
    known = generate(example, camera_intrinsics=torch.tensor([[50., 1.], [50., 1.]]))
    torch.testing.assert_close(absent['reconstructed_raw_matrix'], unknown['reconstructed_raw_matrix'], atol=0, rtol=0)
    assert not torch.allclose(unknown['reconstructed_raw_matrix'], known['reconstructed_raw_matrix'])
    partial = generate(example, camera_intrinsics=torch.tensor([[50., float('nan')]]).expand(2, -1))
    assert torch.isfinite(partial['reconstructed']).all()
    assert not torch.allclose(partial['reconstructed_raw_matrix'], unknown['reconstructed_raw_matrix'])


@pytest.mark.parametrize('value', [0., -1., float('inf')])
def test_invalid_measured_optics_are_rejected(example, value):
    with pytest.raises(ValueError, match='camera_intrinsics'):
        generate(example, camera_intrinsics=torch.tensor([[value, 1.]]).expand(2, -1))


def test_padded_context_and_temporally_unknown_orientation_cannot_contaminate_predictions(example):
    model, _, _, subject, volume, caption, _ = example
    padding = torch.tensor([[False, False, False, True, True]] * 2)
    availability = ~padding
    optics = torch.tensor([24., 1.]).expand(2, 5, 2).clone()
    baseline = model.generate_camera_trajectory(caption_embedding=caption, subject_trajectory=subject,
        subject_volume=volume, padding_mask=padding, camera_intrinsics=optics,
        subject_orientation_available=availability)
    poisoned = subject.masked_fill(padding[..., None], float('nan'))
    optics[padding] = float('inf')
    changed = model.generate_camera_trajectory(caption_embedding=caption, subject_trajectory=poisoned,
        subject_volume=volume, padding_mask=padding, camera_intrinsics=optics,
        subject_orientation_available=availability)
    torch.testing.assert_close(baseline['reconstructed'], changed['reconstructed'], atol=0, rtol=0)
    assert torch.isfinite(changed['reconstructed']).all()


@pytest.mark.parametrize('direct_pose', [False, True])
@pytest.mark.parametrize('decode_mode', ['single_step', 'autoregressive'])
@pytest.mark.parametrize('known_frames', [[True, False, False, False, True],
                                        [False, False, False, False, True]])
def test_explicit_keyframes_influence_unknown_frames_even_with_full_caption(example, direct_pose, decode_mode, known_frames):
    model, _, camera, subject, volume, caption, presence = example
    model.keyframe_pose_conditioning = direct_pose
    known = torch.tensor([known_frames] * 2)
    changed = camera.clone()
    changed[known] += .7
    args = dict(subject_trajectory=subject, subject_volume=volume,
                caption_embedding=caption, caption_presence_mask=presence,
                memory_teacher_forcing_ratio=1., known_camera_mask=known, decode_mode=decode_mode)
    first = model.generate_camera_trajectory(camera_trajectory=camera, **args)
    second = model.generate_camera_trajectory(camera_trajectory=changed, **args)
    assert torch.equal(first['reconstructed'][known], camera[known])
    assert torch.equal(second['reconstructed'][known], changed[known])
    assert not torch.allclose(first['reconstructed_raw_matrix'][~known], second['reconstructed_raw_matrix'][~known])


def test_geometry_context_learns_from_camera_loss(example):
    model, _, camera, subject, volume, caption, _ = example
    model.train()
    output = model(camera, subject, volume, caption_embedding=caption,
                   memory_teacher_forcing_ratio=1., camera_intrinsics=torch.tensor([[24., 1.]]).expand(2, -1),
                   subject_orientation_available=torch.zeros(2, dtype=torch.bool),
                   subject_dimensions_available=torch.zeros(2, dtype=torch.bool))
    output['reconstructed_raw_matrix'].square().mean().backward()
    for parameter in model.geometric_context.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_legacy_geometry_checkpoint_upgrade_is_explicit_and_partial_corruption_stays_strict(example):
    model, module, _, _, _, _, _ = example
    state = {name: tensor for name, tensor in model.state_dict().items()
             if not name.startswith('geometric_context.')}
    with pytest.warns(RuntimeWarning, match='continue training'):
        restored, provenance = load_categorical_generator(OmegaConf.create(module),
            OmegaConf.create({'checkpoint_path': 'unused', 'config': None}), 'cpu',
            checkpoint_payload={'model': state})
    assert provenance['geometric_context_requires_training']
    for name, tensor in restored.state_dict().items():
        assert torch.equal(tensor, torch.zeros_like(tensor) if name.startswith('geometric_context.') else state[name])
    broken = deepcopy(model.state_dict())
    broken.pop('geometric_context.unknown_dimensions')
    with pytest.raises(RuntimeError, match='unknown_dimensions'):
        load_categorical_generator(OmegaConf.create(module),
            OmegaConf.create({'checkpoint_path': 'unused', 'config': None}), 'cpu',
            checkpoint_payload={'model': broken})
