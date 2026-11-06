"""Synthetic transfer views must match native subject information and geometry."""

import copy
import pytest
import torch

from data.ccdm.convertor import CCDMConvertor
from data.et.convertor import ETConvertor
from data.simulation.constants import NumericFeature
from data.simulation.convertor import SIMConvertor
from data.simulation.dataset import SimulationDataset
from data.simulation.utils import (
    CLIP_PARAMETERS, cinematography_struct_parameters, structured_conditioning_from_batch,
)
from training.domain_augmentation import DomainAugmentation, DomainAugmentationConfig


@pytest.fixture
def vocabulary():
    result = {}
    for _, kind in CLIP_PARAMETERS:
        if isinstance(kind, NumericFeature):
            continue
        name = 'boolean' if kind is bool else kind.__name__
        result[name] = {'a': torch.tensor([1., 2., 3., 4.]), 'b': torch.tensor([3., 4., 5., 6.])}
    return result


@pytest.fixture
def augmentation(vocabulary):
    return DomainAugmentation({'enabled': True}, vocabulary)


@pytest.fixture
def batch(monkeypatch):
    # Camera/subject/size statistics intentionally differ and are not centered
    # at zero. Normalized zero is therefore not a zero world-space position.
    parameters = {
        'camera_position': {'mean': torch.tensor([6., -2., 7.]), 'std': torch.tensor([2., 3., 4.])},
        'subject_position': {'mean': torch.tensor([1., 4., -3.]), 'std': torch.tensor([5., 2., 3.])},
        'subject_dimensions': {'mean': torch.tensor([.8, 1.9, .7]), 'std': torch.tensor([.2, .6, .5])},
    }
    monkeypatch.setattr(SimulationDataset, 'get_normalization_parameters', staticmethod(lambda: parameters))
    camera = torch.tensor([
        [4., 2., -5., .1, .2, .3], [5., 3., -4., .2, .4, .1],
        [7., 3., -3., .3, .5, -.1], [9., 5., -1., .4, .8, -.2],
    ], dtype=torch.float64).repeat(3, 1, 1)
    subject = torch.tensor([
        [1., 1., 2., 0., .2, 0.], [2., 1., 3., .1, .5, .2],
        [4., 2., 4., .2, 1., .1], [7., 3., 5., .3, 1.2, .2],
    ], dtype=torch.float64).repeat(3, 1, 1)
    volume = torch.tensor([[[.9, 2.2, .6]]], dtype=torch.float64).repeat(3, 1, 1)
    camera, subject, volume = SimulationDataset.normalize_item(camera, subject, volume)
    padding = torch.tensor([[False, False, False, True], [False] * 4, [False, False, True, True]])
    camera[padding] = 0.
    subject[padding] = 0.
    fields = len(CLIP_PARAMETERS)
    split = len(cinematography_struct_parameters)
    caption = torch.arange(fields * 3 * 4, dtype=torch.float64).reshape(fields, 3, 4) / 10
    metadata = [[(name, 'observed', 0, caption[i, b].clone())
                 for i, (name, _) in enumerate(CLIP_PARAMETERS)] for b in range(3)]
    return {
        'simulation_normalized': True,
        'camera_trajectory': camera,
        'subject_trajectory': subject,
        'subject_volume': volume,
        'padding_mask': padding,
        'cinematography_prompt': caption[:split],
        'simulation_instruction': caption[split:],
        'prompt_none_mask': torch.ones(3, fields, dtype=torch.bool),
        'cinematography_prompt_parameters': [row[:split] for row in metadata],
        'simulation_instruction_parameters': [row[split:] for row in metadata],
    }


def standard(batch):
    raw = SimulationDataset.normalize_item(
        batch['camera_trajectory'], batch['subject_trajectory'], batch['subject_volume'], False,
    )
    return SIMConvertor().to_standard(*raw)


def test_et_view_matches_native_unknown_orientation_size_and_preserves_camera(batch, augmentation):
    result = augmentation.apply(batch, force_domain='et')
    camera, subject, volume = standard(result)
    original_camera, original_subject, _ = standard(batch)
    valid = ~batch['padding_mask']
    assert torch.equal(result['camera_trajectory'], batch['camera_trajectory'])
    assert torch.equal(result['subject_trajectory'][..., :3], batch['subject_trajectory'][..., :3])
    native_et = ETConvertor().from_standard(original_camera, original_subject)
    expected = ETConvertor().to_standard(*native_et)
    torch.testing.assert_close(camera[valid], expected[0][valid])
    torch.testing.assert_close(subject[valid], expected[1][valid])
    torch.testing.assert_close(volume, expected[2])
    torch.testing.assert_close(subject[..., :3, :3][valid], torch.eye(3, dtype=subject.dtype).expand(valid.sum(), 3, 3))


def test_ccdm_frame_change_preserves_subject_relative_projection_and_orientation(batch, augmentation):
    original_camera, original_subject, _ = standard(batch)
    result = augmentation.apply(batch, force_domain='ccdm')
    camera, subject, volume = standard(result)
    valid = ~batch['padding_mask']
    basis = torch.diag(camera.new_tensor([-1., 1., -1.]))
    expected_rotation = basis @ original_subject[..., :3, :3].transpose(-1, -2)
    expected_position = (expected_rotation @ (
        original_camera[..., :3, 3] - original_subject[..., :3, 3]
    )[..., None]).squeeze(-1)
    torch.testing.assert_close(camera[..., :3, 3][valid], expected_position[valid])
    torch.testing.assert_close(camera[..., :3, :3][valid], (expected_rotation @ original_camera[..., :3, :3])[valid])
    torch.testing.assert_close(subject[..., :3, 3][valid], torch.zeros_like(subject[..., :3, 3][valid]))
    torch.testing.assert_close(subject[..., :3, :3][valid], basis.expand(valid.sum(), 3, 3))
    torch.testing.assert_close(volume, camera.new_tensor([.5, 1.7, .3]).expand(3, 3))
    original_local = original_subject[..., :3, :3].transpose(-1, -2) @ (
        original_camera[..., :3, 3] - original_subject[..., :3, 3]
    )[..., None]
    transformed_local = subject[..., :3, :3].transpose(-1, -2) @ camera[..., :3, 3, None]
    # Both cameraAngle (elevation) and subjectView (azimuth) are defined by
    # this subject-local direction in the simulator's setup/angles.ts.
    torch.testing.assert_close(transformed_local[valid], original_local[valid])
    # This also detects flipping the native subject basis or reflecting camera
    # axes: inverse placement must recover the camera's full standard pose.
    restored = CCDMConvertor.place_in_subject_frame(camera, original_subject)
    torch.testing.assert_close(restored[valid], original_camera[valid])
    old_projection = (original_camera[..., :3, :3].transpose(-1, -2)
                      @ (original_subject[..., :3, 3] - original_camera[..., :3, 3])[..., None])
    new_projection = camera[..., :3, :3].transpose(-1, -2) @ (-camera[..., :3, 3, None])
    torch.testing.assert_close(new_projection[valid], old_projection[valid])


@pytest.mark.parametrize('domain', ['et', 'ccdm'])
def test_subject_condition_never_depends_on_target_camera(batch, augmentation, domain):
    changed = dict(batch)
    changed['camera_trajectory'] = batch['camera_trajectory'] + 3.4
    first = augmentation.apply(batch, force_domain=domain)
    second = augmentation.apply(changed, force_domain=domain)
    torch.testing.assert_close(first['subject_trajectory'], second['subject_trajectory'])
    torch.testing.assert_close(first['subject_volume'], second['subject_volume'])
    assert torch.equal(first['prompt_none_mask'], second['prompt_none_mask'])


@pytest.mark.parametrize('domain', ['et', 'ccdm'])
def test_padding_and_input_batch_are_unchanged_even_with_nan_padding(batch, augmentation, domain):
    batch['camera_trajectory'][batch['padding_mask']] = float('nan')
    batch['subject_trajectory'][batch['padding_mask']] = float('nan')
    before = copy.deepcopy(batch)
    result = augmentation.apply(batch, force_domain=domain)
    for name, value in before.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(batch[name], value, equal_nan=True)
    for name in ('camera_trajectory', 'subject_trajectory'):
        torch.testing.assert_close(result[name][batch['padding_mask']], before[name][batch['padding_mask']], equal_nan=True)
        assert torch.isfinite(result[name][~batch['padding_mask']]).all()
    for name in ('cinematography_prompt_parameters', 'simulation_instruction_parameters'):
        assert all(entry[2] == 0 for row in batch[name] for entry in row)
    assert torch.equal(result['padding_mask'], before['padding_mask'])


@pytest.mark.parametrize('domain', ['et', 'ccdm'])
def test_invalid_captions_have_absent_masks_neutral_tokens_and_missing_metadata(batch, augmentation, domain):
    result = augmentation.apply(batch, force_domain=domain)
    names = augmentation.field_names
    if domain == 'et':
        dropped = [name for name in names if any(field in name for field in (
            'shotSize', 'subjectView', 'cameraAngle', 'subjectFraming',
        ))]
        retained = ['movement_type', 'movement_speed', 'setup_kind']
    else:
        dropped = [name for name in names if name.startswith(('movement_', 'constraints_'))]
        dropped += ['initial_shotSize', 'initial_subjectFraming', 'dynamic_type', 'dynamic_direction', 'dynamic_easing']
        retained = ['initial_subjectView', 'final_subjectView', 'initial_cameraAngle', 'setup_kind']
    caption = structured_conditioning_from_batch(result)
    original_caption = structured_conditioning_from_batch(batch)
    metadata = [a + b for a, b in zip(result['cinematography_prompt_parameters'], result['simulation_instruction_parameters'])]
    for name in dropped:
        index = names.index(name)
        assert not result['prompt_none_mask'][:, index].any(), name
        torch.testing.assert_close(caption[index], augmentation.unknown_tokens[index].to(caption).expand(3, -1))
        assert all(row[index][1:3] == (None, -1) for row in metadata)
    for name in retained:
        index = names.index(name)
        assert result['prompt_none_mask'][:, index].all(), name
        torch.testing.assert_close(caption[index], original_caption[index])


def test_existing_missing_field_keeps_its_dataset_fill(batch, augmentation):
    index = augmentation.field_names.index('initial_shotSize')
    batch['prompt_none_mask'][:, index] = False
    batch['cinematography_prompt'][index] = 123.
    result = augmentation.apply(batch, force_domain='ccdm')
    assert not result['prompt_none_mask'][:, index].any()
    torch.testing.assert_close(result['cinematography_prompt'][index], batch['cinematography_prompt'][index])


def test_mixed_batch_preserves_originals_exactly(batch, augmentation, monkeypatch):
    monkeypatch.setattr(augmentation, 'sample_views', lambda *args, **kwargs: torch.tensor([0, 1, 2]))
    result = augmentation.apply(batch)
    for name in ('camera_trajectory', 'subject_trajectory', 'subject_volume', 'prompt_none_mask'):
        assert torch.equal(result[name][0], batch[name][0])
    for name in ('cinematography_prompt', 'simulation_instruction'):
        assert torch.equal(result[name][:, 0], batch[name][:, 0])
    assert torch.equal(result['domain_augmentation_view'], torch.tensor([0, 1, 2]))


def test_seeded_views_are_partition_invariant_and_do_not_advance_rng(augmentation):
    seeds = list(range(100))
    state = torch.random.get_rng_state().clone()
    full = augmentation.sample_views(len(seeds), sample_seeds=seeds)
    parts = torch.cat([augmentation.sample_views(50, sample_seeds=seeds[:50]),
                       augmentation.sample_views(50, sample_seeds=seeds[50:])])
    assert torch.equal(full, parts)
    assert torch.equal(state, torch.random.get_rng_state())
    assert set(full.tolist()) == {0, 1, 2}


def test_disabled_and_original_view_do_not_transform_batch(batch, vocabulary, augmentation):
    for transform in (DomainAugmentation({}, vocabulary), augmentation):
        result = transform.apply(batch, force_domain='original')
        assert not result['domain_augmentation_view'].any()
        for name in batch:
            assert result[name] is batch[name]


@pytest.mark.parametrize('config', [
    {'et_probability': float('nan')}, {'ccdm_probability': -.1},
    {'et_probability': .5, 'ccdm_probability': .5}, {'fill': 'invalid'},
])
def test_invalid_configs_fail_early(config):
    with pytest.raises(ValueError):
        DomainAugmentationConfig(**config)


def test_unnormalized_input_is_rejected(batch, augmentation):
    batch['simulation_normalized'] = False
    with pytest.raises(ValueError, match='normalized'):
        augmentation.apply(batch, force_domain='ccdm')


def test_ccdm_following_camera_becomes_static_with_turning_subject(batch, augmentation):
    _, subject, _ = standard(batch)
    native_camera = torch.eye(4, dtype=subject.dtype).expand_as(subject).clone()
    native_camera[..., :3, 3] = subject.new_tensor([2., 1., 5.])
    placed = CCDMConvertor.place_in_subject_frame(native_camera, subject)
    raw_camera = SIMConvertor().from_standard(placed)[0]
    batch['camera_trajectory'] = SimulationDataset.normalize_item(raw_camera, None, None)[0]
    result = augmentation.apply(batch, force_domain='ccdm')
    actual_camera, _, _ = standard(result)
    valid = ~batch['padding_mask']
    torch.testing.assert_close(actual_camera[valid], native_camera[valid])
