"""Initial-camera ET training views preserve the scene and invalidate labels."""

import copy

import pytest
import torch

from data.simulation.constants import CameraMovementType
from data.simulation.dataset import SimulationDataset
from data.simulation.utils import (
    STRUCTURED_PARAMETERS, cinematography_struct_parameters, get_enum_index,
    structured_conditioning_from_batch,
)
from training.domain_augmentation import DomainAugmentation, DomainAugmentationConfig


def xyz_matrix(angles):
    """Independent analytic XYZ rotation, including noncommuting pitch/roll."""
    x, y, z = angles.unbind(-1)
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    return torch.stack((
        cy * cz, -cy * sz, sy,
        cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy,
        sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy,
    ), dim=-1).reshape(*angles.shape[:-1], 3, 3)


@pytest.fixture
def augmentation():
    return DomainAugmentation({'enabled': True, 'et_coordinate_frame': 'first_camera'})


def set_movement(batch, row, movement):
    index = [name for name, _ in cinematography_struct_parameters].index('movement_type')
    local_id = get_enum_index(CameraMovementType, movement)
    batch['cinematography_prompt_parameters'][row][index] = ('movement_type', movement, local_id, None)
    batch['cinematography_ids'][row, index] = local_id
    batch['prompt_none_mask'][row, index] = local_id >= 0
    batch['structured_presence_mask'][row, index] = local_id >= 0


@pytest.fixture
def batch(monkeypatch):
    stats = {
        'camera_position': {'mean': torch.tensor([6., -2., 7.]), 'std': torch.tensor([2., 3., 4.])},
        'subject_position': {'mean': torch.tensor([1., 4., -3.]), 'std': torch.tensor([5., 2., 3.])},
        'subject_dimensions': {'mean': torch.tensor([.8, 1.9, .7]), 'std': torch.tensor([.2, .6, .5])},
    }
    monkeypatch.setattr(SimulationDataset, 'get_normalization_parameters', staticmethod(lambda: stats))
    camera = torch.tensor([
        [4., 2., -5., .3, .8, -.4], [5., 3., -4., .2, .4, .1],
        [7., 3., -3., .3, .5, -.1], [9., 5., -1., .4, .8, -.2],
    ], dtype=torch.float64).repeat(3, 1, 1)
    subject = torch.tensor([
        [1., 1., 2., 0., .2, 0.], [2., 1., 3., .1, .5, .2],
        [4., 2., 4., .2, 1., .1], [7., 3., 5., .3, 1.2, .2],
    ], dtype=torch.float64).repeat(3, 1, 1)
    volume = torch.tensor([[[.9, 2.2, .6]]], dtype=torch.float64).repeat(3, 1, 1)
    camera, subject, volume = SimulationDataset.normalize_item(camera, subject, volume)
    padding = torch.tensor([[False, False, False, True], [True, False, False, False], [False] * 4])
    camera[padding], subject[padding] = 0., 0.
    fields, split = len(STRUCTURED_PARAMETERS), len(cinematography_struct_parameters)
    metadata = [[(name, 'observed', 0, None)
                 for name, _ in STRUCTURED_PARAMETERS] for _ in range(3)]
    result = {
        'simulation_normalized': True,
        'camera_trajectory': camera, 'subject_trajectory': subject, 'subject_volume': volume,
        'padding_mask': padding,
        'cinematography_ids': torch.zeros(3, split, dtype=torch.long),
        'prompt_none_mask': torch.ones(3, split, dtype=torch.bool),
        'structured_presence_mask': torch.ones(3, fields, dtype=torch.bool),
        'cinematography_prompt_parameters': [row[:split] for row in metadata],
        'simulation_instruction_parameters': [row[split:] for row in metadata],
    }
    for row, movement in enumerate(('truckLeft', 'pedestalUp', 'dutchRight')):
        set_movement(result, row, movement)
    return result


def raw(batch):
    return SimulationDataset.normalize_item(
        batch['camera_trajectory'], batch['subject_trajectory'], batch['subject_volume'], False,
    )


def test_first_camera_frame_is_common_rigid_scene_transform(batch, augmentation):
    source_camera, source_subject, _ = raw(batch)
    result = augmentation.apply(batch, force_domain='et')
    camera, subject, volume = raw(result)
    valid = ~batch['padding_mask']
    for row in range(len(camera)):
        first = valid[row].nonzero()[0, 0]
        q = xyz_matrix(source_camera[row, first, 3:]).T
        origin = source_subject[row, first, :3]
        expected_camera = (q @ (source_camera[row, :, :3] - origin)[..., None]).squeeze(-1)
        expected_subject = (q @ (source_subject[row, :, :3] - origin)[..., None]).squeeze(-1)
        expected_rotation = q @ xyz_matrix(source_camera[row, :, 3:])
        torch.testing.assert_close(camera[row, valid[row], :3], expected_camera[valid[row]])
        torch.testing.assert_close(subject[row, valid[row], :3], expected_subject[valid[row]])
        torch.testing.assert_close(xyz_matrix(camera[row, valid[row], 3:]), expected_rotation[valid[row]])
        torch.testing.assert_close(xyz_matrix(camera[row, first, 3:]), torch.eye(3, dtype=camera.dtype))
        torch.testing.assert_close(subject[row, first, :3], torch.zeros(3, dtype=subject.dtype))
        # The common origin is the subject, not the initial camera position.
        assert not torch.allclose(camera[row, first, :3], torch.zeros(3, dtype=camera.dtype))
    torch.testing.assert_close(subject[..., 3:][valid], torch.zeros_like(subject[..., 3:][valid]))
    torch.testing.assert_close(volume, camera.new_tensor([.5, 1.7, .3]).expand(3, 1, 3))
    assert not torch.equal(result['camera_trajectory'][valid], batch['camera_trajectory'][valid])
    assert not torch.equal(result['subject_trajectory'][..., :3][valid], batch['subject_trajectory'][..., :3][valid])


def test_optical_rays_distances_and_relative_rotations_are_preserved(batch, augmentation):
    old_camera, old_subject, _ = raw(batch)
    camera, subject, _ = raw(augmentation.apply(batch, force_domain='et'))
    valid = ~batch['padding_mask']
    old_rotation, rotation = xyz_matrix(old_camera[..., 3:]), xyz_matrix(camera[..., 3:])
    old_ray = (old_rotation.transpose(-1, -2) @ (old_subject[..., :3] - old_camera[..., :3])[..., None]).squeeze(-1)
    ray = (rotation.transpose(-1, -2) @ (subject[..., :3] - camera[..., :3])[..., None]).squeeze(-1)
    torch.testing.assert_close(ray[valid], old_ray[valid])
    torch.testing.assert_close(torch.linalg.vector_norm(ray[valid], dim=-1),
                               torch.linalg.vector_norm(old_ray[valid], dim=-1))
    pairs = valid[:, 1:] & valid[:, :-1]
    torch.testing.assert_close((rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:])[pairs],
                               (old_rotation[:, :-1].transpose(-1, -2) @ old_rotation[:, 1:])[pairs])


def test_nan_padding_all_padded_rows_and_original_batch_are_preserved(batch, augmentation):
    batch['padding_mask'][2] = True
    for name in ('camera_trajectory', 'subject_trajectory'):
        batch[name][batch['padding_mask']] = float('nan')
    before = copy.deepcopy(batch)
    result = augmentation.apply(batch, force_domain='et')
    for name, value in before.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(batch[name], value, equal_nan=True)
    for name in ('camera_trajectory', 'subject_trajectory'):
        torch.testing.assert_close(result[name][batch['padding_mask']], before[name][batch['padding_mask']], equal_nan=True)
        assert torch.isfinite(result[name][~batch['padding_mask']]).all()
    for key in ('cinematography_prompt_parameters', 'simulation_instruction_parameters'):
        assert [[e[:3] for e in r] for r in batch[key]] == [[e[:3] for e in r] for r in before[key]]


@pytest.mark.parametrize('movement,kept', [
    ('static', True), ('dollyIn', True), ('dollyOut', True), ('truckLeft', True), ('truckRight', True),
    ('panLeft', True), ('panRight', True), ('tiltUp', True), ('tiltDown', True),
    ('dutchLeft', True), ('dutchRight', True), ('pedestalUp', False), ('pedestalDown', False),
    ('arcLeft', False), ('arcRight', False), ('craneUp', False), ('craneDown', False),
    ('follow', False), ('track', False), (None, False),
])
def test_motion_masks_are_per_example_and_update_tokens_and_metadata(batch, augmentation, movement, kept):
    names = augmentation.field_names
    set_movement(batch, 1, movement)
    result = augmentation.apply(batch, force_domain='et')
    caption, original = structured_conditioning_from_batch(result), structured_conditioning_from_batch(batch)
    assert caption.shape == (10, 3, 21)
    assert result['prompt_none_mask'].shape == (3, 10)
    assert result['structured_presence_mask'].shape == (3, 46)
    metadata = [a + b for a, b in zip(result['cinematography_prompt_parameters'], result['simulation_instruction_parameters'])]
    original_metadata = [a + b for a, b in zip(batch['cinematography_prompt_parameters'], batch['simulation_instruction_parameters'])]
    for name in ('movement_type', 'dynamic_direction', 'dynamic_movementMode'):
        i = names.index(name)
        assert result['structured_presence_mask'][:, i].tolist() == [True, kept, True]
        if kept:
            assert all(row[i] == source[i] for row, source in zip(metadata, original_metadata))
        else:
            assert metadata[1][i][1:3] == (None, -1)
        if i < 10:
            assert result['prompt_none_mask'][:, i].tolist() == [True, kept, True]
            if kept:
                torch.testing.assert_close(caption[i], original[i])
            else:
                assert torch.count_nonzero(caption[i, 1]) == 0
                assert result['cinematography_ids'][1, i] == -1
                torch.testing.assert_close(caption[i, [0, 2]], original[i, [0, 2]])
    for i, name in enumerate(names):
        if name.startswith(('constraints_lockedMovement_', 'constraints_lockedRotation_')):
            assert not result['structured_presence_mask'][:, i].any()
            assert all(row[i][1:3] == (None, -1) for row in metadata)
    for name in ('movement_speed', 'constraints_maxSpeed', 'constraints_staticDistance'):
        assert result['structured_presence_mask'][:, names.index(name)].all()


def test_missing_compact_batch_metadata_fails_with_actionable_error(batch, augmentation):
    del batch['cinematography_prompt_parameters']
    with pytest.raises(ValueError, match='data.compact_batches=false'):
        augmentation.apply(batch, force_domain='et')


def test_legacy_et_default_and_non_et_views_are_unchanged(batch, augmentation):
    legacy = DomainAugmentation({'enabled': True})
    explicit = DomainAugmentation({'enabled': True, 'et_coordinate_frame': 'world'})
    assert legacy.config.et_coordinate_frame == 'world'
    for domain, left, right in (
        ('et', legacy, explicit), ('original', legacy, augmentation), ('ccdm', legacy, augmentation),
    ):
        expected, actual = left.apply(batch, force_domain=domain), right.apply(batch, force_domain=domain)
        for key, value in expected.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    result = legacy.apply(batch, force_domain='et')
    assert torch.equal(result['camera_trajectory'], batch['camera_trajectory'])
    assert torch.equal(result['subject_trajectory'][..., :3], batch['subject_trajectory'][..., :3])


@pytest.mark.parametrize('value', ['identity', '', None, True])
def test_invalid_coordinate_frame_is_rejected(value):
    with pytest.raises(ValueError, match='ET coordinate frame'):
        DomainAugmentationConfig(et_coordinate_frame=value)
