"""SIM-only CCDM supervision must agree with the transformed target geometry."""

import copy

import pytest
import torch

from data.ccdm.convertor import CCDMConvertor
from data.simulation.constants import NumericFeature
from data.simulation.convertor import SIMConvertor
from data.simulation.dataset import SimulationDataset
from data.simulation.utils import (
    CLIP_PARAMETERS, cinematography_struct_parameters, structured_conditioning_from_batch,
)
from training.domain_augmentation import DomainAugmentation, DomainAugmentationConfig


def rotation_y(angle, reference):
    c, s = reference.new_tensor(angle).cos(), reference.new_tensor(angle).sin()
    zero, one = reference.new_tensor(0.), reference.new_tensor(1.)
    return torch.stack((c, zero, s, zero, one, zero, -s, zero, c)).reshape(3, 3)


@pytest.fixture
def vocabulary():
    result = {}
    for _, kind in CLIP_PARAMETERS:
        if isinstance(kind, NumericFeature):
            continue
        name = 'boolean' if kind is bool else kind.__name__
        result[name] = {'a': torch.tensor([1., 2., 3., 4.]),
                        'b': torch.tensor([3., 4., 5., 6.])}
    return result


@pytest.fixture
def augmentation(vocabulary):
    return DomainAugmentation({
        'enabled': True, 'ccdm_static_motion': True, 'ccdm_native_manifold': True,
    }, vocabulary)


@pytest.fixture
def batch(monkeypatch):
    stats = {
        'camera_position': {'mean': torch.tensor([6., -2., 7.]), 'std': torch.tensor([2., 3., 4.])},
        'subject_position': {'mean': torch.tensor([1., 4., -3.]), 'std': torch.tensor([5., 2., 3.])},
        'subject_dimensions': {'mean': torch.tensor([.8, 1.9, .7]), 'std': torch.tensor([.2, .6, .5])},
    }
    monkeypatch.setattr(SimulationDataset, 'get_normalization_parameters', staticmethod(lambda: stats))
    subject = torch.eye(4, dtype=torch.float64).expand(3, 4, 4, 4).clone()
    subject[..., :3, :3] = rotation_y(.6, subject)
    subject[..., :3, 3] = subject.new_tensor([4., 2., -1.])
    native = torch.eye(4, dtype=subject.dtype).expand_as(subject).clone()
    native[..., :3, :3] = torch.diag(subject.new_tensor([1., -1., -1.]))
    native[..., :3, 3] = subject.new_tensor([
        [2., 1., 5.], [1.6, 1., 5.], [1.2, 1., 5.], [.8, 1., 5.],
    ])
    # Independent rigid placement: the native front/left basis differs by yaw pi.
    frame = subject[..., :3, :3] @ torch.diag(subject.new_tensor([-1., 1., -1.]))
    camera = native.clone()
    camera[..., :3, :3] = frame @ native[..., :3, :3]
    camera[..., :3, 3] = (frame @ native[..., :3, 3, None]).squeeze(-1) + subject[..., :3, 3]
    raw = SIMConvertor().from_standard(camera, subject, subject.new_tensor([[.9, 2.2, .6]]).expand(3, 3))
    camera, subject, volume = SimulationDataset.normalize_item(*raw)
    fields, split = len(CLIP_PARAMETERS), len(cinematography_struct_parameters)
    captions = torch.arange(fields * 3 * 4, dtype=subject.dtype).reshape(fields, 3, 4) / 10
    metadata = [[(name, 'observed', 0, captions[i, b].clone())
                 for i, (name, _) in enumerate(CLIP_PARAMETERS)] for b in range(3)]
    movement = [name for name, _ in CLIP_PARAMETERS].index('movement_type')
    for row in metadata:
        row[movement] = ('movement_type', 'truckLeft', 0, row[movement][3])
    return {
        'simulation_normalized': True,
        'camera_trajectory': camera, 'subject_trajectory': subject, 'subject_volume': volume,
        'padding_mask': torch.tensor([[False, False, False, True], [False] * 4, [False] * 4]),
        'cinematography_prompt': captions[:split], 'simulation_instruction': captions[split:],
        'prompt_none_mask': torch.ones(3, fields, dtype=torch.bool),
        'cinematography_prompt_parameters': [row[:split] for row in metadata],
        'simulation_instruction_parameters': [row[split:] for row in metadata],
    }


def standard(batch):
    return SIMConvertor().to_standard(*SimulationDataset.normalize_item(
        batch['camera_trajectory'], batch['subject_trajectory'], batch['subject_volume'], False,
    ))


def set_standard(batch, camera=None, subject=None):
    old_camera, old_subject, volume = standard(batch)
    raw = SIMConvertor().from_standard(
        old_camera if camera is None else camera, old_subject if subject is None else subject, volume,
    )
    result = dict(batch)
    result['camera_trajectory'], result['subject_trajectory'], result['subject_volume'] = (
        SimulationDataset.normalize_item(*raw)
    )
    return result


def test_native_manifold_preserves_positions_screen_coordinates_and_removes_roll(batch, augmentation):
    camera, subject, _ = standard(batch)
    roll = .3
    c, s = camera.new_tensor(roll).cos(), camera.new_tensor(roll).sin()
    roll_rotation = camera.new_tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    camera[..., :3, :3] = camera[..., :3, :3] @ roll_rotation
    rolled = set_standard(batch, camera=camera)
    before = CCDMConvertor.to_native_subject_frame(camera, subject)
    result = augmentation.apply(rolled, force_domain='ccdm')
    actual, implicit_subject, _ = standard(result)
    valid = ~batch['padding_mask']
    torch.testing.assert_close(actual[..., :3, 3][valid], before[..., :3, 3][valid])
    # Direct pinhole ratios verify the 5-D screen channels independently of the
    # converter's encoder; shared FOV factors cancel on both sides.
    old_ray = (before[..., :3, :3].transpose(-1, -2) @ (-before[..., :3, 3, None])).squeeze(-1)
    ray = (actual[..., :3, :3].transpose(-1, -2) @ (-actual[..., :3, 3, None])).squeeze(-1)
    torch.testing.assert_close((ray[..., :2] / ray[..., 2:])[valid],
                               (old_ray[..., :2] / old_ray[..., 2:])[valid])
    torch.testing.assert_close(actual[..., 1, 0][valid], torch.zeros_like(actual[..., 1, 0][valid]), atol=1e-8, rtol=0.)
    assert not torch.allclose(actual[..., :3, :3][valid], before[..., :3, :3][valid])
    torch.testing.assert_close(implicit_subject[..., :3, 3][valid], torch.zeros_like(implicit_subject[..., :3, 3][valid]))
    movement = augmentation.field_names.index('movement_type')
    assert not result['prompt_none_mask'][:, movement].any()


def test_stationary_upright_subject_preserves_rigid_motion_and_caption_easing(batch, augmentation):
    camera, subject, _ = standard(batch)
    result = augmentation.apply(batch, force_domain='ccdm')
    actual, _, _ = standard(result)
    valid = ~batch['padding_mask']
    expected = CCDMConvertor.to_native_subject_frame(camera, subject)
    torch.testing.assert_close(actual[valid], expected[valid])
    pairs = valid[:, 1:] & valid[:, :-1]
    old_speed = torch.linalg.vector_norm(camera[:, 1:, :3, 3] - camera[:, :-1, :3, 3], dim=-1)
    speed = torch.linalg.vector_norm(actual[:, 1:, :3, 3] - actual[:, :-1, :3, 3], dim=-1)
    torch.testing.assert_close(speed[pairs], old_speed[pairs])
    # The truckLeft annotation denotes negative camera-right displacement;
    # this verifies label direction as well as generic rigid-frame invariance.
    old_local_step = (camera[:, :-1, :3, :3].transpose(-1, -2)
                      @ (camera[:, 1:, :3, 3] - camera[:, :-1, :3, 3])[..., None]).squeeze(-1)
    local_step = (actual[:, :-1, :3, :3].transpose(-1, -2)
                  @ (actual[:, 1:, :3, 3] - actual[:, :-1, :3, 3])[..., None]).squeeze(-1)
    torch.testing.assert_close(local_step[pairs], old_local_step[pairs])
    assert (local_step[..., 0][pairs] < 0).all()
    old_relative = camera[:, :-1, :3, :3].transpose(-1, -2) @ camera[:, 1:, :3, :3]
    relative = actual[:, :-1, :3, :3].transpose(-1, -2) @ actual[:, 1:, :3, :3]
    torch.testing.assert_close(relative[pairs], old_relative[pairs])
    caption, original = structured_conditioning_from_batch(result), structured_conditioning_from_batch(batch)
    for name in augmentation.CCDM_STATIC_MOTION_FIELDS:
        index = augmentation.field_names.index(name)
        assert result['prompt_none_mask'][:, index].all(), name
        torch.testing.assert_close(caption[index], original[index])
    for name in ('initial_shotSize', 'final_shotSize', 'initial_subjectFraming', 'final_subjectFraming'):
        assert not result['prompt_none_mask'][:, augmentation.field_names.index(name)].any()


def test_yaw_frame_keeps_pedestal_up_on_gravity_axis(batch, augmentation):
    camera, _, _ = standard(batch)
    camera[:, :, :3, 3] = camera[:, :1, :3, 3].clone()
    camera[:, :, 1, 3] += camera.new_tensor([0., .2, .4, .6])
    pedestal = set_standard(batch, camera=camera)
    movement = augmentation.field_names.index('movement_type')
    for row in pedestal['cinematography_prompt_parameters']:
        entry = row[movement]
        row[movement] = (entry[0], 'pedestalUp', entry[2], entry[3])
    actual, _, _ = standard(augmentation.apply(pedestal, force_domain='ccdm'))
    pairs = ~batch['padding_mask'][:, 1:] & ~batch['padding_mask'][:, :-1]
    step = actual[:, 1:, :3, 3] - actual[:, :-1, :3, 3]
    torch.testing.assert_close(step[pairs], step.new_tensor([0., .2, 0.]).expand(pairs.sum(), 3))
    result = augmentation.apply(pedestal, force_domain='ccdm')
    assert result['prompt_none_mask'][:, movement].all()


@pytest.mark.parametrize('movement', ['dollyIn', 'pedestalUp', 'arcRight'])
def test_roll_removal_retains_only_position_motion_facts(batch, augmentation, movement):
    camera, source_subject, _ = standard(batch)
    native = CCDMConvertor.to_native_subject_frame(camera, source_subject)
    if movement == 'dollyIn':
        native[..., :3, 3] = native[:, :1, :3, 3].clone() * native.new_tensor([1., .9, .8, .7])[None, :, None]
    elif movement == 'pedestalUp':
        native[..., :3, 3] = native[:, :1, :3, 3].clone()
        native[..., 1, 3] += native.new_tensor([0., .2, .4, .6])
    else:
        angles = native.new_tensor([0., .15, .3, .45])
        native[..., :3, 3] = torch.stack((5 * angles.sin(), torch.ones_like(angles), 5 * angles.cos()), -1)
        z = -torch.nn.functional.normalize(native[..., :3, 3], dim=-1)
        x = torch.nn.functional.normalize(torch.linalg.cross(z, native.new_tensor([0., 1., 0.]).expand_as(z)), dim=-1)
        native[..., :3, :3] = torch.stack((x, torch.linalg.cross(z, x), z), dim=-1)
    roll = camera.new_tensor([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    native[..., :3, :3] = native[..., :3, :3] @ roll
    frame = source_subject[..., :3, :3] @ torch.diag(camera.new_tensor([-1., 1., -1.]))
    camera[..., :3, :3] = frame @ native[..., :3, :3]
    camera[..., :3, 3] = (frame @ native[..., :3, 3, None]).squeeze(-1) + source_subject[..., :3, 3]
    changed = set_standard(batch, camera=camera)
    index = augmentation.field_names.index('movement_type')
    for row in changed['cinematography_prompt_parameters']:
        entry = row[index]
        row[index] = (entry[0], movement, entry[2], entry[3])
    result = augmentation.apply(changed, force_domain='ccdm')
    actual, _, _ = standard(result)
    source_camera, source_subject, _ = standard(changed)
    expected = CCDMConvertor.to_native_subject_frame(source_camera, source_subject)
    valid = ~batch['padding_mask']
    torch.testing.assert_close(actual[..., :3, 3][valid], expected[..., :3, 3][valid])
    assert not torch.allclose(actual[..., :3, :3][valid], expected[..., :3, :3][valid])
    pairs = valid[:, 1:] & valid[:, :-1]
    if movement == 'dollyIn':
        radius = torch.linalg.vector_norm(actual[..., :3, 3], dim=-1)
        assert ((radius[:, 1:] - radius[:, :-1])[pairs] < 0).all()
    elif movement == 'pedestalUp':
        step = actual[:, 1:, :3, 3] - actual[:, :-1, :3, 3]
        torch.testing.assert_close(step[pairs], step.new_tensor([0., .2, 0.]).expand(pairs.sum(), 3))
    else:
        azimuth = torch.atan2(actual[..., 0, 3], actual[..., 2, 3])
        torch.testing.assert_close((azimuth[:, 1:] - azimuth[:, :-1])[pairs],
                                   azimuth.new_full((int(pairs.sum()),), .15))
    for name in augmentation.CCDM_STATIC_MOTION_FIELDS:
        index = augmentation.field_names.index(name)
        assert result['prompt_none_mask'][:, index].tolist() == [
            name in augmentation.CCDM_POSITION_MOTION_FIELDS,
        ] * 3, name


@pytest.mark.parametrize('movement', ['follow', 'track', 'dutchLeft', None])
def test_subject_inherited_roll_or_unknown_motion_is_not_restored(batch, augmentation, movement):
    index = augmentation.field_names.index('movement_type')
    for row in batch['cinematography_prompt_parameters']:
        entry = row[index]
        row[index] = (entry[0], movement, entry[2], entry[3])
    result = augmentation.apply(batch, force_domain='ccdm')
    assert not result['prompt_none_mask'][:, index].any()


@pytest.mark.parametrize('change', ['translation', 'turn', 'tilt'])
def test_moving_or_tilted_subject_does_not_restore_motion(batch, augmentation, change):
    _, subject, _ = standard(batch)
    if change == 'translation':
        subject[:, 1:, 0, 3] += .1
    elif change == 'turn':
        subject[:, 1:, :3, :3] = rotation_y(.7, subject)
    else:
        # A static subject with a tilted gravity axis is also unsafe.
        x_rotation = subject.new_tensor([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]])
        subject[..., :3, :3] = x_rotation @ subject[..., :3, :3]
    result = augmentation.apply(set_standard(batch, subject=subject), force_domain='ccdm')
    for name in augmentation.CCDM_STATIC_MOTION_FIELDS:
        assert not result['prompt_none_mask'][:, augmentation.field_names.index(name)].any(), name


def test_native_subject_is_target_independent_and_missing_tokens_stay_missing(batch, augmentation):
    changed = dict(batch, camera_trajectory=batch['camera_trajectory'] + 1.7)
    first = augmentation.apply(batch, force_domain='ccdm')
    second = augmentation.apply(changed, force_domain='ccdm')
    torch.testing.assert_close(first['subject_trajectory'], second['subject_trajectory'])
    torch.testing.assert_close(first['subject_volume'], second['subject_volume'])
    index = augmentation.field_names.index('movement_speed')
    batch['prompt_none_mask'][:, index] = False
    batch['cinematography_prompt'][index] = 123.
    result = augmentation.apply(batch, force_domain='ccdm')
    assert not result['prompt_none_mask'][:, index].any()
    torch.testing.assert_close(result['cinematography_prompt'][index], batch['cinematography_prompt'][index])


def test_nan_padding_and_insufficient_valid_frames_are_ignored(batch, augmentation):
    batch['padding_mask'][1] = True
    batch['padding_mask'][2, 1:] = True
    for name in ('camera_trajectory', 'subject_trajectory'):
        batch[name][batch['padding_mask']] = float('nan')
    before = copy.deepcopy(batch)
    result = augmentation.apply(batch, force_domain='ccdm')
    for name in ('camera_trajectory', 'subject_trajectory'):
        torch.testing.assert_close(batch[name], before[name], equal_nan=True)
        torch.testing.assert_close(result[name][batch['padding_mask']], before[name][batch['padding_mask']], equal_nan=True)
        assert torch.isfinite(result[name][~batch['padding_mask']]).all()
    movement = augmentation.field_names.index('movement_type')
    assert result['prompt_none_mask'][:, movement].tolist() == [True, False, False]


def test_static_motion_requires_sim_annotation_metadata(batch, augmentation):
    batch.pop('cinematography_prompt_parameters')
    with pytest.raises(ValueError, match='compact_batches=false'):
        augmentation.apply(batch, force_domain='ccdm')


@pytest.mark.parametrize('config', [
    {'ccdm_static_motion': 'true'}, {'ccdm_native_manifold': 1},
])
def test_new_flags_reject_non_boolean_values(config):
    with pytest.raises(ValueError, match='boolean'):
        DomainAugmentationConfig(**config)
