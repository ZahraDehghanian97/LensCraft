"""Native ET feature contracts, independent of the ET encoder round trip.

Construct normalized native features directly from physical poses using the
published DIRECTOR feature convention, then check the model's physical view.
"""

from pathlib import Path

import pytest
import torch
import yaml

from data.convertor.convertor import convert_to_target
from data.et.config import STANDARDIZATION_CONFIG, STANDARDIZATION_CONFIG_TORCH
from data.sim_format import to_simulation_format
from data.simulation.convertor import SIMConvertor
from data.simulation.dataset import SimulationDataset
from utils.pytorch3d_transform import euler_angles_to_matrix


def yaw_matrix(angle):
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.zeros(*angle.shape, 3, 3, dtype=angle.dtype)
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 2] = sine
    rotation[..., 1, 1] = 1
    rotation[..., 2, 0] = -sine
    rotation[..., 2, 2] = cosine
    return rotation


def native_poses(time):
    camera = torch.stack((2 + .03 * time, -1 + .007 * time, -4 + .02 * time), dim=-1)
    subject = torch.stack((.5 + .016 * time, -.8 + .004 * time, .2 - .009 * time), dim=-1)
    rotation = yaw_matrix(.2 + .001 * time)
    return camera, subject, rotation


def manually_encode_native(camera, subject, rotation):
    """DIRECTOR stores the first two rotation columns, then normalized deltas."""
    stats = {key: value.to(camera) for key, value in STANDARDIZATION_CONFIG_TORCH.items()
             if torch.is_tensor(value)}
    camera_delta = torch.cat((camera[:1], camera[1:] - camera[:-1]), dim=0)
    camera_delta[0] = (camera_delta[0] - stats['shift_mean']) / stats['shift_std']
    camera_delta[1:] = (camera_delta[1:] - stats['norm_mean']) / stats['norm_std']
    subject_delta = torch.cat((subject[:1], subject[1:] - subject[:-1]), dim=0)
    # Native CharacterDataset uses these same three statistics for the first
    # absolute sample and later deltas; it does not use camera shift stats.
    subject_delta = (subject_delta - stats['norm_mean_h']) / stats['norm_std_h']
    columns = rotation[..., :, :2].transpose(-1, -2).reshape(len(camera), 6)
    return torch.cat((columns, camera_delta), dim=-1), subject_delta


@pytest.fixture
def simulation_statistics(monkeypatch):
    statistics = {
        'camera_position': {'mean': torch.tensor([4., -2., 8.]), 'std': torch.tensor([2., 5., 3.])},
        'subject_position': {'mean': torch.tensor([-3., 4., 7.]), 'std': torch.tensor([6., 3., 2.])},
        'subject_dimensions': {'mean': torch.tensor([.9, 1.8, .7]), 'std': torch.tensor([.4, .3, .2])},
    }
    monkeypatch.setattr(SimulationDataset, 'get_normalization_parameters', staticmethod(lambda: statistics))
    return statistics


def make_native_batch(lengths, padding_value=0.):
    native_frames = 300
    camera = torch.full((len(lengths), native_frames, 9), padding_value, dtype=torch.float64)
    subject = torch.full((len(lengths), native_frames, 3), padding_value, dtype=torch.float64)
    padding = torch.ones(len(lengths), native_frames, dtype=torch.bool)
    for index, length in enumerate(lengths):
        if length:
            physical = native_poses(torch.arange(length, dtype=torch.float64))
            camera[index, :length], subject[index, :length] = manually_encode_native(*physical)
            padding[index, :length] = False
    return {
        'camera_trajectory': camera, 'subject_trajectory': subject,
        'subject_volume': None, 'padding_mask': padding,
    }


def test_et_adapter_constants_match_native_director_configuration():
    path = Path(__file__).resolve().parents[1] / 'third_parties/DIRECTOR/configs/dataset/standardization/0300.yaml'
    native = yaml.safe_load(path.read_text())
    for key, value in STANDARDIZATION_CONFIG.items():
        assert native[key] == value, key
    for key, value in STANDARDIZATION_CONFIG_TORCH.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, torch.tensor(native[key]))
        else:
            assert value == native[key]


@pytest.mark.parametrize('padding_value', [0., float('nan')])
def test_et_300_to_30_recovers_physical_paths_and_does_not_integrate_padding(
    simulation_statistics, padding_value,
):
    lengths = [300, 73, 1, 0]
    batch = make_native_batch(lengths, padding_value)
    original = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    camera, subject, volume, padding = to_simulation_format(batch, 'et', target_len=30)
    camera, subject, volume = SimulationDataset.normalize_item(camera, subject, volume, False)
    world_flip = camera.new_tensor([1., -1., -1.])
    for index, length in enumerate(lengths):
        if not length:
            assert padding[index].all()
            continue
        # A complete source clip is reparameterized onto 30 model frames.
        # In particular 73 valid native frames must not be treated as 300.
        time = torch.linspace(0, length - 1, 30, dtype=camera.dtype)
        expected_camera, expected_subject, native_rotation = native_poses(time)
        assert not padding[index].any()
        torch.testing.assert_close(camera[index, :, :3], expected_camera * world_flip, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(subject[index, :, :3], expected_subject * world_flip, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(subject[index, :, 3:], torch.zeros_like(subject[index, :, 3:]))
        # Standard-world Rx(pi), followed by CV-to-GL camera-axis conversion.
        expected_rotation = world_flip[:, None] * native_rotation * world_flip[None, :]
        actual_rotation = euler_angles_to_matrix(camera[index, :, 3:], 'XYZ')
        torch.testing.assert_close(actual_rotation, expected_rotation, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(volume, camera.new_tensor([.5, 1.7, .3]).expand(len(lengths), 1, 3))
    for key, value in original.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(batch[key], value, equal_nan=True)


def test_et_world_flip_preserves_camera_local_subject_ray(simulation_statistics):
    batch = make_native_batch([300])
    camera, subject, volume, padding = to_simulation_format(batch, 'et', target_len=30)
    raw = SimulationDataset.normalize_item(camera, subject, volume, False)
    standard_camera, standard_subject, _ = SIMConvertor().to_standard(*raw)
    time = torch.linspace(0, 299, 30, dtype=camera.dtype)
    native_camera, native_subject, native_rotation = native_poses(time)
    expected_ray = native_rotation.transpose(-1, -2) @ (native_subject - native_camera)[..., None]
    actual_ray = standard_camera[0, :, :3, :3].transpose(-1, -2) @ (
        standard_subject[0, :, :3, 3] - standard_camera[0, :, :3, 3]
    )[..., None]
    torch.testing.assert_close(actual_ray, expected_ray, atol=2e-6, rtol=1e-6)
    assert not padding.any()


def test_et_roll_pitch_and_yaw_use_camera_columns_and_world_rotation(simulation_statistics):
    camera, subject, _ = native_poses(torch.arange(3, dtype=torch.float64))
    cosine, sine = torch.cos(torch.tensor(.4, dtype=camera.dtype)), torch.sin(torch.tensor(.4, dtype=camera.dtype))
    pitch = camera.new_tensor([[1., 0., 0.], [0., cosine, -sine], [0., sine, cosine]])
    quarter_roll = camera.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotation = pitch @ yaw_matrix(camera.new_tensor([.2, .3, .4])) @ quarter_roll
    features, character = manually_encode_native(camera, subject, rotation)
    sim_camera, _, _, _ = convert_to_target(
        'et', 'simulation', features[None], character[None], target_len=3,
    )
    sim_camera, _, _ = SimulationDataset.normalize_item(sim_camera, None, None, False)
    actual = SIMConvertor().to_standard(sim_camera)[0]
    flip = camera.new_tensor([1., -1., -1.])
    torch.testing.assert_close(actual[0, :, :3, :3], flip[:, None] * rotation, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[0, :, :3, 3], camera * flip)


def test_same_length_et_view_preserves_native_padding_and_zero_feature_fill(simulation_statistics):
    batch = make_native_batch([73, 0])
    camera, subject, _, padding = to_simulation_format(batch, 'et', target_len=300)
    assert torch.equal(padding, batch['padding_mask'])
    assert torch.equal(camera[padding], torch.zeros_like(camera[padding]))
    assert torch.equal(subject[padding], torch.zeros_like(subject[padding]))
    assert torch.isfinite(camera).all() and torch.isfinite(subject).all()


def test_simulation_to_et_resamples_positions_before_velocity_encoding(simulation_statistics):
    time = torch.linspace(0, 299, 30, dtype=torch.float64)
    native_camera, native_subject, _ = native_poses(time)
    flip = time.new_tensor([1., -1., -1.])
    camera = torch.zeros(1, 30, 6, dtype=torch.float64)
    subject = torch.zeros_like(camera)
    camera[0, :, :3] = native_camera * flip
    camera[0, :, 4] = -(.2 + .001 * time)
    subject[0, :, :3] = native_subject * flip
    camera, subject, _ = SimulationDataset.normalize_item(camera, subject, None)
    et_camera, et_subject, _, padding = convert_to_target(
        'simulation', 'et', camera, subject, target_len=300,
    )
    expected_camera, expected_subject = manually_encode_native(*native_poses(torch.arange(300, dtype=time.dtype)))
    # Camera and subject velocity statistics must receive 299 native deltas,
    # not 29 model deltas repeated/interpolated in feature space.
    torch.testing.assert_close(et_camera[0], expected_camera, atol=3e-6, rtol=1e-6)
    torch.testing.assert_close(et_subject[0], expected_subject, atol=3e-6, rtol=1e-6)
    assert not padding.any()
