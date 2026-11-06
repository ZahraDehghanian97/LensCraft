"""An optional position anchor must catch drift and ignore temporal padding."""

import pytest
import torch

from training.losses.losses import CameraTrajectoryLoss


def trajectory(frames=4):
    value = torch.zeros(1, frames, 12, dtype=torch.float64)
    value[0, :, 0] = torch.arange(frames, dtype=torch.float64)
    value[..., 3:] = torch.eye(3, dtype=value.dtype).reshape(9)
    return value


def total_loss(module, predicted, target, padding=None):
    return module.compute_total_loss(
        predicted, target, None, None, {},
        tgt_key_padding_mask=padding, projected_pred=predicted,
    )


def test_absolute_position_detects_constant_offset_hidden_from_relative_motion():
    target = trajectory()
    predicted = target.clone()
    predicted[..., :3] += torch.tensor([3., -6., 9.])
    module = CameraTrajectoryLoss(losses_list={'position_absolute': 8.})
    total, terms = total_loss(module, predicted, target)
    # Every coordinate/frame has the same offset: (9 + 36 + 81) / 3.
    assert terms['position_absolute'].item() == pytest.approx(42.)
    assert terms['relative'].item() == 0.
    assert terms['speed'].item() == 0.
    assert total.item() == pytest.approx(8. * 42.)
    # Enabling this position-only term must not silently add rotation weights.
    assert module.losses_list == {'position_absolute': 8.}


def test_absolute_position_detects_nonfirst_frame_drift_and_gives_its_gradient():
    target = trajectory()
    predicted = target.clone()
    predicted[0, 1:, 0] += torch.tensor([1., 2., 3.])
    predicted.requires_grad_()
    module = CameraTrajectoryLoss(losses_list={'position_absolute': 1.})
    total, terms = total_loss(module, predicted, target)
    assert terms['first_frame'].item() == 0.
    assert terms['position_absolute'].item() == pytest.approx((1. + 4. + 9.) / 12.)
    total.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, 0, :3].count_nonzero().item() == 0
    assert (predicted.grad[0, 1:, 0] > 0).all()
    assert predicted.grad[..., 3:].count_nonzero().item() == 0


def test_absolute_position_ignores_nan_padding_in_value_and_gradient():
    target = trajectory(frames=5)
    predicted = target.clone()
    predicted[:, :3, :3] += 2.
    target[:, 3:] = float('nan')
    predicted[:, 3:] = float('nan')
    predicted.requires_grad_()
    padding = torch.tensor([[False, False, False, True, True]])
    module = CameraTrajectoryLoss(losses_list={'position_absolute': 8.})
    total, terms = total_loss(module, predicted, target, padding)
    assert terms['position_absolute'].item() == pytest.approx(4.)
    assert total.item() == pytest.approx(32.)
    total.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[:, 3:].count_nonzero().item() == 0
    assert predicted.grad[:, :3, :3].abs().sum().item() > 0


def test_legacy_objective_has_no_absolute_position_weight_or_term():
    target = trajectory()
    predicted = target.clone()
    predicted[..., :3] += 2.
    module = CameraTrajectoryLoss(losses_list={'first_frame': 2., 'relative': 16., 'speed': 80.})
    total, terms = total_loss(module, predicted, target)
    assert 'position_absolute' not in module.losses_list
    assert 'position_absolute' not in terms
    # Legacy relative/step and rotation terms vanish for this translated path.
    assert total.item() == pytest.approx(2. * 4.)
