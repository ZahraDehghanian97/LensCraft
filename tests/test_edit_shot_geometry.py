"""Physical shot-edit scores obey the simulator ROI and calibration contract."""

import math

import numpy as np
import pytest
import torch

from testing.edit_shot_geometry import shot_size_geometry_totals


AFTER = "initial_shot_target_log_occupancy_error"
BEFORE = "reconstruction_initial_shot_target_log_occupancy_error"
GAIN = "initial_shot_target_log_occupancy_improvement"
RATE = "initial_shot_target_log_occupancy_improved_rate"
UNAVAILABLE = "initial_shot_geometry_unavailable_rate"
INVALID = "initial_shot_geometry_invalid_rate"


def example(scale=1.0):
    # A 2-high close-up ROI at z=0 has nearest face z=.5. tan(vfov/2)=1;
    # camera distance 2.5 therefore produces maximum NDC occupancy exactly 1.
    generated = np.array([[[0.0, 1.0, 2.5, 0.0, 0.0, 0.0]]])
    before = generated.copy()
    before[..., 2] = 4.5
    generated[..., :3] *= scale
    before[..., :3] *= scale
    return generated, before, {
        "subject_trajectory": np.zeros((1, 1, 6)),
        "subject_dimensions": np.array([[[2.0, 4.0, 2.0]]]) * scale,
        "intrinsics": np.array([[[17.5, 1.0]]]),
        "valid_mask": np.array([[True]]),
        "shot_size": "closeUp",
    }


@pytest.mark.parametrize("scale", [1.0, 1e-4, 1000.0])
def test_analytic_closeup_improves_and_is_scene_scale_invariant(scale):
    edited, before, kwargs = example(scale)
    result = shot_size_geometry_totals(edited, before, **kwargs)
    assert result[AFTER] == pytest.approx((0.0, 1))
    assert result[BEFORE] == pytest.approx((math.log(2), 1))
    assert result[GAIN] == pytest.approx((math.log(2), 1))
    assert result[RATE] == (1.0, 1)
    assert result[UNAVAILABLE] == (0.0, 1)
    assert result[INVALID] == (0.0, 1)


def test_noop_has_zero_gain_and_worse_edit_has_negative_gain():
    edited, before, kwargs = example()
    noop = shot_size_geometry_totals(before, before, **kwargs)
    assert noop[GAIN] == (0.0, 1)
    assert noop[RATE] == (0.0, 1)
    worse = shot_size_geometry_totals(before, edited, **kwargs)
    assert worse[GAIN] == pytest.approx((-math.log(2), 1))
    assert worse[RATE] == (0.0, 1)


def test_uses_first_valid_frame_once_and_ignores_padded_nonfinite_poses():
    edited, before, kwargs = example()
    edited = np.repeat(edited, 3, axis=1)
    before = np.repeat(before, 3, axis=1)
    edited[:, [0, 2]] = np.nan
    before[:, [0, 2]] = np.nan
    kwargs["valid_mask"] = np.array([[False, True, True]])
    result = shot_size_geometry_totals(edited, before, **kwargs)
    assert result[AFTER] == pytest.approx((0.0, 1))
    assert result[GAIN] == pytest.approx((math.log(2), 1))
    kwargs["valid_mask"][:] = False
    assert all(value == (0.0, 0) for value in shot_size_geometry_totals(edited, before, **kwargs).values())


@pytest.mark.parametrize("field", ["subject_trajectory", "subject_dimensions", "intrinsics"])
def test_missing_geometry_is_unavailable_and_never_a_perfect_score(field):
    edited, before, kwargs = example()
    kwargs[field] = None
    result = shot_size_geometry_totals(edited, before, **kwargs)
    assert result[AFTER] == (0.0, 0)
    assert result[GAIN] == (0.0, 0)
    assert result[UNAVAILABLE] == (1.0, 1)
    assert result[INVALID] == (0.0, 0)


def test_nonfinite_calibration_marks_only_affected_clip_unavailable():
    edited, before, kwargs = example()
    edited, before = np.repeat(edited, 2, axis=0), np.repeat(before, 2, axis=0)
    kwargs = {name: np.repeat(value, 2, axis=0) if isinstance(value, np.ndarray) else value
              for name, value in kwargs.items()}
    kwargs["intrinsics"][1] = np.nan
    result = shot_size_geometry_totals(edited, before, **kwargs)
    assert result[AFTER] == pytest.approx((0.0, 1))
    assert result[UNAVAILABLE] == (1.0, 2)
    assert result[INVALID] == (0.0, 1)


@pytest.mark.parametrize("kind", ["behind", "nonfinite", "nonpositive_dimensions", "nonpositive_lens"])
def test_invalid_generated_pose_or_geometry_does_not_gain_success(kind):
    edited, before, kwargs = example()
    if kind == "behind":
        edited[..., 2] = -2.5
    elif kind == "nonfinite":
        edited[..., 3] = np.nan
    elif kind == "nonpositive_dimensions":
        kwargs["subject_dimensions"][..., 1] = 0
    else:
        kwargs["intrinsics"][..., 0] = 0
    result = shot_size_geometry_totals(edited, before, **kwargs)
    assert result[AFTER] == (0.0, 0)
    assert result[GAIN] == (0.0, 0)
    assert result[RATE] == (0.0, 0)
    assert result[INVALID] == (1.0, 1)
    if kind in ("behind", "nonfinite"):
        assert result[BEFORE][1] == 1


def test_authored_attention_box_uses_subject_local_position_and_rotation():
    # Rotate subject and camera 90deg about Y. Authored offset (1,.5,.25)
    # becomes world (.25,.5,-1); camera local +Z points along world +X.
    edited = np.array([[[2.75, .5, -1, 0, math.pi / 2, 0]]])
    before = edited.copy()
    before[..., 0] += 2
    subject = np.array([[[0, 0, 0, 0, math.pi / 2, 0]]])
    result = shot_size_geometry_totals(
        torch.as_tensor(edited), torch.as_tensor(before),
        subject_trajectory=subject, subject_dimensions=np.array([[[4., 6., 2.]]]),
        intrinsics=np.array([[[17.5, 1.]]]), valid_mask=np.array([[True]]),
        shot_size="closeUp", attention_dimensions=np.array([[2., 1., 1.]]),
        attention_position=np.array([[1., .5, .25]]),
    )
    assert result[AFTER] == pytest.approx((0.0, 1), abs=1e-12)
    assert result[BEFORE] == pytest.approx((math.log(2), 1))


@pytest.mark.parametrize("aspect,distance", [(2.0, 4.5), (0.5, 2.5)])
def test_portrait_and_landscape_intrinsics_match_threejs_long_sensor_edge(aspect, distance):
    edited, _, kwargs = example()
    edited[..., 2] = distance
    kwargs["intrinsics"][..., 1] = aspect
    result = shot_size_geometry_totals(edited, edited, **kwargs)
    assert result[AFTER] == pytest.approx((0.0, 1), abs=1e-12)


@pytest.mark.parametrize("shot,dimensions,z", [
    ("extremeCloseUp", [2., 4., 2.], 1.5),
    ("mediumCloseUp", [2., 4., 2.], 3.125),
    ("mediumShot", [2., 4., 2.], 3.75),
    ("fullShot", [2., 4., 2.], 5.0),
    ("longShot", [2., 4., 2.], 1 + 4 / .75),
    ("veryLongShot", [2., 4., 2.], 9.0),
    ("extremeLongShot", [2., 4., 2.], 1 + 4 / .3),
])
def test_all_simulator_roi_interpolation_weights_and_occupancy_targets(shot, dimensions, z):
    edited, _, kwargs = example()
    edited[..., 2] = z
    kwargs["shot_size"] = shot
    kwargs["subject_dimensions"] = np.array([[dimensions]])
    result = shot_size_geometry_totals(edited, edited, **kwargs)
    assert result[AFTER] == pytest.approx((0.0, 1), abs=1e-12)


def test_malformed_inputs_are_rejected():
    edited, before, kwargs = example()
    with pytest.raises(ValueError, match="both attention"):
        shot_size_geometry_totals(edited, before, attention_dimensions=np.ones((1, 3)), **kwargs)
    with pytest.raises(ValueError, match="Unknown shot_size"):
        shot_size_geometry_totals(edited, before, **{**kwargs, "shot_size": "unknown"})
    with pytest.raises(ValueError, match="valid_mask"):
        shot_size_geometry_totals(edited, before, **{**kwargs, "valid_mask": [[1]]})
    with pytest.raises(ValueError, match="matching"):
        shot_size_geometry_totals(edited, before[:, :, :3], **kwargs)
