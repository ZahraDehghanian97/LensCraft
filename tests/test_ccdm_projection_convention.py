"""Independent screen-ray oracle from publisher Unity camcontrol.cs, not a round trip."""
import math

import pytest
import torch

from data.ccdm.convertor import CCDMConvertor


@pytest.mark.parametrize('screen_x', [-.5, 0., .5])
@pytest.mark.parametrize('screen_y', [-.5, 0., .5])
def test_publisher_screen_y_down_and_x_left_in_standard_camera(screen_x, screen_y):
    converter = CCDMConvertor()
    feature = torch.tensor([[0., 0., 5., screen_x, screen_y]], dtype=torch.float64)
    pose = converter.convert_ccdm_to_transform(feature)[0]
    ray_cv = pose[:3, :3].T @ -pose[:3, 3]
    assert ray_cv[2] > 0
    # Unity normalizes Screen.height - pixel_y. OpenCV camera Y is also down.
    assert ray_cv[1] / ray_cv[2] == pytest.approx(screen_y * math.tan(math.pi / 8), abs=1e-12)
    # Publisher replay negates sx; OpenCV camera X is right.
    assert ray_cv[0] / ray_cv[2] == pytest.approx(-screen_x * math.tan(math.pi / 8) * 16 / 9, abs=1e-12)
    torch.testing.assert_close(pose[:3, :3].T @ pose[:3, :3], torch.eye(3, dtype=torch.float64))
    assert torch.det(pose[:3, :3]).item() == pytest.approx(1.)


def test_vertical_default_and_explicit_horizontal_override():
    default = CCDMConvertor()
    assert default.vfov_deg == 45.
    assert default.hfov_deg == pytest.approx(72.734351410049, abs=1e-10)
    explicit = CCDMConvertor(hfov_deg=45.)
    assert explicit.hfov_deg == 45.
    assert explicit.vfov_deg == pytest.approx(2 * math.degrees(math.atan(math.tan(math.pi / 8) * 9 / 16)))
    with pytest.raises(ValueError):
        CCDMConvertor(hfov_deg=45., vfov_deg=45.)
