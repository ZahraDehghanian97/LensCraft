"""The native caption axes must agree with simulation's subject-local views."""
import math

import pytest
import torch

from data.ccdm.convertor import CCDMConvertor
from data.simulation.convertor import SIMConvertor


@pytest.mark.parametrize("position, angle", [
    ([0., 0., 5.], 0.), ([5., 0., 0.], -math.pi / 2),
    ([-5., 0., 0.], math.pi / 2), ([0., 0., -5.], math.pi),
])
def test_native_front_left_right_back_match_simulation_subject_frame(position, angle):
    native = torch.tensor([[*position, 0., 0.]])
    camera, subject, volume = CCDMConvertor().to_standard(native)
    local_position = subject[0, :3, :3].T @ camera[0, :3, 3]
    actual = torch.atan2(local_position[0], -local_position[2])
    residual = torch.atan2(torch.sin(actual - angle), torch.cos(actual - angle))
    assert abs(residual.item()) < 1e-6
    # Subject-frame correction never changes camera world coordinates.
    assert torch.equal(camera[0, :3, 3], native[0, :3])
    assert torch.det(subject[0, :3, :3]).item() == 1
    restored = CCDMConvertor().from_standard(camera, subject, volume)[0]
    torch.testing.assert_close(restored, native, atol=1e-5, rtol=1e-5)


def test_centered_positive_z_camera_is_upright_and_subject_heading_roundtrips():
    camera, subject, volume = CCDMConvertor().to_standard(torch.tensor([[0., 0., 5., 0., 0.]]))
    sim_camera, sim_subject, sim_volume = SIMConvertor().from_standard(camera, subject, volume)
    torch.testing.assert_close(sim_camera[0, 3:], torch.zeros(3), atol=1e-6, rtol=0)
    restored_camera, restored_subject, _ = SIMConvertor().to_standard(sim_camera, sim_subject, sim_volume)
    torch.testing.assert_close(restored_camera, camera, atol=1e-6, rtol=0)
    torch.testing.assert_close(restored_subject, subject, atol=1e-6, rtol=0)


def test_placement_rotates_positions_and_aim_with_translating_turning_subject():
    from data.ccdm.alignment import align_ccdm_to_subject
    from utils.pytorch3d_transform import euler_angles_to_matrix

    camera = torch.tensor([[[0., 0., 5., 0., 0., 0.]]]).repeat(2, 3, 1)
    subject = torch.tensor([[[2., 1., 3., 0., 0., 0.],
                             [4., 1., 6., 0., 0.8, 0.],
                             [8., 2., 9., 0., math.pi, 0.]]]).repeat(2, 1, 1)
    original = camera.clone()
    placed = align_ccdm_to_subject(camera, subject)
    rotation = euler_angles_to_matrix(subject[..., 3:], "XYZ")
    expected = subject[..., :3] + (rotation @ camera.new_tensor([0., 0., -5.]))
    torch.testing.assert_close(placed[..., :3], expected, atol=1e-5, rtol=1e-5)
    transform, subject_transform, _ = SIMConvertor().to_standard(placed, subject)
    toward_subject = subject[..., :3] - placed[..., :3]
    torch.testing.assert_close(transform[..., :3, 2], toward_subject / 5, atol=1e-5, rtol=1e-5)
    native = CCDMConvertor().from_standard(transform, subject_transform)[0]
    torch.testing.assert_close(native, camera.new_tensor([0., 0., 5., 0., 0.]).expand(2, 3, 5),
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(camera, original)


def test_native_subject_placement_is_identity():
    from data.ccdm.alignment import align_ccdm_to_subject
    native = torch.tensor([[[0., 0., 5., 0., 0.], [3., 1., 5., 0.2, -0.1]]])
    camera, subject, volume = CCDMConvertor().to_standard(native)
    sim, sub, _ = SIMConvertor().from_standard(camera, subject, volume)
    placed = align_ccdm_to_subject(sim, sub)
    restored = SIMConvertor().to_standard(placed)[0]
    torch.testing.assert_close(restored, camera, atol=1e-5, rtol=1e-5)


def test_evaluation_and_inference_apply_same_ccdm_subject_placement(monkeypatch):
    from data.ccdm.alignment import align_ccdm_to_subject
    from data.simulation.dataset import SimulationDataset
    from testing import process as evaluation
    from inferencing import process as inference
    from testing.baseline_modes import NORM_ITEM, NO_NORM_ITEM

    camera = torch.zeros(1, 3, 6)
    native = camera.clone()
    native[..., 2] = 5
    subject = camera.clone()
    subject[..., 0] = torch.tensor([1., 2., 3.])
    subject[..., 4] = torch.tensor([0., math.pi / 2, math.pi])
    padding = torch.zeros(1, 3, dtype=torch.bool)
    view = (camera, subject, None, padding)
    monkeypatch.setattr(SimulationDataset, "normalize_item", staticmethod(lambda c, s, v, n: (c, s, v)))
    for module in (evaluation, inference):
        if hasattr(module, "generate_baseline_trajectory"):
            monkeypatch.setattr(module, "generate_baseline_trajectory", lambda *a, **kw: (native.clone(), None))
        else:
            # The archived server revision has inline baseline generation.
            monkeypatch.setattr(module, "convert_to_target", lambda source, *a, **kw:
                                (native.clone() if source == "ccdm" else camera, subject, None, padding))
            if hasattr(module, "_to_sim_space"):
                monkeypatch.setattr(module, "_to_sim_space", lambda *a, **kw: native.clone())
        monkeypatch.setattr(module, "to_simulation_format", lambda *a, **kw: view)
    from types import SimpleNamespace
    model = SimpleNamespace(generate_using_text=lambda *a, **kw: native.clone())
    batch = {"text_prompts": ["front view"], "camera_trajectory": camera,
             "subject_trajectory": subject, "subject_volume": None, "padding_mask": padding}
    result = evaluation._generate_baseline_variants(
        None, model, batch, [NORM_ITEM, NO_NORM_ITEM], "simulation", "ccdm", 3, camera, subject, 1)
    expected = align_ccdm_to_subject(native, subject)
    torch.testing.assert_close(result[NORM_ITEM], expected)
    torch.testing.assert_close(result[NO_NORM_ITEM], native)
    inferred = inference.inference_batch(model, batch,
                                         "cpu", model_type="ccdm", seq_length=3)[0]
    torch.testing.assert_close(inferred["prompt_generation"], expected)


def test_coordinate_cache_version_is_scoped_to_ccdm(monkeypatch, tmp_path):
    from testing import trajectory_cache as cache
    def config(model, dataset):
        return {"training": {"model": {"data_format": {"type": model}}},
                "data": {"dataset": {"config": {"_target_": dataset}}}}
    configs = [config("ccdm", "SimulationDataset"), config("simulation", "CCDMDataset"),
               config("simulation", "SimulationDataset")]
    before = [cache.build_trajectory_cache_key(c) for c in configs]
    monkeypatch.setattr(cache, "CCDM_COORDINATE_FRAME_VERSION", 999)
    after = [cache.build_trajectory_cache_key(c) for c in configs]
    assert before[0] != after[0] and before[1] != after[1] and before[2] == after[2]
    annotation = tmp_path / "annotations.json"
    annotation.write_text("[]")
    configs[1]["data"]["dataset"]["config"]["annotations_path"] = str(annotation)
    first = cache.build_trajectory_cache_key(configs[1])
    annotation.write_text('[{"changed": true}]')
    assert cache.build_trajectory_cache_key(configs[1]) != first
