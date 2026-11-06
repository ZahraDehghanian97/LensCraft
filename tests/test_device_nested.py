"""Nested physical references must follow the evaluation tensor device."""
from collections import namedtuple

import torch

from utils.device import move_batch_to_device


def test_nested_geometry_masks_follow_camera_device_without_mutation():
    camera = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    padding = torch.zeros((1, 2), dtype=torch.bool)
    reference = {
        "camera_trajectory": camera.clone(),
        "padding_mask": padding.clone(),
        "simulation_normalized": False,
    }
    batch = {
        "camera_trajectory": camera,
        "geometry_reference": reference,
        "simulation_reference": {"geometry_reference": reference},
        "model_observation": [{"profile": "et_compatible_v1", "available": False}],
    }
    # Meta is an actual different tensor device, available on CPU-only runners.
    # A shallow implementation leaves the nested masks on CPU and fails here.
    moved = move_batch_to_device(batch, torch.device("meta"))
    assert moved["camera_trajectory"].device.type == "meta"
    for physical in (moved["geometry_reference"], moved["simulation_reference"]["geometry_reference"]):
        assert physical["padding_mask"].device == moved["camera_trajectory"].device
        assert physical["camera_trajectory"].device.type == "meta"
        assert physical["padding_mask"].dtype == torch.bool
        assert physical["simulation_normalized"] is False
    assert batch["camera_trajectory"].device.type == "cpu"
    assert batch["geometry_reference"] is reference
    assert reference["padding_mask"].device.type == "cpu"
    assert torch.equal(camera, torch.arange(12, dtype=torch.float32).reshape(1, 2, 6))
    assert moved is not batch
    assert moved["simulation_reference"] is not batch["simulation_reference"]
    assert moved["model_observation"] == batch["model_observation"]


def test_nested_lists_tuples_and_namedtuples_preserve_metadata_and_types():
    Pair = namedtuple("Pair", "tensor note")
    value = torch.tensor([3], dtype=torch.long)
    marker = object()
    batch = {"containers": [({"value": value}, Pair(value, "raw GT")), marker, None]}
    moved = move_batch_to_device(batch, torch.device("meta"))
    containers = moved["containers"]
    assert isinstance(containers, list) and containers is not batch["containers"]
    assert isinstance(containers[0], tuple)
    assert isinstance(containers[0][1], Pair)
    assert containers[0][0]["value"].device.type == "meta"
    assert containers[0][1].tensor.device.type == "meta"
    assert containers[0][1].note == "raw GT"
    assert containers[1] is marker and containers[2] is None
    assert value.device.type == "cpu" and value.item() == 3
