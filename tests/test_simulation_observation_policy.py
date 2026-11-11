"""ET-like exports expose incomplete conditions without altering physical GT."""

import copy
import json

import msgpack
import pytest
import torch

from data.simulation.caption import extract_text_prompt
from data.simulation.constants import NumericFeature, cinematography_struct, simulation_struct
from data.simulation.dataset import SimulationDataset, collate_fn
from data.simulation.observation import ModelObservationPolicy
from data.simulation.utils import (
    STRUCTURED_PARAMETERS, cinematography_struct_parameters, structured_conditioning_from_batch,
)


DECLARATION = {
    "profile": "et_compatible_v1",
    "subjectOrientation": "unavailable",
    "subjectDimensions": "default",
    "defaultSubjectDimensions": [.5, 1.7, .3],
    "simulationInstructions": "unavailable",
    "captionPolicy": "motion_only",
}


def complete_annotation(struct):
    result = {}
    for key, kind in struct:
        if isinstance(kind, list):
            result[key] = complete_annotation(kind)
        elif isinstance(kind, NumericFeature):
            result[key] = (kind.minimum + kind.maximum) / 2
        elif kind is bool:
            result[key] = False
        else:
            result[key] = list(kind)[0].value
    return result


def flatten_annotation(value, prefix):
    result = []
    for key, child in value.items():
        path = f"{prefix}__{key}"
        if isinstance(child, dict):
            result.extend(flatten_annotation(child, path))
        else:
            result.append((path, child))
    return result


@pytest.fixture
def export_factory(tmp_path, monkeypatch):
    parameters = {
        "camera_position": {"mean": torch.tensor([5., -2., 3.]), "std": torch.tensor([2., 3., 4.])},
        "subject_position": {"mean": torch.tensor([1., 4., -3.]), "std": torch.tensor([5., 2., 3.])},
        "subject_dimensions": {"mean": torch.tensor([.8, 1.9, .7]), "std": torch.tensor([.2, .6, .5])},
    }
    monkeypatch.setattr(SimulationDataset, "get_normalization_parameters",
                        staticmethod(lambda *args, **kwargs: parameters))
    prompt = complete_annotation(cinematography_struct)
    prompt["movement"] = {"type": "truckRight", "speed": "constant"}
    instruction = complete_annotation(simulation_struct)
    subject = torch.tensor([
        [1., .2, 2., .1, .4, .2], [1.1, .3, 2.1, .2, .6, .3],
        [1.2, .4, 2.2, .3, .8, .4], [1.3, .5, 2.3, .4, 1., .5],
    ])
    camera = torch.tensor([
        [3., 1., 6., .1, .2, .3], [3.1, 1., 6., .1, .25, .3],
        [3.2, 1., 6., .1, .3, .3], [3.3, 1., 6., .1, .35, .3],
    ])
    volume = torch.tensor([[.8, 1.9, .6]])
    count = 0

    def make(*, policy=None, normalize=False, observed_text=None, camera_offset=0., reference_frames=None):
        nonlocal count
        directory = tmp_path / str(count)
        count += 1
        directory.mkdir()
        physical_prompt = copy.deepcopy(prompt)
        if observed_text is not None:
            physical_prompt["observedMotionText"] = observed_text
        prompt_fields = flatten_annotation(physical_prompt, "cinematography")
        instruction_fields = flatten_annotation(instruction, "simulation")
        fields = prompt_fields + instruction_fields
        dictionary = {
            "schemaVersion": 2, "datasetId": "observation-test",
            "keys": [key for key, _ in fields], "values": [[value] for _, value in fields],
        }
        actual_camera = camera.clone()
        actual_camera[:, :3] += camera_offset
        payload = [
            [[index, 0] for index in range(len(prompt_fields))],
            [[index + len(prompt_fields), 0] for index in range(len(instruction_fields))],
            [{"i": "human", "c": "human", "d": volume[0].tolist(), "m": "idle", "f": subject.tolist()}],
            [row + [2., 16 / 9] for row in actual_camera.tolist()],
        ]
        manifest = {"schemaVersion": 2, "datasetId": "observation-test", "config": {"subjectCount": 1, "instructionCount": 1}}
        if policy is not None:
            manifest["modelObservation"] = policy
        (directory / "manifest.json").write_text(json.dumps(manifest))
        (directory / "parameter_dictionary.msgpack").write_bytes(msgpack.packb(dictionary, use_bin_type=True))
        (directory / "simulation_observation-test_000.msgpack").write_bytes(msgpack.packb(payload, use_bin_type=True))
        dataset = SimulationDataset(
            str(directory), normalize=normalize, target_frame_count=4,
            reference_frame_count=reference_frames,
        )
        return dataset, {
            "camera": actual_camera, "subject": subject, "volume": volume,
            "prompt": physical_prompt, "instruction": instruction, "parameters": parameters,
        }

    return make


@pytest.mark.parametrize("normalize", [False, True])
def test_regular_export_keeps_existing_values_and_caption(export_factory, normalize):
    dataset, physical = export_factory(normalize=normalize, observed_text="Profile-specific authored text")
    item = dataset[0]
    expected = (physical["camera"], physical["subject"], physical["volume"])
    if normalize:
        expected = SimulationDataset._normalize_item_with_parameters(*expected, physical["parameters"])
    for key, value in zip(("camera_trajectory", "subject_trajectory", "subject_volume"), expected):
        assert torch.equal(item[key], value)
    assert item["cinematography_ids"].shape == (10,)
    assert item["cinematography_ids"].dtype == torch.long
    assert item["prompt_none_mask"].shape == (10,)
    assert item["prompt_none_mask"].all()
    assert item["structured_presence_mask"].shape == (46,)
    assert item["structured_presence_mask"].all()
    entries = item["cinematography_prompt_parameters"] + item["simulation_instruction_parameters"]
    assert len(item["cinematography_prompt_parameters"]) == 10
    assert len(item["simulation_instruction_parameters"]) == 36
    assert all(entry[3] is None for entry in entries)
    tokens = structured_conditioning_from_batch(collate_fn([item]))
    assert tokens.shape == (10, 1, 21)
    assert torch.equal(tokens.sum(-1), torch.ones(10, 1))
    assert item["raw_prompt"] == physical["prompt"]
    assert item["text_prompt"] == extract_text_prompt(physical["prompt"], "idle")
    assert "geometry_reference" not in item
    assert "subject_orientation_available" not in item
    assert "model_observation" not in collate_fn([item, item])


@pytest.mark.parametrize("normalize", [False, True])
def test_profile_separates_condition_from_full_physical_truth(export_factory, normalize):
    ordinary, _ = export_factory(normalize=normalize)
    dataset, physical = export_factory(policy=DECLARATION, normalize=normalize)
    item = dataset[0]
    original = ordinary[0]
    assert torch.equal(item["camera_trajectory"], original["camera_trajectory"])
    assert torch.equal(item["subject_trajectory"][..., :3], original["subject_trajectory"][..., :3])
    assert torch.count_nonzero(item["subject_trajectory"][..., 3:]) == 0
    expected_volume = torch.tensor([[.5, 1.7, .3]])
    if normalize:
        expected_volume = (expected_volume - physical["parameters"]["subject_dimensions"]["mean"]) / physical["parameters"]["subject_dimensions"]["std"]
    assert torch.equal(item["subject_volume"], expected_volume)
    reference = item["geometry_reference"]
    assert reference["simulation_normalized"] is False
    assert torch.equal(reference["camera_trajectory"], physical["camera"])
    assert torch.equal(reference["subject_trajectory"], physical["subject"])
    assert torch.equal(reference["subject_volume"], physical["volume"])
    assert item["subject_orientation_available"].item() is False
    assert item["subject_dimensions_available"].item() is False
    assert "metadata_only" in item["model_observation"]["subject_availability_encoding"]
    assert item["raw_prompt"] == physical["prompt"]
    assert item["raw_instruction"] == physical["instruction"]
    # Mutating condition placeholders cannot overwrite preserved truth.
    item["subject_volume"].zero_()
    item["subject_trajectory"].zero_()
    assert torch.equal(reference["subject_trajectory"], physical["subject"])
    assert torch.equal(reference["subject_volume"], physical["volume"])


def test_profile_masks_endpoints_and_every_simulator_token(export_factory):
    dataset, _ = export_factory(policy=DECLARATION)
    item = dataset[0]
    fields = [name for name, _ in STRUCTURED_PARAMETERS]
    kept = {fields.index("movement_type"), fields.index("movement_speed")}
    assert item["prompt_none_mask"].shape == (10,)
    assert item["prompt_none_mask"].nonzero().flatten().tolist() == sorted(kept)
    assert item["structured_presence_mask"].shape == (46,)
    assert item["structured_presence_mask"].nonzero().flatten().tolist() == sorted(kept)
    assert len(item["simulation_instruction_parameters"]) == 36
    entries = item["cinematography_prompt_parameters"] + item["simulation_instruction_parameters"]
    for index, (_, _, value_index, vector) in enumerate(entries):
        assert (value_index >= 0) == (index in kept)
        assert vector is None
    high_level_count = len(cinematography_struct_parameters)
    assert not item["structured_presence_mask"][high_level_count:].any()
    tokens = structured_conditioning_from_batch(collate_fn([item]))
    assert tokens.shape == (10, 1, 21)
    assert tokens.sum().item() == 2


def test_profile_can_mark_diagnostic_speed_as_unavailable(export_factory):
    text = "The camera moves right and pans left."
    dataset, physical = export_factory(
        policy={**DECLARATION, "movementSpeed": "unavailable"}, observed_text=text,
    )
    item = dataset[0]
    fields = [name for name, _ in STRUCTURED_PARAMETERS]
    assert item["prompt_none_mask"].nonzero().flatten().tolist() == [fields.index("movement_type")]
    assert item["structured_presence_mask"].nonzero().flatten().tolist() == [fields.index("movement_type")]
    speed = item["cinematography_prompt_parameters"][fields.index("movement_speed")]
    assert speed[1] is None and speed[2] == -1
    assert item["raw_prompt"]["movement"]["speed"] == physical["prompt"]["movement"]["speed"] == "constant"
    assert item["text_prompt"] == text
    assert item["model_observation"]["movement_speed_available"] is False
    assert "diagnostic placeholder" in item["model_observation"]["movement_speed_semantics"]


def test_profile_condition_and_authored_caption_do_not_depend_on_target_camera(export_factory):
    text = "The camera moves right and pans left."
    first, _ = export_factory(policy=DECLARATION, observed_text=text)
    second, _ = export_factory(policy=DECLARATION, observed_text=text, camera_offset=7.)
    left, right = first[0], second[0]
    for key in ("subject_trajectory", "subject_volume", "cinematography_ids",
                "prompt_none_mask", "structured_presence_mask"):
        assert torch.equal(left[key], right[key])
    assert left["cinematography_prompt_parameters"] == right["cinematography_prompt_parameters"]
    assert left["simulation_instruction_parameters"] == right["simulation_instruction_parameters"]
    assert not torch.equal(left["camera_trajectory"], right["camera_trajectory"])
    assert left["text_prompt"] == right["text_prompt"] == text
    assert "front" not in left["text_prompt"]


def test_profile_reference_and_availability_survive_collation_and_compaction(export_factory):
    from data.datamodule import compact_simulation_collate_fn

    dataset, _ = export_factory(policy=DECLARATION, reference_frames=8)
    items = [dataset[0], dataset[0]]
    for batch in (collate_fn(items), compact_simulation_collate_fn(items)):
        assert batch["geometry_reference"]["camera_trajectory"].shape == (2, 4, 6)
        assert batch["geometry_reference"]["subject_trajectory"][..., 3:].abs().sum() > 0
        assert batch["geometry_reference"]["simulation_normalized"] is False
        assert not batch["subject_orientation_available"].any()
        assert not batch["subject_dimensions_available"].any()
        assert batch["subject_orientation_available"].dtype == torch.bool
        assert len(batch["model_observation"]) == 2
        assert batch["simulation_reference"]["subject_trajectory"].shape == (2, 8, 6)
        assert torch.count_nonzero(batch["simulation_reference"]["subject_trajectory"][..., 3:]) == 0
        assert batch["simulation_reference"]["geometry_reference"]["subject_trajectory"][..., 3:].abs().sum() > 0
    ordinary, _ = export_factory()
    with pytest.raises(ValueError, match="cannot mix model observation policies"):
        collate_fn([ordinary[0], items[0]])


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("reference_frames", [None, 8])
def test_geometry_evaluation_uses_full_body_truth_on_the_selected_timeline(
    export_factory, normalize, reference_frames,
):
    from data.sim_format import to_simulation_format
    from testing.process import _update_geometry_metrics
    from testing.metrics.modules.geometry import geometry_metric_totals

    dataset, _ = export_factory(policy=DECLARATION, normalize=normalize, reference_frames=reference_frames)
    batch = collate_fn([dataset[0]])
    length = reference_frames or 4
    view = to_simulation_format(batch, "simulation", target_len=length)
    assert torch.count_nonzero(view[1][..., 3:]) == 0
    reference = (batch["simulation_reference"]["geometry_reference"]
                 if reference_frames is not None else batch["geometry_reference"])

    class CaptureGeometry:
        def update_geometry_metrics(self, item, generated, target, **kwargs):
            self.item, self.generated, self.target, self.kwargs = item, generated, target, kwargs
            self.totals = geometry_metric_totals(generated, target, **kwargs)

    callback = CaptureGeometry()
    _update_geometry_metrics(callback, "prompt_generation", view[0], view, batch)
    assert torch.equal(callback.target, reference["camera_trajectory"])
    assert torch.equal(callback.kwargs["subject_trajectory"], reference["subject_trajectory"])
    assert torch.equal(callback.kwargs["subject_dimensions"], reference["subject_volume"])
    assert torch.equal(callback.kwargs["intrinsics"], reference["camera_intrinsics"])
    torch.testing.assert_close(callback.generated, reference["camera_trajectory"], atol=1e-6, rtol=1e-6)
    placeholder_subject, placeholder_volume = view[1], view[2]
    if normalize:
        _, placeholder_subject, placeholder_volume = SimulationDataset.normalize_item(
            view[0], placeholder_subject, placeholder_volume, False,
        )
    placeholder = geometry_metric_totals(
        callback.generated, callback.target,
        subject_trajectory=placeholder_subject, subject_dimensions=placeholder_volume,
        intrinsics=callback.kwargs["intrinsics"],
    )
    assert callback.totals["subject_bbox_width"][1] == length
    assert abs(callback.totals["subject_bbox_width"][0] - placeholder["subject_bbox_width"][0]) > 1e-5


@pytest.mark.parametrize("changes", [
    {"profile": "future_profile"}, {"subjectOrientation": "measured"},
    {"subjectDimensions": "observed"}, {"simulationInstructions": "available"},
    {"captionPolicy": "all_fields"}, {"defaultSubjectDimensions": [0., 1., 1.]},
    {"defaultSubjectDimensions": [True, 1., 1.]}, {"defaultSubjectDimensions": [float("nan"), 1., 1.]},
    {"defaultSubjectDimensions": [.6, 1.7, .3]}, {"defaultSubjectDimensions": [.5, 1.7]},
    {"movementSpeed": "observed"},
])
def test_invalid_observation_contract_fails_before_loading(changes):
    with pytest.raises(ValueError, match="modelObservation|et_compatible_v1"):
        ModelObservationPolicy.from_manifest({"modelObservation": {**DECLARATION, **changes}})


def test_absent_policy_is_opt_in_and_malformed_policy_is_not_silently_ignored():
    assert ModelObservationPolicy.from_manifest(None) is None
    assert ModelObservationPolicy.from_manifest({}) is None
    with pytest.raises(ValueError, match="must be an object"):
        ModelObservationPolicy.from_manifest({"modelObservation": None})


@pytest.mark.parametrize("text", ["", "  ", 17, None])
def test_authored_caption_must_be_a_nonempty_string(text):
    policy = ModelObservationPolicy.from_manifest({"modelObservation": DECLARATION})
    with pytest.raises(ValueError, match="non-empty string"):
        policy.observed_motion_text({"observedMotionText": text})
