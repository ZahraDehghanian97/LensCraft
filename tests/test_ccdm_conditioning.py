"""Native CCDM IDs, caption alignment and masks must survive cohort selection."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.ccdm.dataset import CCDMDataset, collate_fn
from data.simulation.constants import cinematography_struct, simulation_struct
from data.simulation.utils import structured_conditioning_from_batch


def null_struct(struct):
    return {key: null_struct(value) if isinstance(value, list) else None for key, value in struct}


@pytest.fixture
def native_data(tmp_path, monkeypatch):
    monkeypatch.setattr(CCDMDataset, "_normalization_parameters", None)
    np.save(tmp_path / "Mean_Std.npy", {"Mean": np.zeros(5), "Std": np.ones(5)})
    np.save(tmp_path / "data.npy", {"cam": [np.ones((i + 2, 5)) * i for i in range(3)],
                                    "info": [["static"], ["unused"], ["dolly"]]})
    (tmp_path / "indices.json").write_text(json.dumps([2, 0]))
    rows = []
    for i, text, movement in [(0, "static", "static"), (2, "dolly", "dollyIn")]:
        prompt = null_struct(cinematography_struct)
        prompt["movement"]["type"] = movement
        rows.append({"custom_id": f"ccdm-{i:06d}", "source_prompt": text,
                     "cinematographyPrompts": [prompt],
                     "simulationInstructions": [null_struct(simulation_struct)]})
    (tmp_path / "annotations.json").write_text(json.dumps(rows))
    return tmp_path


def dataset(path, annotated=True):
    return CCDMDataset(str(path / "data.npy"), device="cpu",
                       original_seq_len=6, sample_indices_path=str(path / "indices.json"),
                       annotations_path=str(path / "annotations.json") if annotated else None)


def test_selected_ids_caption_tokens_and_padding_remain_aligned(native_data):
    ds = dataset(native_data)
    batch = collate_fn([ds[0], ds[1]])
    assert batch["item_ids"] == ["ccdm-000002", "ccdm-000000"]
    assert batch["text_prompts"] == ["dolly", "static"]
    assert (~batch["padding_mask"]).sum(1).tolist() == [4, 2]
    assert batch["cinematography_ids"].shape == (2, 10)
    assert batch["cinematography_ids"].dtype == torch.long
    assert batch["prompt_none_mask"].sum(1).tolist() == [1, 1]
    assert batch["structured_presence_mask"].shape == (2, 46)
    assert batch["structured_presence_mask"].sum(1).tolist() == [1, 1]
    tokens = structured_conditioning_from_batch(batch)
    assert tokens.shape == (10, 2, 21)
    assert batch["cinematography_ids"][:, 4].tolist() == [3, 0]
    assert tokens[4, 0, 3] == tokens[4, 1, 0] == 1
    assert torch.equal(tokens.sum(-1).T, batch["prompt_none_mask"].float())
    assert "caption_feat" not in batch
    assert torch.equal(batch["camera_trajectory"][0, -1], batch["camera_trajectory"][0, 3])


@pytest.mark.parametrize("movement", ["craneUp", "dollyIn", "arcLeft"])
def test_legacy_annotations_keep_original_fields_bytes_and_conditioning(native_data, movement):
    path = native_data / "annotations.json"
    rows = json.loads(path.read_text())
    row = rows[1]
    row["cinematographyPrompts"][0]["movement"]["type"] = movement
    row["simulationInstructions"][0]["dynamic"]["type"] = "interpolation"
    path.write_text(json.dumps(rows))
    before = path.read_bytes()

    ds = dataset(native_data)
    item = ds[0]
    batch = collate_fn([item, ds[1]])
    assert path.read_bytes() == before
    assert ds.annotations["ccdm-000002"] == row
    assert item["raw_prompt"] == row["cinematographyPrompts"][0]
    assert item["raw_instruction"] == row["simulationInstructions"][0]
    dynamic = next(parameter for parameter in item["simulation_instruction_parameters"]
                   if parameter[0] == "dynamic_type")
    assert dynamic[1] == "interpolation" and dynamic[2] >= 0 and dynamic[3] is None
    assert len(item["cinematography_prompt_parameters"]) == 10
    assert len(item["simulation_instruction_parameters"]) == 36
    assert all(parameter[3] is None for parameter in (
        item["cinematography_prompt_parameters"] + item["simulation_instruction_parameters"]
    ))
    assert batch["item_ids"] == ["ccdm-000002", "ccdm-000000"]
    assert batch["text_prompts"] == ["dolly", "static"]
    assert (~batch["padding_mask"]).sum(1).tolist() == [4, 2]
    assert batch["prompt_none_mask"].sum(1).tolist() == [1, 1]
    assert batch["structured_presence_mask"].sum(1).tolist() == [2, 1]
    tokens = structured_conditioning_from_batch(batch)
    assert tokens.shape == (10, 2, 21)
    assert tokens.sum().item() == 2
    assert torch.equal(batch["camera_trajectory"][0, -1], batch["camera_trajectory"][0, 3])


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_text", "invalid_schema",
                                      "missing_field", "extra_field", "wrong_type", "nonfinite"])
def test_annotations_fail_closed(native_data, problem):
    path = native_data / "annotations.json"
    rows = json.loads(path.read_text())
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "wrong_text":
        rows[0]["source_prompt"] = "different caption"
    elif problem == "invalid_schema":
        rows[0]["cinematographyPrompts"][0]["movement"]["type"] = "invented"
    elif problem == "missing_field":
        del rows[0]["simulationInstructions"][0]["dynamic"]["direction"]
    elif problem == "extra_field":
        rows[0]["simulationInstructions"][0]["dynamic"]["unsupported"] = None
    elif problem == "wrong_type":
        rows[0]["simulationInstructions"][0]["constraints"]["allFramesVisibility"] = "true"
    else:
        rows[0]["simulationInstructions"][0]["constraints"]["maxSpeed"] = float("inf")
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        dataset(native_data)


@pytest.mark.parametrize("indices", [[0, 0], [3], [-1], [True], []])
def test_invalid_cohort_indices_rejected(native_data, indices):
    (native_data / "indices.json").write_text(json.dumps(indices))
    with pytest.raises(ValueError, match="indices"):
        dataset(native_data)


def test_original_unannotated_dataset_still_works(native_data):
    ds = dataset(native_data, annotated=False)
    batch = collate_fn([ds[0], ds[1]])
    assert batch["cinematography_ids"].shape == (2, 10)
    assert torch.all(batch["cinematography_ids"] == -1)
    assert batch["prompt_none_mask"].shape == (2, 10)
    assert not batch["prompt_none_mask"].any()
    assert batch["structured_presence_mask"].shape == (2, 46)
    assert not batch["structured_presence_mask"].any()
    tokens = structured_conditioning_from_batch(batch)
    assert tokens.shape == (10, 2, 21)
    assert torch.count_nonzero(tokens) == 0
    assert "caption_feat" not in batch
    assert len(batch["text_prompts"]) == 2


@pytest.mark.parametrize("annotated", [False, True])
def test_ccdm_inference_preserves_structured_caption_and_presence(native_data, monkeypatch, annotated):
    from inferencing import process as inference

    ds = dataset(native_data, annotated=annotated)
    batch = collate_fn([ds[0], ds[1]])
    batch["random_prompt_index"] = [1, 0]
    camera = torch.zeros(2, 6, 6)
    subject = torch.zeros_like(camera)
    volume = torch.ones(2, 1, 3)
    padding = batch["padding_mask"]
    monkeypatch.setattr(inference, "to_simulation_format", lambda *args, **kwargs:
                        (camera, subject, volume, padding))

    class CaptureModel:
        def __init__(self):
            self.calls = []

        def generate_camera_trajectory(self, **kwargs):
            self.calls.append(kwargs)
            return {"reconstructed": camera.clone()}

    model = CaptureModel()
    results, *_ = inference.inference_batch(
        model, batch, "cpu", dataset_type="ccdm", seq_length=6,
        num_keyframes=1, keyframe_sample_seeds=[42, 43],
    )
    prompt = model.calls[0]
    assert prompt["memory_teacher_forcing_ratio"] == 1.
    expected = structured_conditioning_from_batch(batch)
    assert expected.shape == (10, 2, 21)
    torch.testing.assert_close(prompt["caption_embedding"], expected)
    torch.testing.assert_close(prompt["caption_presence_mask"], batch["prompt_none_mask"])
    # The different-prompt control must shuffle the annotation mask too.
    control = model.calls[-1]
    torch.testing.assert_close(control["caption_embedding"], expected[:, [1, 0]])
    torch.testing.assert_close(control["caption_presence_mask"], batch["prompt_none_mask"][[1, 0]])
    assert "source_trajectory" in results
    if not annotated:
        assert all(torch.count_nonzero(call["caption_embedding"]) == 0 for call in model.calls)
        assert all(not call["caption_presence_mask"].any() for call in model.calls)


def test_caption_modes_enabled_only_with_structured_annotations():
    # Load just the mode selection function to avoid importing the full GPU runner.
    import ast
    from testing.keyframes import DEFAULT_KEYFRAME_COUNTS, keyframe_mode, validate_keyframe_counts
    import testing.baseline_modes as modes

    source = ast.parse((Path(__file__).parents[1] / "src/test.py").read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_select_metric_items")
    scope = dict(BASELINE_MODELS=("ccdm", "et", "gendop"), CAPTIONED_DATASETS=("simulation", "et"),
                 DEFAULT_KEYFRAME_COUNTS=DEFAULT_KEYFRAME_COUNTS, keyframe_mode=keyframe_mode,
                 validate_keyframe_counts=validate_keyframe_counts,
                 BASELINE_ITEMS=modes.BASELINE_ITEMS, NORM_ITEM=modes.NORM_ITEM)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "mode_selection", "exec"), scope)
    select = scope["_select_metric_items"]
    assert "prompt_generation" not in select("lens_craft", "ccdm", False)
    annotated = select("lens_craft", "ccdm", False, structured_captions=True)
    assert len(annotated) == 13
    assert "prompt_generation" in annotated and "key_framing+prompt_k4" in annotated
