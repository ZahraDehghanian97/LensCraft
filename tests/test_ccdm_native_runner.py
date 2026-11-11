"""Native scorer writes provenance and never substitutes missing evaluator assets."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from testing.ccdm_native import CCDMNativeClassifier


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_ccdm_native.py"
    spec = importlib.util.spec_from_file_location("evaluate_ccdm_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_classifier_assets_are_recorded_without_scores(tmp_path):
    output = tmp_path / "missing"
    result = runner().main([
        "--classifier-checkpoint", str(tmp_path / "classifier.pth"),
        "--classifier-data", str(tmp_path / "data.npy"),
        "--generated", str(tmp_path / "generated.npz"), "--output", str(output),
    ])
    assert result == 2
    report = json.loads((output / "status.json").read_text())
    assert report["status"] == "missing_assets" and len(report["missing"]) == 3
    assert not (output / "results.json").exists()


def test_native_scorer_end_to_end_with_unlabeled_numeric_artifact(tmp_path):
    # Tiny synthetic data verifies the scorer, not any scientific model score.
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = CCDMNativeClassifier(frames=6)
    with torch.no_grad():
        model.output_process[0].bias.fill_(8)
    checkpoint = tmp_path / "classifier.pth"
    torch.save(model.state_dict(), checkpoint)
    rng = np.random.default_rng(7)
    split = tmp_path / "split.npz"
    train, test = rng.normal(size=(12, 6, 5)), rng.normal(size=(6, 6, 5))
    np.savez(split, train_cam=train, train_label=np.tile(np.arange(6), 2),
             test_cam=test, test_label=np.arange(6))
    module = runner()
    all_frames = np.concatenate(list(train) + list(test), axis=0)
    manifest = tmp_path / "evaluator_manifest.json"
    manifest.write_text(json.dumps({
        "classifier_checkpoint": module.file_identity(checkpoint),
        "classifier_data": module.file_identity(split), "frames": 6,
        "normalization": {"mean": all_frames.mean(axis=0).tolist(), "std": all_frames.std(axis=0).tolist()},
    }))
    generated = tmp_path / "generated.npz"
    np.savez(generated, trajectories=rng.normal(size=(8, 6, 5)))
    output = tmp_path / "scored"
    args = ["--classifier-checkpoint", str(checkpoint), "--classifier-data", str(split),
            "--generated", str(generated), "--output", str(output),
            "--classifier-origin", "retrained", "--training-manifest", str(manifest),
            "--diversity-pairs", "100"]
    assert runner().main(args) == 0
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "complete"
    assert report["accuracy"] is None and report["multimodality"] is None
    assert report["classifier_origin_user_declared"] == "retrained"
    assert report["paper_table_reproduction_verified"] is False
    assert report["matched_cohort_verified"] is False and report["matched_cohort"] is None
    assert report["classifier"]["frames"] == 6
    assert np.isfinite(report["fid_train"]) and np.isfinite(report["diversity_generated"])
    assert report["inputs"]["classifier_checkpoint"]["sha256"]
    assert (output / "classifier_features.npz").is_file()
    with pytest.raises(FileExistsError):
        runner().main(args)
    metadata = json.loads(manifest.read_text())
    metadata["classifier_data"]["sha256"] = "wrong-split"
    manifest.write_text(json.dumps(metadata))
    args[args.index("--output") + 1] = str(tmp_path / "mismatched")
    with pytest.raises(ValueError, match="does not match classifier_data"):
        runner().main(args)


@pytest.fixture
def matched_binding(tmp_path):
    module = runner()
    data = {"test_source_indices": np.arange(20, 26), "test_label": np.arange(6),
            "train_source_indices": np.arange(12), "train_label": np.tile(np.arange(6), 2)}
    generated = np.zeros((4, 6, 5), dtype=np.float32)
    expected = np.array([0, 0, 1, 1])
    payload = {"trajectories": generated, "labels": expected, "source_indices": np.array([20, 20, 21, 21])}
    generated_path, split_path = tmp_path / "generated.npz", tmp_path / "split.npz"
    np.savez(generated_path, **payload)
    np.savez(split_path, **data)
    identities = {"generated": module.file_identity(generated_path),
                  "classifier_data": module.file_identity(split_path)}
    args = SimpleNamespace(cohort=tmp_path / "cohort.json", generation_manifest=tmp_path / "complete.json")
    cohort = {"protocol": "ccdm-native-shared-cohort-v1", "source_indices": [20, 21], "labels": [0, 1],
              "classifier_data_sha256": identities["classifier_data"]["sha256"],
              "data_manifest_sha256": "a" * 64, "source_data_sha256": "b" * 64}
    args.cohort.write_text(json.dumps(cohort))
    generation = {"protocol": "ccdm-native-matched-generation-v1", "state": "complete",
                  "generated": identities["generated"], "cohort": module.file_identity(args.cohort),
                  "source_data": {"sha256": "b" * 64}, "repeats": 2, "source_draws": 2,
                  "output_count": 4, "processed_samples": 4, "frames": 6, "target_camera_used": 0}
    args.generation_manifest.write_text(json.dumps(generation))
    return SimpleNamespace(module=module, args=args, data=data, payload=payload, generated=generated,
                           expected=expected, identities=identities, cohort=cohort, generation=generation,
                           training={"training": {"data_manifest_sha256": "a" * 64}})


def verify_binding(case):
    return case.module.verify_matched_cohort(
        case.args, case.data, case.payload, case.generated, case.expected, case.identities, case.training, 6,
    )


def test_matched_binding_preserves_ordered_repeats_and_records_provenance(matched_binding):
    case = matched_binding
    result = verify_binding(case)
    assert result == {"source_draws": 2, "repeats": 2, "generated_count": 4,
                      "classifier_heldout_verified": True, "generator_heldout_verified": False}
    assert case.identities["cohort"] == case.module.file_identity(case.args.cohort)
    assert case.identities["generation_manifest"] == case.module.file_identity(case.args.generation_manifest)


@pytest.mark.parametrize("field,value", [
    ("state", "running"), ("output_count", 3), ("processed_samples", 3),
    ("frames", 5), ("target_camera_used", 1), ("target_camera_used", False),
    ("repeats", 0), ("source_draws", 3),
    ("generated", {"sha256": "changed"}), ("cohort", {"sha256": "changed"}),
    ("source_data", {"sha256": "changed"}),
])
def test_matched_binding_rejects_incomplete_or_changed_generation(matched_binding, field, value):
    case = matched_binding
    case.generation[field] = value
    case.args.generation_manifest.write_text(json.dumps(case.generation))
    with pytest.raises(ValueError):
        verify_binding(case)


@pytest.mark.parametrize("field", ["classifier_data_sha256", "data_manifest_sha256"])
def test_matched_binding_rejects_other_classifier_preparation(matched_binding, field):
    case = matched_binding
    case.cohort[field] = "changed"
    case.args.cohort.write_text(json.dumps(case.cohort))
    case.generation["cohort"] = case.module.file_identity(case.args.cohort)
    case.args.generation_manifest.write_text(json.dumps(case.generation))
    with pytest.raises(ValueError, match="binding differs"):
        verify_binding(case)


@pytest.mark.parametrize("tamper", ["order", "labels", "train_source", "split_overlap"])
def test_matched_binding_rejects_changed_rows_or_nonheldout_ids(matched_binding, tamper):
    case = matched_binding
    if tamper == "order":
        case.payload["source_indices"] = np.array([20, 21, 20, 21])
    elif tamper == "labels":
        case.expected[0] = 5
    elif tamper == "train_source":
        case.cohort["source_indices"][0] = 0
        case.payload["source_indices"][:2] = 0
        case.args.cohort.write_text(json.dumps(case.cohort))
        case.generation["cohort"] = case.module.file_identity(case.args.cohort)
        case.args.generation_manifest.write_text(json.dumps(case.generation))
    else:
        case.data["train_source_indices"][0] = 20
    with pytest.raises(ValueError):
        verify_binding(case)


@pytest.mark.parametrize("option", ["--cohort", "--generation-manifest"])
def test_matched_arguments_are_required_together(tmp_path, option):
    with pytest.raises(SystemExit):
        runner().parse_args([
            "--classifier-checkpoint", "classifier.pth", "--classifier-data", "split.npz",
            "--generated", "generated.npz", "--output", str(tmp_path / "output"),
            option, "manifest.json",
        ])
