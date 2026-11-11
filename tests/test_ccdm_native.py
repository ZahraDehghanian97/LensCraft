"""Native CCDM metrics agree with independently executed released definitions."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import linalg
from torch import nn

from testing.ccdm_native import (
    CCDMNativeClassifier, calculate_diversity, evaluate_native_features,
    extract_native_features, frechet_distance, load_classifier, prepare_native_trajectories,
)


UPSTREAM = Path(__file__).resolve().parents[1] / "third_parties/Camera-control/[2024][EG]Text+keyframe"


def released_definitions(filename, names):
    tree = ast.parse((UPSTREAM / filename).read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = {"np": np, "torch": torch, "nn": nn, "linalg": linalg}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(UPSTREAM / filename), "exec"), namespace)
    return namespace


def test_classifier_matches_exact_upstream_features_logits_and_keys(tmp_path):
    definitions = released_definitions("classify.py", {
        "PositionalEncoding", "TimestepEmbedder", "Transformer",
    })
    torch.manual_seed(81)
    upstream = definitions["Transformer"](n_feature=5, n_label=6, sliding_wind=60).eval()
    ours = CCDMNativeClassifier(frames=60).eval()
    assert set(ours.state_dict()) == set(upstream.state_dict())
    ours.load_state_dict(upstream.state_dict(), strict=True)
    x = torch.randn(2, 60, 5)
    with torch.inference_mode():
        reference_features = upstream.forward_feature(x)
        reference_logits = upstream(x)
        torch.testing.assert_close(ours.forward_feature(x), reference_features, rtol=0, atol=0)
        torch.testing.assert_close(ours(x), reference_logits, rtol=0, atol=0)
    checkpoint = tmp_path / "classifier.pth"
    torch.save(upstream.state_dict(), checkpoint)
    loaded, metadata = load_classifier(checkpoint)
    assert metadata["frames"] == 60 and metadata["feature_dim"] == 60
    assert len(metadata["sha256"]) == 64
    extracted = extract_native_features(loaded, x, batch_size=2)
    np.testing.assert_allclose(extracted["features"], reference_features.numpy(), rtol=0, atol=0)
    np.testing.assert_array_equal(extracted["predictions"], reference_logits.argmax(dim=1).numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_normalization_then_padding_with_no_length_interpolation(dtype):
    rng = np.random.RandomState(1)
    raw = rng.normal(size=(3, 5)).astype(dtype)
    mean = rng.normal(size=5).astype(dtype)
    std = rng.uniform(size=5).astype(dtype)
    prepared = prepare_native_trajectories([raw, raw[:1]], mean, std, expected_frames=4)
    normalized = (raw - mean) / (std + 1e-8)
    np.testing.assert_array_equal(prepared[0, :3], normalized.astype(np.float32))
    np.testing.assert_array_equal(prepared[0, 3], prepared[0, 2])
    np.testing.assert_array_equal(prepared[1], np.repeat(prepared[1, :1], 4, axis=0))
    assert prepared.dtype == np.float32
    with pytest.raises(ValueError, match="resampling/truncation"):
        prepare_native_trajectories([raw], mean, std, expected_frames=2)


@pytest.mark.parametrize("trajectories,mean,std", [
    ([], np.zeros(5), np.ones(5)),
    ([np.zeros((0, 5))], np.zeros(5), np.ones(5)),
    ([np.zeros((3, 6))], np.zeros(5), np.ones(5)),
    ([np.full((3, 5), np.nan)], np.zeros(5), np.ones(5)),
    ([np.zeros((3, 5))], np.zeros(4), np.ones(5)),
    ([np.zeros((3, 5))], np.zeros(5), -np.ones(5)),
])
def test_invalid_native_inputs_fail(trajectories, mean, std):
    with pytest.raises(ValueError):
        prepare_native_trajectories(trajectories, mean, std)


def test_released_metric_parity_and_train_feature_standardization():
    definitions = released_definitions("metric.py", {
        "calculate_frechet_distance", "calculate_activation_statistics", "calculate_diversity",
    })
    rng = np.random.RandomState(91)
    train = rng.normal(size=(30, 5)).astype(np.float32)
    generated = rng.normal(2, 3, size=(20, 5)).astype(np.float32)
    reference = rng.normal(0, 2, size=(20, 5)).astype(np.float32)
    mean, std = train.mean(0), train.std(0)
    train_z, gen_z = (train - mean) / std, (generated - mean) / std
    mu_train, cov_train = definitions["calculate_activation_statistics"](train_z)
    mu_gen, cov_gen = definitions["calculate_activation_statistics"](gen_z)
    expected_fid = definitions["calculate_frechet_distance"](mu_train, cov_train, mu_gen, cov_gen)
    first, second = rng.randint(0, len(gen_z), 100), rng.randint(0, len(gen_z), 100)
    expected_diversity = float(definitions["calculate_diversity"](gen_z, first, second))
    assert calculate_diversity(gen_z, first, second) == expected_diversity
    report = evaluate_native_features(train, generated, reference_features=reference, seed=13, diversity_pairs=100)
    assert report["fid_train"] == pytest.approx(expected_fid, rel=1e-7)
    paired_rng = np.random.RandomState(13)
    paired_first = paired_rng.randint(0, len(gen_z), 100)
    paired_second = paired_rng.randint(0, len(gen_z), 100)
    assert report["diversity_generated"] == float(definitions["calculate_diversity"](gen_z, paired_first, paired_second))
    ref_z = (reference - mean) / std
    expected_test_fid = definitions["calculate_frechet_distance"](
        *definitions["calculate_activation_statistics"](ref_z), mu_gen, cov_gen,
    )
    assert report["fid_test_diagnostic"] == pytest.approx(expected_test_fid, rel=1e-7)
    assert report["fid_train"] != report["fid_test_diagnostic"]
    assert report["multimodality"] is None and report["metadata"]["reference_count"] == 20
    json.dumps(report, allow_nan=False)


def test_metrics_reject_undefined_standardization_and_bad_accuracy():
    train = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.float32)
    with pytest.raises(ValueError, match="zero/nonfinite"):
        evaluate_native_features(np.ones((3, 2)), train)
    with pytest.raises(ValueError, match="dimensions differ"):
        evaluate_native_features(train, np.ones((3, 3)))
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_native_features(train, np.full((3, 2), np.nan))
    with pytest.raises(ValueError, match="both expected_labels"):
        evaluate_native_features(train, train, expected_labels=[0, 1, 2])
    with pytest.raises(ValueError, match="integer in"):
        evaluate_native_features(train, train, expected_labels=[0, 1, 6], generated_predictions=[0, 1, 2])


def test_prompt_multimodality_uses_within_prompt_distinct_samples():
    train = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.float32)
    # Group means are far apart but samples under each prompt are identical.
    generated = np.array([[0, 0], [0, 0], [100, 100], [100, 100]], dtype=np.float32)
    report = evaluate_native_features(
        train, generated, prompt_group_ids=["prompt_a", "prompt_a", "prompt_b", "prompt_b"],
        expected_labels=[0, 1, 2, 3], generated_predictions=[0, 0, 2, 2], diversity_pairs=20,
        multimodality_pairs_per_group=10,
    )
    assert report["multimodality"] == 0.0
    assert report["diversity_generated"] > 0
    assert report["accuracy"] == 0.5
    assert report["metadata"]["multimodality_groups"] == 2
    with pytest.raises(ValueError, match="at least two"):
        evaluate_native_features(train, generated, prompt_group_ids=["a", "a", "b", "c"], diversity_pairs=2)


def test_checkpoint_requires_plain_six_class_state_dict(tmp_path):
    path = tmp_path / "wrong.pth"
    torch.save({"state_dict": {"pred.0.weight": torch.zeros(6, 60)}}, path)
    with pytest.raises(ValueError, match="plain tensor"):
        load_classifier(path)
    torch.save({"pred.0.weight": torch.zeros(5, 60)}, path)
    with pytest.raises(ValueError, match="six-class"):
        load_classifier(path)


def test_frechet_known_gaussians():
    # Identical unit covariances leave just squared displacement of means.
    assert frechet_distance([0, 0], np.eye(2), [3, 4], np.eye(2)) == 25.0
    with pytest.raises(ValueError, match="incompatible shapes"):
        frechet_distance([0], np.eye(2), [0], np.eye(2))
