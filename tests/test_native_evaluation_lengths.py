"""Native baseline generation stays separate from the learned metric view."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from data.sim_format import to_simulation_format
from data.ccdm.dataset import CCDMDataset
from data.simulation.dataset import SimulationDataset
from testing.baseline_modes import BASELINE_ITEMS
from testing.process import test_batch as evaluate_batch
from testing.trajectory_cache import validate_cached_metric_batch


class ReferenceModel:
    decoder = SimpleNamespace(seq_length=30)
    memory_tokens_count = 2

    def generate_camera_trajectory(self, **kwargs):
        subject = kwargs["subject_trajectory"]
        assert subject.shape[1] == 30
        assert kwargs["padding_mask"].shape[1] == 30
        return {"reconstructed": torch.zeros_like(subject)}

    def embed_trajectory(self, camera, subject, volume, src_key_mask, subject_key_padding_mask=None):
        assert camera.shape[1] == subject.shape[1] == src_key_mask.shape[1] == 30
        return camera.new_ones(2, camera.shape[0], 4)


class NativeBaseline:
    def __init__(self, kind, length):
        self.kind, self.length = kind, length
        self.calls = 0

    def generate_using_text(self, prompts, subject, trajectory, mask):
        assert trajectory.shape[1] == mask.shape[1] == self.length
        self.calls += 1
        if self.kind == "ccdm":
            result = trajectory.new_zeros(len(prompts), self.length, 5)
            result[..., 2] = 5
            return result
        if self.kind == "et":
            from data.et.dataset import ETDataset
            return ETDataset.normalize_item(trajectory, subject, None, False)[0]
        return trajectory.clone()


class MetricExtractor:
    def encode_trajectory(self, camera, subject, volume, mask):
        assert camera.shape[1] == subject.shape[1] == mask.shape[1] == 30
        return camera.mean(dim=1)

    def encode_text(self, captions):
        return torch.ones(len(captions), 6)


class Metrics:
    def __init__(self):
        self.items = []

    def update_clatr_metrics(self, name, **features):
        assert all(torch.isfinite(value).all() for value in features.values())
        self.items.append(name)

    def update_clip_score(self, *args):
        pass


def batch_at_length(length):
    camera = torch.zeros(2, length, 6)
    camera[..., 2] = 5
    subject = torch.zeros_like(camera)
    camera[..., 0] = torch.linspace(0, 1, length)
    reference = {
        "camera_trajectory": camera[:, torch.linspace(0, length - 1, 30).round().long()].clone(),
        "subject_trajectory": subject[:, :30].clone(),
        "subject_volume": torch.ones(2, 1, 3),
        "padding_mask": torch.zeros(2, 30, dtype=torch.bool),
    }
    return {
        "camera_trajectory": camera, "subject_trajectory": subject,
        "subject_volume": torch.ones(2, 1, 3),
        "padding_mask": torch.zeros(2, length, dtype=torch.bool),
        "text_prompts": ["Static camera", "Move camera right"],
        "cinematography_prompt": torch.ones(2, 2, 4),
        "prompt_none_mask": torch.ones(2, 2, dtype=torch.bool),
        "simulation_reference": reference,
    }


class NativeEvaluationLengthTests(unittest.TestCase):
    def test_exact_reference_view_does_not_replace_native_input(self):
        batch = batch_at_length(300)
        batch["simulation_reference"]["camera_trajectory"] += 7
        native = to_simulation_format(batch, "simulation", target_len=300)
        reference = to_simulation_format(batch, "simulation", target_len=30)
        self.assertIs(native[0], batch["camera_trajectory"])
        torch.testing.assert_close(reference[0], batch["simulation_reference"]["camera_trajectory"])

    def test_native_generation_metrics_and_cache_replay(self):
        parameters = {
            key: {"mean": torch.zeros(3), "std": torch.ones(3)}
            for key in ("camera_position", "subject_position", "subject_dimensions")
        }
        with patch.object(SimulationDataset, "_normalization_parameters", parameters), patch.object(
            CCDMDataset, "_normalization_parameters", {"mean": torch.zeros(5), "std": torch.ones(5)}
        ):
            for kind, length in (("ccdm", 300), ("et", 300), ("gendop", 30)):
                with self.subTest(model=kind):
                    batch = batch_at_length(length)
                    model, metrics = NativeBaseline(kind, length), Metrics()
                    args = (ReferenceModel(), model, batch, metrics, torch.device("cpu"), list(BASELINE_ITEMS))
                    kwargs = dict(model_type=kind, seq_length=length, clatr_extractor=MetricExtractor())
                    cached = evaluate_batch(*args, **kwargs)
                    self.assertEqual(set(metrics.items), set(BASELINE_ITEMS))
                    for value in cached["trajectories"].values():
                        self.assertEqual(tuple(value.shape), (2, length, 6))
                    validate_cached_metric_batch(
                        cached, list(BASELINE_ITEMS), model_type=kind,
                        require_encoder_features=False, expected_token_count=2,
                        expected_embedding_dim=4,
                        expected_sequence_length=length,
                    )
                    calls = model.calls
                    evaluate_batch(*args, **kwargs, cached_outputs=cached)
                    self.assertEqual(model.calls, calls)


if __name__ == "__main__":
    unittest.main()
