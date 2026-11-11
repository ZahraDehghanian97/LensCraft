"""Categorical labels and cycle distributions never supervise padding/missing fields."""
import math
import unittest
from unittest.mock import patch

import torch

from models.categorical_latent import FIELD_NAMES, from_ids, masked_category_logits, masked_softmax
from training.losses.categorical_loss import CategoricalCameraTrajectoryLoss
from training.losses.losses import CameraTrajectoryLoss


def output_from_logits(logits, cycle=None):
    batch = logits.shape[1]
    matrix = torch.zeros((batch, 3, 12), dtype=logits.dtype)
    matrix[..., 3:] = torch.eye(3, dtype=logits.dtype).reshape(9)
    result = {
        "embeddings": masked_softmax(logits),
        "embedding_logits": masked_category_logits(logits),
        "reconstructed_raw_matrix": matrix,
        "reconstructed_rot_matrix": matrix,
    }
    if cycle is not None:
        result["cycle_embeddings"] = cycle
    return result, torch.zeros((batch, 3, 6), dtype=logits.dtype)


class TestCategoricalLoss(unittest.TestCase):
    def test_ce_native20_gives_right_class_gradient_and_ignores_padding(self):
        logits = torch.zeros((10, 1, 21), dtype=torch.float64, requires_grad=True)
        ids = torch.full((1, 10), -1, dtype=torch.long)
        ids[0, 4] = 20
        target, presence = from_ids(ids, dtype=torch.float64)
        output, trajectory = output_from_logits(logits)
        module = CategoricalCameraTrajectoryLoss(losses_list={"cycle": 0})
        loss, terms = module(output, trajectory, target, {"prompt_none_mask": presence})

        self.assertAlmostEqual(float(loss), math.log(21), places=12)
        self.assertIn(FIELD_NAMES[4], terms["categorical_elements"])
        loss.backward()
        self.assertLess(float(logits.grad[4, 0, 20]), 0)
        self.assertTrue((logits.grad[4, 0, :20] > 0).all())
        self.assertEqual(int(torch.count_nonzero(logits.grad[:4])), 0)
        self.assertEqual(int(torch.count_nonzero(logits.grad[5:])), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ce_macroaverages_active_fields_not_all_labeled_samples(self):
        logits = torch.zeros((10, 2, 21), dtype=torch.float64, requires_grad=True)
        ids = torch.full((2, 10), -1, dtype=torch.long)
        ids[:, 4] = 20
        ids[:, 0] = 0
        target, presence = from_ids(ids, dtype=torch.float64)
        presence[1, 0] = False
        mask46 = torch.cat([presence, torch.ones((2, 36), dtype=torch.bool)], dim=1)
        output, trajectory = output_from_logits(logits)
        loss, _ = CategoricalCameraTrajectoryLoss(losses_list={"cycle": 0})(
            output, trajectory, target, {"prompt_none_mask": mask46},
        )
        self.assertAlmostEqual(float(loss), (math.log(5) + math.log(21)) / 2, places=12)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad[0, 1])), 0)

    def test_probability_fallback_matches_logits_ce_with_exact_zero_padding(self):
        logits = torch.zeros((10, 1, 21), dtype=torch.float64, requires_grad=True)
        ids = torch.zeros((1, 10), dtype=torch.long)
        ids[0, 4] = 20
        target, presence = from_ids(ids, dtype=torch.float64)
        output, trajectory = output_from_logits(logits)
        module = CategoricalCameraTrajectoryLoss(losses_list={"cycle": 0})
        expected, _ = module(output, trajectory, target, {"prompt_none_mask": presence})
        output.pop("embedding_logits")
        actual, _ = module(output, trajectory, target, {"prompt_none_mask": presence})
        self.assertTrue(torch.isfinite(actual))
        self.assertAlmostEqual(float(actual), float(expected), places=12)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_missing_labels_are_excluded_even_if_prompt_mask_is_true(self):
        logits = torch.zeros((10, 1, 21), dtype=torch.float64)
        logits[2, 0, 7] = 1000
        logits.requires_grad_()
        ids = torch.full((1, 10), -1, dtype=torch.long)
        ids[0, 0] = 1
        target, _ = from_ids(ids, dtype=torch.float64)
        output, trajectory = output_from_logits(logits)
        loss, _ = CategoricalCameraTrajectoryLoss(losses_list={"cycle": 0})(
            output, trajectory, target, {"prompt_none_mask": torch.ones((1, 10), dtype=torch.bool)},
        )
        self.assertAlmostEqual(float(loss), math.log(5), places=12)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad[2])), 0)

    def test_cycle_is_zero_identical_positive_perturbed_and_always_detached(self):
        target, presence = from_ids(torch.zeros((1, 10), dtype=torch.long), dtype=torch.float64)
        original_logits = torch.zeros((10, 1, 21), dtype=torch.float64, requires_grad=True)
        cycle_logits = torch.zeros((10, 1, 21), dtype=torch.float64)
        cycle_logits[:, :, 0] = 1
        cycle_logits.requires_grad_()
        output, trajectory = output_from_logits(original_logits)
        module = CategoricalCameraTrajectoryLoss(losses_list={"categorical": 0})
        output["cycle_embeddings"] = output["embeddings"]
        identical, _ = module(output, trajectory, target, {"prompt_none_mask": presence})
        self.assertAlmostEqual(float(identical), 0, places=12)
        output["cycle_embeddings"] = masked_softmax(cycle_logits)
        loss, _ = module(output, trajectory, target, {"prompt_none_mask": presence})
        self.assertGreater(float(loss), 0)
        loss.backward()
        self.assertGreater(float(cycle_logits.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(cycle_logits.grad).all())
        self.assertIsNone(original_logits.grad)

    def test_cycle_one_hot_zeros_and_unavailable_cycle_fields_have_finite_backward(self):
        original, presence = from_ids(torch.zeros((1, 10), dtype=torch.long), dtype=torch.float64)
        original.requires_grad_()
        cycle = original.detach().clone().requires_grad_()
        output, trajectory = output_from_logits(torch.zeros((10, 1, 21), dtype=torch.float64))
        output["embeddings"], output["cycle_embeddings"] = original, cycle
        output["cycle_prompt_mask"] = torch.zeros((1, 10), dtype=torch.bool)
        module = CategoricalCameraTrajectoryLoss(losses_list={"categorical": 0})
        masked, _ = module(output, trajectory, original.detach(), {"prompt_none_mask": presence})
        self.assertEqual(float(masked), 0)
        masked.backward()
        self.assertEqual(int(torch.count_nonzero(cycle.grad)), 0)
        cycle.grad = None
        output.pop("cycle_prompt_mask")
        loss, _ = module(output, trajectory, original.detach(), {"prompt_none_mask": presence})
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(cycle.grad).all())
        self.assertIsNone(original.grad)

    def test_no_annotations_produce_connected_exact_zero_and_zero_gradients(self):
        original_logits = torch.zeros((10, 2, 21), dtype=torch.float64, requires_grad=True)
        cycle_logits = torch.ones((10, 2, 21), dtype=torch.float64, requires_grad=True)
        target, _ = from_ids(torch.full((2, 10), -1, dtype=torch.long), dtype=torch.float64)
        output, trajectory = output_from_logits(original_logits, masked_softmax(cycle_logits))
        loss, terms = CategoricalCameraTrajectoryLoss()(output, trajectory, target, {})
        self.assertEqual(float(loss), 0)
        self.assertEqual(float(terms["categorical"]), 0)
        self.assertEqual(float(terms["cycle"]), 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(original_logits.grad)), 0)
        self.assertEqual(int(torch.count_nonzero(cycle_logits.grad)), 0)

    def test_geometry_values_gradients_padding_and_known_masks_match_parent(self):
        weights = {
            "first_frame": 2, "relative": 16, "speed": 80,
            "rotation_absolute": 2, "rotation_raw": 1,
            "position_absolute": 3, "keyframe_position": 4, "keyframe_rotation": 2,
        }
        target = torch.zeros((1, 4, 12), dtype=torch.float64)
        target[..., 3:] = torch.eye(3, dtype=torch.float64).reshape(9)
        pred = target.clone()
        pred[0, :3, 0] = torch.tensor([1., 2., 4.])
        pred[:, 3:] = float("nan")
        target[:, 3:] = float("nan")
        padding = torch.tensor([[False, False, False, True]])
        known = torch.tensor([[True, False, True, False]])
        expected_pred = pred.clone().requires_grad_()
        actual_pred = pred.clone().requires_grad_()
        expected_terms = CameraTrajectoryLoss(losses_list=weights).compute_trajectory_losses(
            expected_pred, target, tgt_key_padding_mask=padding,
            projected_pred=expected_pred, known_mask=known,
        )
        expected = sum(weights[name] * value for name, value in expected_terms.items())
        expected_terms['total'] = expected.detach().item()
        actual, actual_terms = CategoricalCameraTrajectoryLoss(
            losses_list={**weights, "categorical": 0, "cycle": 0},
        ).compute_total_loss(
            actual_pred, target, None, None, {}, tgt_key_padding_mask=padding,
            projected_pred=actual_pred, known_mask=known,
        )
        self.assertTrue(torch.equal(actual, expected))
        for key in expected_terms:
            self.assertEqual(float(actual_terms[key]), float(expected_terms[key]))
        expected.backward()
        actual.backward()
        self.assertTrue(torch.equal(actual_pred.grad, expected_pred.grad))
        self.assertTrue(torch.isfinite(actual_pred.grad).all())

    def test_constructor_exposes_only_categorical_and_geometry_objectives(self):
        import inspect
        self.assertEqual(set(inspect.signature(CategoricalCameraTrajectoryLoss).parameters),
                         {'losses_list', 'rotation_weight', 'probability_epsilon'})
        self.assertTrue(CategoricalCameraTrajectoryLoss().cycle_target_detach)
        for key in ("clip", "contrastive", "cosine"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                CategoricalCameraTrajectoryLoss(losses_list={key: 1})


if __name__ == "__main__":
    unittest.main()
