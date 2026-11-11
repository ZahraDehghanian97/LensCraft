"""A real Lightning training batch crosses the categorical input/loss boundary."""
from functools import partial
from contextlib import ExitStack
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import lightning as L
import torch
from torch.utils.data import DataLoader

from data.categorical_conditioning import conditioning_from_batch
from models.camera_trajectory_model import LensCraft
from models.categorical_latent import FIELD_WIDTHS, valid_category_mask
from training.autoencoder_trainer import LightningLensCraft
from training.base_trainer import MaskConfig, NoiseConfig, TeacherForcingConfig
from training.losses.categorical_loss import CategoricalCameraTrajectoryLoss
from training.multi_dataset_trainer import MultiDatasetTrainer


class RecordingCategoricalLoss(CategoricalCameraTrajectoryLoss):
    """Observe the actual loss call without replacing any production operation."""

    def __init__(self):
        super().__init__(
            losses_list={"categorical": 1.0, "cycle": 0.3, "first_frame": 1.0,
                         "rotation_absolute": 0.1, "rotation_raw": 0.1},
        )
        self.training_calls = []
        self.validation_calls = []

    def __call__(self, output, camera, target, batch, *args, **kwargs):
        loss, terms = super().__call__(output, camera, target, batch, *args, **kwargs)
        record = {"output": output, "target": target.detach().clone(),
                  "presence": batch["prompt_none_mask"].detach().clone(),
                  "terms": {name: value.detach().clone() for name, value in terms.items()
                            if torch.is_tensor(value)}}
        if torch.is_grad_enabled():
            output["embedding_logits"].retain_grad()
            output["cycle_embeddings"].retain_grad()
            # In prompt mode, q comes from a caption-decoded trajectory. The
            # original probabilities can affect this KL only through its target;
            # no gradient here proves that the real cycle target was detached.
            record["cycle_target_gradient"] = torch.autograd.grad(
                terms["cycle"], output["embeddings"], allow_unused=True,
                retain_graph=True,
            )[0]
            self.training_calls.append(record)
        else:
            self.validation_calls.append(record)
        return loss, terms


class GradientAudit(L.Callback):
    def __init__(self):
        self.semantic_head_gradient = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        gradient = pl_module.model.encoder.semantic_projection.weight.grad
        self.semantic_head_gradient = None if gradient is None else gradient.detach().clone()


class TestCategoricalTraining(unittest.TestCase):
    def test_multidataset_ccdm_branch_keeps_categorical_and_cycle_supervision(self):
        torch.manual_seed(42)
        model = LensCraft(
            latent_dim=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, dropout_rate=0.0, seq_length=5,
        ).cpu()
        objective = RecordingCategoricalLoss()
        module = MultiDatasetTrainer(
            model=model, optimizer=partial(torch.optim.AdamW, lr=1e-3),
            lr_scheduler=None, loss_module=objective,
            noise=NoiseConfig(0.0, 0.0), mask=MaskConfig(0.0, 0.0, 0.0),
            teacher_forcing_schedule=TeacherForcingConfig(1.0, 1.0, 0.0, 0.0),
            compile_enabled=False,
        )
        # Only scheduling reads this controller; the real BaseTrainer forward,
        # camera model, cycle encoder, and categorical objective run below.
        module._trainer = SimpleNamespace(current_epoch=0, max_epochs=1)
        ids = torch.tensor([[width - 1 for width in FIELD_WIDTHS], [0] * 10])
        ids[0, 1] = -1
        ids[1, 3] = -1
        mask = ids >= 0
        generator = torch.Generator().manual_seed(7)
        batch = {
            "camera_trajectory": torch.randn(2, 5, 6, generator=generator) / 5,
            "subject_trajectory": torch.randn(2, 5, 6, generator=generator) / 5,
            "subject_volume": torch.rand(2, 1, 3, generator=generator) + 0.5,
            "padding_mask": torch.zeros(2, 5, dtype=torch.bool),
            "cinematography_ids": ids, "prompt_none_mask": mask,
        }
        before_mask = mask.clone()

        def already_simulation(source, target, camera, subject, volume, padding, **kwargs):
            self.assertEqual((source, target, kwargs["target_len"]), ("ccdm", "simulation", 5))
            return camera, subject, volume, padding

        # Coordinates are intentionally already SIM-shaped: this regression
        # exercises multi-dataset dispatch independently of native normalization.
        with patch("training.multi_dataset_trainer.convert_to_target", side_effect=already_simulation), \
                patch.object(module, "_forward_step", wraps=module._forward_step) as forward:
            loss, terms = module._process_ccdm_batch(batch, "train")
        self.assertIs(forward.call_args.kwargs["compute_cycle_embeddings"], True)
        self.assertTrue(torch.equal(forward.call_args.kwargs["caption_presence_mask"], ids >= 0))
        for name in ("categorical", "cycle"):
            self.assertIn(name, terms)
            self.assertTrue(torch.isfinite(terms[name]))
            self.assertGreater(float(terms[name]), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = model.encoder.semantic_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertEqual(len(objective.training_calls), 1)
        record = objective.training_calls[0]
        self.assertEqual(record["presence"].shape, (2, 10))
        self.assertEqual(record["output"]["cycle_embeddings"].shape, (10, 2, 21))
        self.assertIsNone(record["cycle_target_gradient"])
        self.assertTrue(torch.equal(batch["prompt_none_mask"], before_mask))

    def test_actual_lightning_step_trains_ce_and_detached_cycle_kl_without_clip(self):
        torch.manual_seed(42)
        model = LensCraft(
            latent_dim=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, dropout_rate=0.0, seq_length=5,
        ).cpu()
        generator = torch.Generator().manual_seed(7)
        ids = torch.tensor([[width - 1 for width in FIELD_WIDTHS], [0] * 10])
        ids[0, 1] = -1
        ids[1, 3] = -1
        ids[1, 4] = 19
        padding = torch.zeros((2, 5), dtype=torch.bool)
        padding[0, -1] = True
        batch = {
            "camera_trajectory": torch.randn(2, 5, 6, generator=generator) / 5,
            "subject_trajectory": torch.randn(2, 5, 6, generator=generator) / 5,
            "subject_volume": torch.rand(2, 1, 3, generator=generator) + 0.5,
            "padding_mask": padding,
            "cinematography_ids": ids,
            "prompt_none_mask": ids >= 0,
        }
        # This has the same compact input contract as the categorical dataset:
        # no CLIP tensors, free-text features, or parameter metadata are needed.
        expected_target = conditioning_from_batch(batch, model)
        before_ids, before_presence = ids.clone(), batch["prompt_none_mask"].clone()
        before_head = model.encoder.semantic_projection.weight.detach().clone()
        objective = RecordingCategoricalLoss()
        module = LightningLensCraft(
            model=model,
            optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=0.0),
            lr_scheduler=None,
            loss_module=objective,
            noise=NoiseConfig(0.0, 0.0),
            mask=MaskConfig(0.0, 0.0, 0.0),
            teacher_forcing_schedule=TeacherForcingConfig(1.0, 1.0, 0.0, 0.0),
            compile_enabled=False,
            use_cycle_consistency=True,
            conditioning={"enabled": True, "keyframe_counts": [1, 2]},
        )
        # Fix only the stochastic mode selection; _step and _forward_step still
        # execute their real prompt-conditioning, CE, geometry, and cycle paths.
        module.conditioning.sample = lambda: ("prompt_generation", None)
        audit = GradientAudit()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = L.Trainer(
                accelerator="cpu", devices=1, fast_dev_run=True,
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, num_sanity_val_steps=0,
                default_root_dir=temporary, callbacks=[audit], precision="32-true",
            )
            trainer.fit(module, train_dataloaders=DataLoader([batch], batch_size=None),
                        val_dataloaders=DataLoader([batch], batch_size=None))

        self.assertEqual(trainer.global_step, 1)
        self.assertEqual(len(objective.training_calls), 1)
        self.assertEqual(len(objective.validation_calls), 1)
        record = objective.training_calls[0]
        output = record["output"]
        self.assertEqual(output["conditioning_mode"], "prompt_generation")
        self.assertEqual(output["cycle_embeddings"].shape, (10, 2, 21))
        self.assertTrue(torch.equal(record["target"], expected_target))
        self.assertTrue(torch.equal(record["presence"], before_presence))
        self.assertTrue(torch.equal(output["cycle_prompt_mask"], before_presence))
        self.assertIsNone(record["cycle_target_gradient"])
        for observed in objective.training_calls + objective.validation_calls:
            for name in ("categorical", "cycle", "first_frame", "rotation_absolute", "rotation_raw"):
                self.assertIn(name, observed["terms"])
                self.assertTrue(torch.isfinite(observed["terms"][name]), name)
            self.assertGreater(float(observed["terms"]["categorical"]), 0)
            self.assertGreaterEqual(float(observed["terms"]["cycle"]), 0)
        logits_gradient = output["embedding_logits"].grad
        cycle_gradient = output["cycle_embeddings"].grad
        self.assertIsNotNone(logits_gradient)
        self.assertIsNotNone(cycle_gradient)
        self.assertTrue(torch.isfinite(logits_gradient).all())
        self.assertTrue(torch.isfinite(cycle_gradient).all())
        self.assertGreater(float(logits_gradient.abs().sum()), 0)
        self.assertGreater(float(cycle_gradient.abs().sum()), 0)
        valid = valid_category_mask()[:, None, :].expand_as(logits_gradient)
        unknown = ~before_presence.T[..., None].expand_as(logits_gradient)
        self.assertEqual(int(logits_gradient.masked_select(~valid | unknown).count_nonzero()), 0)
        self.assertEqual(int(cycle_gradient.masked_select(~valid | unknown).count_nonzero()), 0)
        self.assertIsNotNone(audit.semantic_head_gradient)
        self.assertTrue(torch.isfinite(audit.semantic_head_gradient).all())
        self.assertGreater(float(audit.semantic_head_gradient.abs().sum()), 0)
        self.assertFalse(torch.equal(before_head, model.encoder.semantic_projection.weight.detach()))
        self.assertTrue(torch.equal(ids, before_ids))
        self.assertTrue(torch.equal(batch["prompt_none_mask"], before_presence))


if __name__ == "__main__":
    unittest.main()
