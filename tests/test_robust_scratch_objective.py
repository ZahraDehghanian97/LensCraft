"""Check supervision isolation, semantic masks and actual decoder gradients.

The forward tests use the repository's BaseTrainer and CameraTrajectoryLoss.
Tiny model and loss fixtures isolate gradients without a trained checkpoint.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    from training.losses.losses import CameraTrajectoryLoss
    from training.robust_scratch import objective

    class TinyEncoder(torch.nn.Module):
        def __init__(self, fields=4, width=8):
            super().__init__()
            self.projection = torch.nn.Linear(6, width)
            self.slots = torch.nn.Parameter(torch.randn(fields, width))
            self.calls = []

        def forward(self, camera, subject, camera_mask=None,
                    subject_key_padding_mask=None):
            self.calls.append((camera.detach().clone(),
                               None if camera_mask is None else camera_mask.detach().clone()))
            valid = (torch.ones(camera.shape[:2], dtype=torch.bool, device=camera.device)
                     if camera_mask is None else ~camera_mask)
            mean = camera.masked_fill(~valid[..., None], 0).sum(1)
            mean = mean / valid.sum(1).clamp_min(1)[:, None]
            encoded = self.projection(mean) + subject.mean(1)
            return encoded[None] + self.slots[:, None]

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TinyEncoder()
            self.subject_projection = torch.nn.Linear(6, 8)
            self.volume_projection = torch.nn.Linear(3, 8)
            self.decoder = torch.nn.Linear(8, 6)
            self.frame_parameters = torch.nn.Parameter(torch.randn(4, 6) / 10)
            self.last_generator_camera = None

        def forward(self, src, subject_trajectory, subject_volume,
                    tgt_key_padding_mask=None, src_key_mask=None,
                    caption_embedding=None, memory_teacher_forcing_ratio=1.0,
                    **kwargs):
            self.last_generator_camera = src.detach().clone()
            subject = torch.cat((self.subject_projection(subject_trajectory),
                                 self.volume_projection(subject_volume)[:, None]), dim=1)
            encoded = self.encoder(src, subject, src_key_mask,
                                   subject_key_padding_mask=tgt_key_padding_mask)
            memory = caption_embedding if memory_teacher_forcing_ratio == 1 else encoded
            generated = self.decoder(memory.mean(0) + subject.mean(1))[:, None]
            generated = generated + self.frame_parameters[None]
            identity = torch.eye(3).reshape(1, 1, 9).expand(*generated.shape[:2], 9)
            matrix = torch.cat((generated[..., :3], identity), dim=-1)
            return {'subject_embedding': subject, 'embeddings': encoded,
                    'reconstructed': generated, 'reconstructed_raw_matrix': matrix,
                    'reconstructed_rot_matrix': matrix}

    class MiniLoss:
        def __init__(self):
            self.losses_list = {'clip': 3.0, 'first_frame': 2.0, 'cycle': 100.0}
            self.weighted_clip_loss = False
            self.encoder_loss_function = 'clip'
            self.clip_embeddings = None

        def __call__(self, output, target, caption, batch, padding=None, **kwargs):
            geometry = (output['reconstructed_raw_matrix'][:, 0, :3] - target[:, 0, :3]).square().mean()
            clip = (output['embeddings'] - caption).square().mean()
            cycle = geometry.new_zeros(())
            if output.get('cycle_embeddings') is not None:
                cycle = (output['cycle_embeddings'] - output['embeddings']).square().mean()
            result = {'clip': clip, 'first_frame': geometry, 'cycle': cycle}
            total = sum(self.losses_list[key] * value for key, value in result.items())
            result['total'] = total.detach().item()
            return total, result



@unittest.skipIf(torch is None, 'Meaningful objective checks require the training dependencies')
class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(817)
        self.camera = torch.randn(2, 4, 6)
        self.subject = torch.randn(2, 4, 6)
        self.volume = torch.randn(2, 3)
        self.caption = torch.randn(4, 2, 8)
        self.padding = torch.tensor([[False, False, False, True], [False] * 4])
        self.presence = torch.ones(2, 4, dtype=torch.bool)

    def make_stepper(self):
        model = TinyModel()
        stepper = objective.RobustForward(
            model, torch.optim.AdamW(model.parameters()), None, MiniLoss(),
            noise=SimpleNamespace(initial_std=0, final_std=0),
            mask=SimpleNamespace(initial_ratio=0, final_ratio=0, memory_ratio=0),
            teacher_forcing_schedule=SimpleNamespace(
                memory_initial_ratio=1, memory_final_ratio=1,
                trajectory_initial_ratio=0, trajectory_final_ratio=0),
            compile_enabled=False, use_merged_memory=False,
        )
        return stepper

    def forward(self, stepper, *, mode='prompt_generation', camera=None,
                training=True, cycle=True, presence=None):
        return stepper._forward_step(
            self.camera if camera is None else camera,
            self.subject, self.volume, self.caption, self.padding,
            is_training=training, compute_cycle_embeddings=cycle,
            conditioning_mode=mode,
            keyframe_count=1 if mode.startswith('key_framing') else None,
            keyframe_sample_seeds=[91, 92],
            caption_presence_mask=self.presence if presence is None else presence,
        )

    def test_prompt_target_camera_cannot_change_generated_output(self):
        stepper = self.make_stepper()
        for training in (True, False):
            first = self.forward(stepper, training=training)
            second = self.forward(stepper, camera=self.camera * 7 + 13, training=training)
            torch.testing.assert_close(first['reconstructed'], second['reconstructed'], rtol=0, atol=0)
            self.assertEqual(int(torch.count_nonzero(stepper.model.last_generator_camera)), 0)
            self.assertFalse(first['known_mask'].any())
            if training:
                self.assertFalse(torch.equal(first['embeddings'], second['embeddings']))

    def test_full_camera_supervision_runs_only_after_generation(self):
        stepper = self.make_stepper()
        out = self.forward(stepper, mode='key_framing')
        calls = stepper.model.encoder.calls
        self.assertEqual(len(calls), 3)
        self.assertGreater(int(calls[0][1].sum()), int(self.padding.sum()))
        clean, clean_mask = calls[1]
        torch.testing.assert_close(clean, self.camera.masked_fill(self.padding[..., None], 0))
        torch.testing.assert_close(clean_mask, self.padding)
        generated, generated_mask = calls[2]
        torch.testing.assert_close(generated, out['reconstructed'].detach().masked_fill(self.padding[..., None], 0))
        torch.testing.assert_close(generated_mask, self.padding)

    def test_each_generated_valid_frame_gets_cycle_gradient(self):
        stepper = self.make_stepper()
        out = self.forward(stepper, mode='key_framing+prompt')
        out['reconstructed'].retain_grad()
        value, _ = objective.anchored_cycle_loss(
            out['cycle_embeddings'], out['anchored_cycle_target'], out['anchored_cycle_mask'])
        value.backward()
        norms = out['reconstructed'].grad.norm(dim=-1)
        self.assertTrue((norms[~self.padding] > 0).all())
        self.assertTrue((norms[self.padding] == 0).all())
        self.assertGreater(float(stepper.model.decoder.weight.grad.norm()), 0)
        self.assertTrue(torch.isfinite(stepper.model.decoder.weight.grad).all())

    def test_caption_cycle_target_is_supplied_caption_not_clean_encoder(self):
        stepper = self.make_stepper()
        out = self.forward(stepper)
        torch.testing.assert_close(out['anchored_cycle_target'], self.caption, rtol=0, atol=0)
        self.assertFalse(out['anchored_cycle_target'].requires_grad)
        self.assertEqual(out['cycle_target_kind'], 'supplied_caption')
        self.assertFalse(torch.equal(out['anchored_cycle_target'], out['embeddings']))

    def test_camera_cycle_target_is_detached_clean_encoder(self):
        stepper = self.make_stepper()
        for mode in ('reconstruction', 'key_framing'):
            out = self.forward(stepper, mode=mode)
            torch.testing.assert_close(out['anchored_cycle_target'], out['embeddings'])
            self.assertFalse(out['anchored_cycle_target'].requires_grad)
            self.assertIsNone(out['anchored_cycle_mask'])
            self.assertEqual(out['cycle_target_kind'], 'detached_clean_camera')

    def test_effective_decoder_missingness_mask_controls_cycle(self):
        stepper = self.make_stepper()
        effective = self.presence.clone()
        effective[:, 2:] = False

        class DropFields:
            def apply(self, caption, original, **kwargs):
                result = caption.clone()
                result[2:] = 0
                return result, effective

        stepper.caption_missingness = DropFields()
        out = self.forward(stepper)
        torch.testing.assert_close(out['anchored_cycle_mask'], effective)
        torch.testing.assert_close(out['anchored_cycle_target'], self.caption)

    def test_unknown_fields_and_detached_targets_have_zero_gradient(self):
        prediction = torch.randn(4, 2, 8, requires_grad=True)
        target = torch.randn(4, 2, 8, requires_grad=True)
        mask = self.presence.clone()
        mask[:, 2:] = False
        value, _ = objective.anchored_cycle_loss(prediction, target, mask)
        value.backward()
        self.assertEqual(int(torch.count_nonzero(prediction.grad[2:])), 0)
        self.assertGreater(float(prediction.grad[:2].norm()), 0)
        self.assertIsNone(target.grad)
        changed = target.detach().clone()
        changed[2:] = float('nan')
        other, _ = objective.anchored_cycle_loss(prediction.detach(), changed, mask)
        torch.testing.assert_close(value.detach(), other)

    def test_all_unknown_nonfinite_fills_have_finite_zero_loss_and_gradient(self):
        prediction = torch.full((4, 2, 8), float('nan'), requires_grad=True)
        target = torch.full_like(prediction, float('inf'))
        value, _ = objective.anchored_cycle_loss(prediction, target, torch.zeros_like(self.presence))
        self.assertEqual(float(value.detach()), 0)
        value.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(int(torch.count_nonzero(prediction.grad)), 0)

    def test_masked_field_reduction_and_float32_autocast_are_explicit(self):
        prediction = torch.ones(4, 2, 8, dtype=torch.bfloat16, requires_grad=True)
        target = torch.zeros_like(prediction)
        mask = torch.zeros_like(self.presence)
        mask[0, 0] = True
        with torch.autocast('cpu', dtype=torch.bfloat16):
            value, fields = objective.anchored_cycle_loss(
                prediction, target, mask, encoder_loss_function='mse')
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(float(value.detach()), 0.25)
        self.assertEqual([float(x.detach()) for x in fields], [1, 0, 0, 0])

    def test_known_nonfinite_and_malformed_masks_fail(self):
        prediction = torch.randn(4, 2, 8)
        prediction[0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            objective.anchored_cycle_loss(prediction, self.caption, self.presence)
        for mask in (self.presence.float(), torch.ones(4, 2, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                objective.anchored_cycle_loss(self.caption, self.caption, mask)

    def test_base_geometry_weights_and_total_are_preserved_without_double_count(self):
        stepper = self.make_stepper()
        out = self.forward(stepper)
        base = CameraTrajectoryLoss(losses_list={
            'clip': 3.0, 'first_frame': 2.0, 'relative': 16.0, 'speed': 80.0,
            'rotation_absolute': 2.0, 'rotation_raw': 1.0, 'cycle': 100.0,
        })
        weights = dict(base.losses_list)
        wrapped = objective.RobustObjective(base)
        batch = {'prompt_none_mask': self.presence}
        original_output = dict(out)
        original_output.pop('cycle_embeddings')
        before, terms = base(original_output, self.camera, self.caption, batch, self.padding)
        after, robust_terms = wrapped(out, self.camera, self.caption, batch, self.padding)
        cycle, _ = objective.anchored_cycle_loss(
            out['cycle_embeddings'], out['anchored_cycle_target'], out['anchored_cycle_mask'])
        torch.testing.assert_close(after, before + 100 * cycle)
        for key in weights:
            if key != 'cycle' and key in terms:
                torch.testing.assert_close(robust_terms[key], terms[key])
        self.assertEqual(robust_terms['total'], float(after.detach()))
        self.assertEqual(base.losses_list, weights)
        self.assertEqual(wrapped.losses_list, weights)
        self.assertIn('cycle_embeddings', out)
        self.assertEqual(float(terms['cycle'].detach()), 0)

    def test_missing_cycle_is_zero_but_unanchored_cycle_fails(self):
        stepper = self.make_stepper()
        out = self.forward(stepper, cycle=False)
        wrapped = objective.RobustObjective(MiniLoss())
        _, terms = wrapped(out, self.camera, self.caption, {}, self.padding)
        self.assertEqual(float(terms['cycle']), 0)
        out['cycle_embeddings'] = out['embeddings']
        with self.assertRaisesRegex(ValueError, 'anchored target'):
            wrapped(out, self.camera, self.caption, {}, self.padding)

    def test_position_unit_changes_preserve_semantic_anchor(self):
        stepper = self.make_stepper()
        out = self.forward(stepper)
        wrapped = objective.RobustObjective(MiniLoss())
        original, _ = wrapped(out, self.camera, self.caption, {}, self.padding)
        mean = torch.tensor([2., -3., 4.])
        std = torch.tensor([3., 5., 7.])
        diagonal = torch.tensor([[1., 2., 3.], [3., 2., 1.]]).norm(dim=-1)[:, None, None]

        # A geometry-only adapter may change position units without changing
        # the generated semantic tensors or the supplied-caption cycle target.
        transformed = dict(out)
        for name in ('reconstructed_raw_matrix', 'reconstructed_rot_matrix'):
            value = out[name]
            transformed[name] = torch.cat(((value[..., :3] * std + mean) / diagonal,
                                           value[..., 3:]), dim=-1)
        target = torch.cat(((self.camera[..., :3] * std + mean) / diagonal,
                            self.camera[..., 3:]), dim=-1)
        changed, terms = wrapped(transformed, target, self.caption, {}, self.padding)
        expected, _ = objective.anchored_cycle_loss(
            out['cycle_embeddings'], out['anchored_cycle_target'], out['anchored_cycle_mask'])
        torch.testing.assert_close(terms['cycle'], expected)
        torch.testing.assert_close(out['anchored_cycle_target'], self.caption)
        self.assertIs(transformed['cycle_embeddings'], out['cycle_embeddings'])
        self.assertFalse(torch.isclose(original, changed))

    def test_clean_supervision_not_computed_in_validation(self):
        stepper = self.make_stepper()
        self.forward(stepper, training=False, cycle=False)
        self.assertEqual(len(stepper.model.encoder.calls), 1)


if __name__ == '__main__':
    unittest.main()
