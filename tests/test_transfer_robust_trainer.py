"""Regression coverage for transfer training and held-out proxy selection."""

import pytest
import torch

from data.simulation.constants import NumericFeature
from data.simulation.dataset import SimulationDataset
from data.simulation.utils import CLIP_PARAMETERS, cinematography_struct_size
from training.autoencoder_trainer import LightningLensCraft
from training.base_trainer import BaseTrainer, MaskConfig, NoiseConfig, TeacherForcingConfig
from training.caption_missingness import CaptionMissingness
from training.conditioning import TRAINING_MODES, prepare_conditioning
from training.losses.losses import CameraTrajectoryLoss
from training.module_factory import checkpoint_monitor
from training.transfer_robust_trainer import TransferRobustTrainer


def vocabulary(width=8):
    result = {}
    for _, kind in CLIP_PARAMETERS:
        if isinstance(kind, NumericFeature):
            continue
        name = 'boolean' if kind is bool else kind.__name__
        result[name] = {'first': torch.ones(width), 'second': -torch.ones(width)}
    return result


@pytest.fixture
def trainer(monkeypatch):
    model = torch.nn.Linear(6, 6)
    model.caption_presence_conditioning = True
    module = TransferRobustTrainer(
        model=model, optimizer=None, lr_scheduler=None,
        loss_module=CameraTrajectoryLoss(
            losses_list={'clip': 1., 'cycle': 1.},
            encoder_loss_function='mse', clip_embeddings=vocabulary(),
        ),
        noise=NoiseConfig(0., 0.), mask=MaskConfig(0., 0.),
        teacher_forcing_schedule=TeacherForcingConfig(1., 1., 0., 0.),
        compile_enabled=False,
        conditioning={'keyframe_counts': [2]},
        validation_conditioning={'keyframe_counts': [2], 'max_batches': 2},
        caption_missingness={'probability': 1., 'fill': 'mean'},
        validation_caption_missingness={'max_batches': 2, 'score_weight': 3.},
        domain_augmentation={'enabled': True, 'et_probability': .25, 'ccdm_probability': .25},
        validation_domain_augmentation={'max_batches': 2, 'keyframe_count': 2, 'score_weight': 2.},
    )
    module.recorded_metrics = {}

    def record(self, name, value, *args, **kwargs):
        self.recorded_metrics[name] = torch.as_tensor(value).detach().clone()

    monkeypatch.setattr(BaseTrainer, 'log', record)
    monkeypatch.setattr(BaseTrainer, 'on_validation_epoch_end', lambda self: None)
    monkeypatch.setattr(SimulationDataset, 'normalize_item', staticmethod(lambda c, s, v, n: (c, s, v)))
    return module


@pytest.fixture
def batch():
    size, frames, width = 2, 5, 8
    camera = torch.zeros(size, frames, 6)
    camera[..., 0] = torch.arange(frames).float() / 10
    camera[..., 2] = 5.
    subject = torch.zeros_like(camera)
    subject[..., 0] = torch.arange(frames).float() / 20
    subject[..., 4] = torch.arange(frames).float() / 10
    caption = torch.arange(len(CLIP_PARAMETERS) * size * width, dtype=torch.float32)
    caption = caption.reshape(len(CLIP_PARAMETERS), size, width) / 100
    return {
        'camera_trajectory': camera, 'subject_trajectory': subject,
        'subject_volume': torch.ones(size, 1, 3),
        'padding_mask': torch.zeros(size, frames, dtype=torch.bool),
        'cinematography_prompt': caption[:cinematography_struct_size],
        'simulation_instruction': caption[cinematography_struct_size:],
        'prompt_none_mask': torch.ones(size, len(CLIP_PARAMETERS), dtype=torch.bool),
    }


def test_stochastic_augmentation_is_training_only(trainer, batch, monkeypatch):
    train_view = dict(batch, domain_augmentation_view=torch.tensor([1, 2]))
    augmentation_calls, forwarded, validated = [], [], []

    def augment(value, **kwargs):
        augmentation_calls.append((value, kwargs))
        return train_view

    def step(self, value, index, stage):
        forwarded.append((value, index, stage))
        return torch.tensor(1.)

    monkeypatch.setattr(trainer.domain_augmentation, 'apply', augment)
    monkeypatch.setattr(LightningLensCraft, '_step', step)
    monkeypatch.setattr(trainer, '_validate_domain_batch', lambda value, index: validated.append((value, index)))
    for stage in ('train', 'val', 'test'):
        trainer._step(batch, 3, stage)
    assert len(augmentation_calls) == 1
    assert augmentation_calls[0][0] is batch and augmentation_calls[0][1] == {}
    assert forwarded[0][0] is train_view
    assert forwarded[1][0] is batch and forwarded[2][0] is batch
    assert len(validated) == 1 and validated[0][0] is batch and validated[0][1] == 3
    assert trainer.recorded_metrics['train_domain_fraction/et'].item() == .5
    assert trainer.recorded_metrics['train_domain_fraction/ccdm'].item() == .5
    assert trainer.recorded_metrics['train_ccdm_motion_present_fraction'].item() == 1.
    assert trainer.recorded_metrics['train_ccdm_motion_sample_count'].item() == 1.
    assert trainer.recorded_metrics['train_ccdm_motion_present_count'].item() == 1.


def test_proxy_validation_repeats_without_rng_changes_and_covers_all_modes(trainer, batch, monkeypatch):
    calls, seen_views = [], []
    original_apply = trainer.domain_augmentation.apply

    def apply(value, **kwargs):
        view = original_apply(value, **kwargs)
        seen_views.append((kwargs['force_domain'], tuple(kwargs['sample_seeds']), view))
        return view

    def forward(camera, subject, volume, caption, padding, **kwargs):
        mode = kwargs['conditioning_mode']
        _, _, known, memory_ratio = prepare_conditioning(
            camera, padding, mode, kwargs['keyframe_count'], kwargs['keyframe_sample_seeds'],
        )
        mask = kwargs['caption_presence_mask']
        if memory_ratio > 0:
            caption, mask = trainer.caption_missingness.apply(
                caption, mask, sample_seeds=kwargs['caption_missingness_seeds'],
                force=kwargs['force_caption_missingness'],
            )
        calls.append((mode, known.clone(), mask.clone(), kwargs))
        generated = camera.clone()
        generated[..., 0] += mask.sum(dim=1).to(camera.dtype)[:, None] / 100
        return {'reconstructed': generated, 'known_mask': known}

    monkeypatch.setattr(trainer.domain_augmentation, 'apply', apply)
    monkeypatch.setattr(trainer, '_forward_step', forward)
    state_before = torch.get_rng_state().clone()
    results = []
    for _ in range(2):
        trainer.on_validation_epoch_start()
        trainer._validate_domain_batch(batch, 0)
        trainer._validate_domain_batch(batch, 1)
        trainer._validate_domain_batch(batch, 2)  # The configured limit is strict.
        assert trainer._domain_sample_offset == 4
        results.append({case: {key: value.clone() for key, value in metrics.items()}
                        for case, metrics in trainer._domain_totals.items()})
    assert torch.equal(torch.get_rng_state(), state_before)
    expected_modes = list(TRAINING_MODES) * 8
    assert [mode for mode, *_ in calls] == expected_modes
    assert [domain for domain, *_ in seen_views] == ['et', 'ccdm'] * 4
    first_seeds = tuple(range(27182, 27184))
    second_seeds = tuple(range(27184, 27186))
    assert [seeds for _, seeds, _ in seen_views] == [first_seeds, first_seeds, second_seeds, second_seeds] * 2
    for case in results[0]:
        for metric in results[0][case]:
            torch.testing.assert_close(results[0][case][metric], results[1][case][metric])
    for _, _, _, kwargs in calls:
        assert kwargs['is_training'] is False
        assert kwargs['force_caption_missingness'] is True
    assert torch.equal(seen_views[0][2]['camera_trajectory'], batch['camera_trajectory'])
    assert not torch.equal(seen_views[1][2]['camera_trajectory'], batch['camera_trajectory'])


def fill_metrics(metrics, position, *, known_position=0., rotation=0.):
    for name, pair in metrics.items():
        value = (position if name == 'position_error_normalized' else
                 known_position if name == 'known_position_error_normalized' else
                 rotation if name == 'rotation_error_deg' else 0.)
        pair.copy_(pair.new_tensor([value * 5, 5.]))


def test_selection_aggregates_clean_sparse_and_both_domains_all_families(trainer):
    trainer.on_validation_epoch_start()
    # All clean families have score 2. Sparse captions score 10, weighted 3:
    # the inherited clean/sparse score must therefore be (2 + 3*10)/4 = 8.
    for metrics in trainer._conditioning_totals.values():
        fill_metrics(metrics, 2.)
    fill_metrics(trainer._caption_missingness_totals, 10.)
    trainer._conditioning_sample_offset = 2
    trainer._domain_sample_offset = 2
    expected_domain = {}
    for domain, offset in (('et', 0.), ('ccdm', 4.)):
        scores = []
        for index, mode in enumerate(TRAINING_MODES):
            position = offset + index + 1
            fill_metrics(trainer._domain_totals[domain, mode], position, known_position=2.)
            scores.append(position + (2. if mode.startswith('key_framing') else 0.))
        expected_domain[domain] = sum(scores) / 4
    trainer.on_validation_epoch_end()
    metrics = trainer.recorded_metrics
    assert metrics['val_robustness_score'].item() == pytest.approx(8.)
    for domain in ('et', 'ccdm'):
        assert metrics[f'val_domain/{domain}/score'].item() == pytest.approx(expected_domain[domain])
    expected = (8. + 2. * sum(expected_domain.values())) / 5.
    assert metrics['val_transfer_score'].item() == pytest.approx(expected)
    # Prompt-only selection is independently equally weighted even when the
    # legacy sparse/domain weights are 3 and 2 respectively.
    assert metrics['val_transfer_prompt_score'].item() == pytest.approx((2. + 10. + 1. + 5.) / 4.)
    assert metrics['val_ccdm_prompt_score'].item() == pytest.approx(5.)
    assert metrics['val_domain_sample_count'].item() == 2


def test_ccdm_prompt_monitor_excludes_other_domains_and_reference_conditioned_modes(trainer):
    results = []
    for other_error in (1., 100.):
        trainer.on_validation_epoch_start()
        for metrics in trainer._conditioning_totals.values():
            fill_metrics(metrics, other_error)
        fill_metrics(trainer._caption_missingness_totals, other_error)
        for (domain, mode), metrics in trainer._domain_totals.items():
            position = 8. if (domain, mode) == ('ccdm', 'prompt_generation') else other_error
            fill_metrics(metrics, position, known_position=other_error)
        trainer.on_validation_epoch_end()
        results.append({key: value.item() for key, value in trainer.recorded_metrics.items()})
    assert all(metrics['val_ccdm_prompt_score'] == pytest.approx(8.) for metrics in results)
    assert all(metrics['val_ccdm_prompt_score'] ==
               metrics['val_transfer_prompt/ccdm_proxy_score'] for metrics in results)
    assert results[1]['val_transfer_prompt_score'] > results[0]['val_transfer_prompt_score']
    assert results[1]['val_transfer_score'] > results[0]['val_transfer_score']


def test_prompt_selection_uses_motion_but_not_reference_conditioned_scores(trainer):
    trainer.validation_conditioning.motion_score_weight = 1.
    results = []
    for reference_error in (1., 100.):
        trainer.on_validation_epoch_start()
        for name, metrics in trainer._conditioning_totals.items():
            fill_metrics(metrics, 2. if name == 'prompt_generation' else reference_error)
        fill_metrics(trainer._caption_missingness_totals, 4.)
        for (domain, mode), metrics in trainer._domain_totals.items():
            position = (6. if domain == 'et' else 8.) if mode == 'prompt_generation' else reference_error
            fill_metrics(metrics, position, known_position=reference_error)
        # All four prompt views receive absolute orientation error .1 and
        # temporal error .4 + (9+18+27)/(3*180) = .5.
        prompt_totals = [trainer._conditioning_totals['prompt_generation'],
                         trainer._caption_missingness_totals,
                         trainer._domain_totals['et', 'prompt_generation'],
                         trainer._domain_totals['ccdm', 'prompt_generation']]
        for metrics in prompt_totals:
            for key, value in {'rotation_error_deg': 18., 'step_error_normalized': .4,
                               'relative_rotation_error_deg': 9., 'rotation_step_error_deg': 18.,
                               'rotation_path_error_deg': 27.}.items():
                metrics[key].copy_(metrics[key].new_tensor([value * 5, 5.]))
        trainer.on_validation_epoch_end()
        results.append({key: value.item() for key, value in trainer.recorded_metrics.items()})
    for metrics in results:
        assert metrics['val_transfer_prompt_score'] == pytest.approx(5.6)
        for name, expected in (('clean_sim', 2.6), ('sparse_sim', 4.6),
                               ('et_proxy', 6.6), ('ccdm_proxy', 8.6)):
            assert metrics[f'val_transfer_prompt/{name}_score'] == pytest.approx(expected)
    assert results[1]['val_transfer_score'] > results[0]['val_transfer_score']


@pytest.mark.parametrize('missing', ['clean_sim', 'sparse_sim'])
def test_prompt_selection_rejects_missing_component_and_resets_previous_epoch(trainer, monkeypatch, missing):
    trainer._transfer_clean_prompt_score = torch.tensor(123.)
    trainer._transfer_sparse_prompt_score = torch.tensor(456.)
    trainer.on_validation_epoch_start()
    assert trainer._transfer_clean_prompt_score is None
    assert trainer._transfer_sparse_prompt_score is None
    for metrics in trainer._domain_totals.values():
        fill_metrics(metrics, 1.)

    def incomplete_parent(self):
        self.log('val_robustness_score', torch.tensor(1.))
        if missing != 'clean_sim':
            self.log('val_conditioning/prompt_generation/score', torch.tensor(1.))
        if missing != 'sparse_sim':
            self.log('val_caption_missingness_score', torch.tensor(1.))

    monkeypatch.setattr(LightningLensCraft, 'on_validation_epoch_end', incomplete_parent)
    with pytest.raises(RuntimeError, match=f'No {missing} prompt validation score'):
        trainer.on_validation_epoch_end()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_prompt_selection_rejects_nonfinite_component(trainer, monkeypatch, value):
    trainer.on_validation_epoch_start()
    for metrics in trainer._domain_totals.values():
        fill_metrics(metrics, 1.)

    def nonfinite_parent(self):
        self.log('val_robustness_score', torch.tensor(1.))
        self.log('val_conditioning/prompt_generation/score', torch.tensor(value))
        self.log('val_caption_missingness_score', torch.tensor(1.))

    monkeypatch.setattr(LightningLensCraft, 'on_validation_epoch_end', nonfinite_parent)
    with pytest.raises(RuntimeError, match='Nonfinite clean_sim prompt validation score'):
        trainer.on_validation_epoch_end()


def test_scratch_recipe_restores_original_scheduler_and_explicit_prompt_monitor():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    config_dir = str(Path(__file__).resolve().parents[1] / 'config')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name='config', overrides=[
            'training=high_level_scratch', 'checkpoint_monitor=val_transfer_prompt_score',
            'initialize_from_checkpoint=null', 'resume_checkpoint=null',
        ])
    assert config.initialize_from_checkpoint is None and config.resume_checkpoint is None
    assert config.callbacks.val_checkpoint.monitor == 'val_transfer_prompt_score'
    assert config.training.caption_missingness.profile == 'high_level_only'
    assert config.training.caption_missingness.mask_bank_path is None
    weights = config.training.conditioning.mode_weights
    assert weights.prompt_generation / sum(weights.values()) == .5
    assert config.training.domain_augmentation.et_coordinate_frame == 'world'
    assert config.training.optimizer.lr == 1e-4
    optimizer = instantiate(config.training.optimizer)([torch.nn.Parameter(torch.zeros(()))])
    scheduler = instantiate(config.training.lr_scheduler)(optimizer=optimizer, total_steps=46900)
    scheduler.step(10)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(3e-4)
    scheduler.step(46900)
    assert 1e-4 < optimizer.param_groups[0]['lr'] < 3e-4
    for key in ('validation_conditioning', 'validation_caption_missingness',
                'validation_domain_augmentation'):
        assert config.training[key].max_batches == 16


@pytest.mark.parametrize('domain', ['et', 'ccdm'])
@pytest.mark.parametrize('mode', TRAINING_MODES)
def test_checkpoint_selection_rejects_missing_proxy_family(trainer, domain, mode):
    trainer.on_validation_epoch_start()
    for metrics in trainer._conditioning_totals.values():
        fill_metrics(metrics, 1.)
    fill_metrics(trainer._caption_missingness_totals, 1.)
    for case, metrics in trainer._domain_totals.items():
        if case != (domain, mode):
            fill_metrics(metrics, 1.)
    with pytest.raises(RuntimeError, match='No valid proxy validation poses'):
        trainer.on_validation_epoch_end()


def test_invalid_domain_labels_contribute_no_clip_or_prompt_cycle_gradient(trainer, batch, monkeypatch):
    view = trainer.domain_augmentation.apply(batch, force_domain='ccdm')
    caption = trainer._prepare_clip_embeddings(view)
    present = view['prompt_none_mask']
    assert present.any() and (~present).any()

    class CaptureModel(torch.nn.Module):
        def forward(self, **kwargs):
            self.kwargs = kwargs
            return {}

    trainer.model = CaptureModel()
    monkeypatch.setattr(trainer, '_calculate_schedule_parameters', lambda: {
        'noise_std': 0., 'mask_ratio': 0., 'memory_mask_ratio': 0.,
        'memory_teacher_forcing_ratio': 1., 'trajectory_teacher_forcing_ratio': 0.,
    })
    forwarded = trainer._forward_step(
        view['camera_trajectory'], view['subject_trajectory'], view['subject_volume'],
        caption, view['padding_mask'], is_training=False,
        conditioning_mode='prompt_generation', caption_presence_mask=present,
    )
    assert torch.equal(trainer.model.kwargs['caption_presence_mask'], present)
    assert torch.equal(forwarded['cycle_prompt_mask'], present)
    # Nonzero residuals on the removed fields make this fail if either loss
    # accidentally stops honoring its mask, even though unknown fills are zero.
    prediction = torch.full_like(caption, .5, requires_grad=True)
    cycle = torch.full_like(caption, 2., requires_grad=True)
    geometry = torch.zeros(*view['camera_trajectory'].shape[:2], 12)
    total, _ = trainer.loss_module.compute_total_loss(
        geometry, geometry, prediction, caption, view, cycle_embeddings=cycle,
        cycle_prompt_mask=forwarded['cycle_prompt_mask'],
    )
    total.backward()
    invalid = (~present).T[..., None].expand_as(prediction)
    assert torch.count_nonzero(prediction.grad[invalid]).item() == 0
    assert torch.count_nonzero(cycle.grad[invalid]).item() == 0
    assert prediction.grad[~invalid].abs().sum().item() > 0
    assert cycle.grad[~invalid].abs().sum().item() > 0


def test_presence_enabled_model_strictly_loads_existing_checkpoint(tmp_path):
    from models.camera_trajectory_model import LensCraft
    from utils.checkpoint import load_checkpoint

    options = dict(nhead=2, num_encoder_layers=1, num_decoder_layers=1,
                   dim_feedforward=16, dropout_rate=0., seq_length=5, latent_dim=8)
    previous = LensCraft(**options, caption_presence_conditioning=False)
    robust = LensCraft(**options, caption_presence_conditioning=True)
    checkpoint = tmp_path / 'previous.ckpt'
    torch.save({'state_dict': {'model.' + key: value for key, value in previous.state_dict().items()}}, checkpoint)
    load_checkpoint(checkpoint, robust, torch.device('cpu'))
    assert set(previous.state_dict()) == set(robust.state_dict())
    for name, value in previous.state_dict().items():
        torch.testing.assert_close(value, robust.state_dict()[name])
    tokens = torch.ones(len(CLIP_PARAMETERS), 2, 8)
    present = torch.ones(2, len(CLIP_PARAMETERS), dtype=torch.bool)
    present[:, 0] = False
    output = robust._prepare_caption_tokens(tokens, present)
    torch.testing.assert_close(output[0], robust.missing_caption_encoding.expand(2, -1))
    torch.testing.assert_close(output[1:], tokens[1:])


def test_high_level_profile_cycle_masks_dropped_fields_but_clean_encoder_keeps_targets(trainer, batch, monkeypatch):
    trainer.caption_missingness = CaptionMissingness({
        'profile': 'high_level_only', 'probability': 1.,
        'min_keep_ratio': 1., 'max_keep_ratio': 1., 'endpoint_drop_probability': 0.,
    }, vocabulary())
    caption = trainer._prepare_clip_embeddings(batch)
    original_caption = caption.clone()
    original_presence = batch['prompt_none_mask'].clone()

    class CaptureModel(torch.nn.Module):
        def forward(self, **kwargs):
            self.kwargs = kwargs
            return {}

    trainer.model = CaptureModel()
    monkeypatch.setattr(trainer, '_calculate_schedule_parameters', lambda: {
        'noise_std': 0., 'mask_ratio': 0., 'memory_mask_ratio': 0.,
        'memory_teacher_forcing_ratio': 1., 'trajectory_teacher_forcing_ratio': 0.,
    })
    output = trainer._forward_step(
        batch['camera_trajectory'], batch['subject_trajectory'], batch['subject_volume'],
        caption, batch['padding_mask'], is_training=True,
        conditioning_mode='prompt_generation', caption_presence_mask=batch['prompt_none_mask'],
        caption_missingness_seeds=[1, 2],
    )
    effective = trainer.model.kwargs['caption_presence_mask']
    assert torch.equal(output['cycle_prompt_mask'], effective)
    assert effective[:, :cinematography_struct_size].all()
    assert not effective[:, cinematography_struct_size:].any()
    assert torch.equal(caption, original_caption)
    assert torch.equal(batch['prompt_none_mask'], original_presence)
    assert torch.equal(trainer.model.kwargs['caption_embedding'][:cinematography_struct_size],
                       original_caption[:cinematography_struct_size])

    prediction = torch.full_like(caption, .5, requires_grad=True)
    cycle = torch.full_like(caption, 2., requires_grad=True)
    geometry = torch.zeros(*batch['camera_trajectory'].shape[:2], 12)
    total, _ = trainer.loss_module.compute_total_loss(
        geometry, geometry, prediction, caption, batch, cycle_embeddings=cycle,
        cycle_prompt_mask=output['cycle_prompt_mask'],
    )
    total.backward()
    dropped = (~effective).T[..., None].expand_as(prediction)
    # The camera encoder still supervises every observed clean annotation.
    # Only the prompt-conditioned cycle excludes omitted decoder conditions.
    assert prediction.grad[dropped].abs().sum().item() > 0
    assert torch.count_nonzero(cycle.grad[dropped]).item() == 0
    assert cycle.grad[~dropped].abs().sum().item() > 0


def test_auto_monitor_prefers_transfer_over_sparse_or_clean():
    settings = {'validation_domain_augmentation': {'enabled': True},
                'validation_caption_missingness': {'enabled': True},
                'validation_conditioning': {'enabled': True}}
    assert checkpoint_monitor('auto', settings) == 'val_transfer_score'
    settings['validation_domain_augmentation']['enabled'] = False
    assert checkpoint_monitor('auto', settings) == 'val_robustness_score'
    assert checkpoint_monitor('custom_validation', settings) == 'custom_validation'
