"""Checkpoint objectives for supervised ET adaptation and simulation replay."""

import pytest
import torch

from training.autoencoder_trainer import LightningLensCraft
from training.base_trainer import BaseTrainer, MaskConfig, NoiseConfig, TeacherForcingConfig
from training.et_adaptation_trainer import ETAdaptationTrainer


@pytest.fixture
def make_trainer(monkeypatch):
    def record(self, name, value, *args, **kwargs):
        self.recorded_metrics[name] = torch.as_tensor(value).detach().clone()

    monkeypatch.setattr(BaseTrainer, 'log', record)
    monkeypatch.setattr(BaseTrainer, 'on_validation_epoch_end', lambda self: None)

    def create(**kwargs):
        module = ETAdaptationTrainer(
            model=torch.nn.Linear(6, 6), optimizer=None, lr_scheduler=None,
            loss_module=None,
            noise=NoiseConfig(0., 0.), mask=MaskConfig(0., 0.),
            teacher_forcing_schedule=TeacherForcingConfig(1., 1., 0., 0.),
            compile_enabled=False,
            validation_conditioning={
                'keyframe_counts': [1, 4], 'max_batches': 0,
                'known_pose_weight': 1., 'motion_score_weight': 1.,
            },
            **kwargs,
        )
        module.recorded_metrics = {}
        return module

    return create


def fill_case(totals, position, **means):
    """Populate valid additive validation results, including motion counts."""
    means = {'position_error_normalized': position, **means}
    for name, pair in totals.items():
        pair.copy_(pair.new_tensor([5 * means.get(name, 0.), 5.]))


def populate_validation(module, sim_score=2.):
    module.on_validation_epoch_start()
    for domain, state in module._domain_states.items():
        state['_conditioning_sample_offset'] = 5
        for name, totals in state['_conditioning_totals'].items():
            if domain == 'simulation':
                fill_case(totals, sim_score)
            elif name == 'prompt_generation':
                # Absolute pose: 2 + 90/180. Motion: .25 +
                # (36/180 + 18/180 + 54/180)/3. Full score = 2.95.
                fill_case(totals, 2., rotation_error_deg=90.,
                          step_error_normalized=.25, relative_rotation_error_deg=36.,
                          rotation_step_error_deg=18., rotation_path_error_deg=54.)
            elif name == 'reconstruction':
                fill_case(totals, 8.)
            else:
                fill_case(totals, 6. if '+' in name else 4.,
                          known_position_error_normalized=1., known_rotation_error_deg=36.)


def test_balanced_objective_remains_the_default(make_trainer):
    module = make_trainer()
    assert module.et_validation_objective == 'balanced'
    populate_validation(module)
    module.on_validation_epoch_end()
    # Two K values belong to one family each, retaining equal family weights.
    expected = (2.95 + 8. + 5.2 + 7.2) / 4
    assert module.et_validation_score.item() == pytest.approx(expected)
    assert module.recorded_metrics['val_adaptation_score'].item() == pytest.approx(expected)


@pytest.mark.parametrize('objective,et_score', [
    ('balanced', (2.95 + 8. + 5.2 + 7.2) / 4),
    ('prompt_generation', 2.95),
])
@pytest.mark.parametrize('sim_score', [1., 3.])
def test_selected_et_objective_keeps_one_sided_simulation_regression(
    make_trainer, objective, et_score, sim_score,
):
    module = make_trainer(et_validation_objective=objective, simulation_validation_weight=2.)
    module.set_reference_simulation_score(2.)
    populate_validation(module, sim_score)
    module.on_validation_epoch_end()
    metrics = module.recorded_metrics
    assert module.et_validation_score.item() == pytest.approx(et_score)
    assert metrics['val_et_selection_score'].item() == pytest.approx(et_score)
    assert metrics['val_sim_conditioning_score'].item() == pytest.approx(sim_score)
    assert metrics['val_sim_regression'].item() == pytest.approx(sim_score - 2.)
    assert metrics['val_adaptation_score'].item() == pytest.approx(
        et_score + 2 * max(sim_score - 2., 0.)
    )
    # Balanced ET diagnostics retain their original meaning in prompt mode.
    assert module._domain_scores['et'].item() == pytest.approx((2.95 + 8. + 5.2 + 7.2) / 4)


def test_prompt_selection_excludes_supplied_et_poses_but_sim_penalty_stays_balanced(make_trainer):
    module = make_trainer(et_validation_objective='prompt_generation', simulation_validation_weight=2.)
    module.set_reference_simulation_score(2.)
    populate_validation(module)
    et_cases = module._domain_states['et']['_conditioning_totals']
    for name, totals in et_cases.items():
        if name != 'prompt_generation':
            fill_case(totals, 100., known_position_error_normalized=100.)
    sim_cases = module._domain_states['simulation']['_conditioning_totals']
    fill_case(sim_cases['reconstruction'], 10.)
    module.on_validation_epoch_end()
    assert module.et_validation_score.item() == pytest.approx(2.95)
    # SIM prompt remains 2, but its full-family score is (2+10+2+2)/4 = 4.
    assert module._domain_scores['simulation'].item() == pytest.approx(4.)
    assert module.recorded_metrics['val_adaptation_score'].item() == pytest.approx(6.95)


def test_validation_score_cannot_be_reused_from_previous_epoch(make_trainer):
    module = make_trainer(et_validation_objective='prompt_generation')
    populate_validation(module)
    module.on_validation_epoch_end()
    assert module.et_validation_score.item() == pytest.approx(2.95)
    module.on_validation_epoch_start()
    with pytest.raises(RuntimeError, match='ET validation must complete'):
        _ = module.et_validation_score


@pytest.mark.parametrize('objective', ['prompt', '', None])
def test_unknown_validation_objective_is_rejected(make_trainer, objective):
    with pytest.raises(ValueError, match='et_validation_objective'):
        make_trainer(et_validation_objective=objective)


def test_replay_weight_is_a_loss_share_and_et_keeps_native_caption_masks(make_trainer, monkeypatch):
    module = make_trainer(simulation_replay_ratio=.25)
    original_missingness = object()
    module.caption_missingness = original_missingness
    observed = []

    def step(self, batch, batch_idx, stage):
        observed.append((self.dataset_mode, self.caption_missingness, stage))
        return batch['loss']

    monkeypatch.setattr(LightningLensCraft, '_step', step)
    et_loss = torch.tensor(2., requires_grad=True)
    sim_loss = torch.tensor(6., requires_grad=True)
    total = module.training_step({
        'et': {'camera_trajectory': torch.zeros(2, 3, 6), 'loss': et_loss},
        'simulation': {'loss': sim_loss},
    }, 0)
    total.backward()
    assert total.item() == pytest.approx(3.)
    assert et_loss.grad.item() == pytest.approx(.75)
    assert sim_loss.grad.item() == pytest.approx(.25)
    assert observed == [('et', None, 'train'), ('simulation', original_missingness, 'train')]
    assert module.caption_missingness is original_missingness
    assert module._metric_domain is None
