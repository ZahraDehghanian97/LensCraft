"""Training interventions must not change the CCDM selection reference."""

import pytest
import torch

from data.simulation.dataset import SimulationDataset
from models.categorical_latent import FIELD_NAMES, from_ids

cinematography_struct_size = len(FIELD_NAMES)
from training.base_trainer import MaskConfig, NoiseConfig, TeacherForcingConfig
from training.ccdm_focus_trainer import CCDMFocusTrainer
from training.losses.categorical_loss import CategoricalCameraTrajectoryLoss
from training.transfer_robust_trainer import TransferRobustTrainer



def make_trainer(control):
    model = torch.nn.Linear(6, 6)
    return CCDMFocusTrainer(
        model=model, optimizer=None, lr_scheduler=None,
        loss_module=CategoricalCameraTrajectoryLoss(losses_list={'categorical': 1., 'cycle': 1.}),
        noise=NoiseConfig(0., 0.), mask=MaskConfig(0., 0.),
        teacher_forcing_schedule=TeacherForcingConfig(1., 1., 0., 0.),
        compile_enabled=False, conditioning={'keyframe_counts': [2]},
        validation_conditioning={'keyframe_counts': [2], 'max_batches': 2},
        caption_missingness={'probability': 1., 'fill': 'zero'},
        validation_caption_missingness={'max_batches': 2},
        domain_augmentation={
            'enabled': True, 'et_probability': 0., 'ccdm_probability': .75,
            'ccdm_static_motion': not control, 'ccdm_native_manifold': not control,
        },
        validation_domain_augmentation={'max_batches': 2, 'keyframe_count': 2},
    )


def example_batch():
    size, frames, width = 2, 5, 8
    camera = torch.zeros(size, frames, 6)
    camera[..., 0] = torch.arange(frames).float() / 10
    camera[..., 2] = 5.
    subject = torch.zeros_like(camera)
    ids = torch.zeros(size, 10, dtype=torch.long)
    caption, presence = from_ids(ids)
    return {
        'camera_trajectory': camera, 'subject_trajectory': subject,
        'subject_volume': torch.ones(size, 1, 3),
        'padding_mask': torch.zeros(size, frames, dtype=torch.bool),
        'cinematography_ids': ids,
        'prompt_none_mask': presence,
        'cinematography_prompt_parameters': [
            [(name, 'dollyIn' if name == 'movement_type' else 'observed', 0, None)
             for name in FIELD_NAMES]
            for _ in range(size)
        ],
    }


def test_recipes_share_validation_targets_and_captions_and_restore_training(monkeypatch):
    monkeypatch.setattr(SimulationDataset, 'normalize_item',
                        staticmethod(lambda c, s, v, n: (c, s, v)))
    batch = example_batch()
    calls = []

    def capture_validation(self, value, batch_idx):
        calls.append(self.domain_augmentation.apply(value, force_domain='ccdm'))

    monkeypatch.setattr(TransferRobustTrainer, '_validate_domain_batch', capture_validation)
    trainers = [make_trainer(control=True), make_trainer(control=False)]
    training_operators = [trainer.domain_augmentation for trainer in trainers]
    for trainer in trainers:
        trainer._validate_domain_batch(batch, 0)
    for key in ('camera_trajectory', 'subject_trajectory', 'subject_volume',
                'cinematography_ids', 'prompt_none_mask'):
        torch.testing.assert_close(calls[0][key], calls[1][key])
    for trainer, operator in zip(trainers, training_operators):
        assert trainer.domain_augmentation is operator
    assert training_operators[0].config.ccdm_native_manifold is False
    assert training_operators[1].config.ccdm_native_manifold is True
    # Their training captions remain distinct on this exact stationary,
    # upright and native-representable motion; only validation is shared.
    movement = trainers[0].domain_augmentation.field_names.index('movement_type')
    control = training_operators[0].apply(batch, force_domain='ccdm')
    motion = training_operators[1].apply(batch, force_domain='ccdm')
    assert not control['prompt_none_mask'][:, movement].any()
    assert motion['prompt_none_mask'][:, movement].all()


def test_validation_restores_training_operator_when_validation_fails(monkeypatch):
    trainer = make_trainer(control=True)
    training_operator = trainer.domain_augmentation

    def fail(self, batch, batch_idx):
        assert self.domain_augmentation is self.ccdm_validation_augmentation
        raise RuntimeError('validation failed')

    monkeypatch.setattr(TransferRobustTrainer, '_validate_domain_batch', fail)
    with pytest.raises(RuntimeError, match='validation failed'):
        trainer._validate_domain_batch({}, 0)
    assert trainer.domain_augmentation is training_operator
