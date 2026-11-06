#!/usr/bin/env python3
"""Train from scratch or fine-tune on explicit ET train/validation inventories.

Prepare --data-dir with prepare_et_adaptation.py first. This runner creates two
ETDataset instances and fixed native-split loaders; it never randomly splits the
annotation inventory. Optional SIM replay uses only the original training split;
checkpoint selection also penalizes regression on original SIM validation. The model's
simulation coordinate normalization and architecture come from --base-config.
With --from-scratch, initialize the complete generator randomly and optimize and
select checkpoints using ET only; no generator checkpoint or SIM loader is used.
Pass OUTPUT/inference_config.yaml to evaluate_et_robustness.py --train-config.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-config', 'checkpoint', 'data-dir', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Randomly initialize LensCraft; no checkpoint, SIM replay, or SIM validation selection.')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--director-config', type=Path)
    parser.add_argument('--epochs', type=int, help='Defaults to 100 from scratch, 10 for fine-tuning.')
    parser.add_argument('--train-batches', type=int,
                        help='Maximum ET batches per epoch; zero runs one complete ET pass, including with SIM replay.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, help='Constant learning rate; defaults to 1e-4 from scratch, 1e-5 for fine-tuning.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--conditioning-val-batches', type=int, default=0,
                        help='Zero evaluates conditioning and motion over all ET validation batches.')
    parser.add_argument('--val-batches', type=int, default=0, help='Zero covers the full explicit ET validation split.')
    parser.add_argument('--simulation-replay-ratio', type=float, default=0.0,
                        help='SIM share of each mixed training loss, in [0, 1); zero disables replay.')
    parser.add_argument('--simulation-val-batches', type=int, default=4,
                        help='Positive number of batches from the original SIM validation split.')
    parser.add_argument('--simulation-validation-weight', type=float,
                        help='Weight of SIM regression penalty in checkpoint selection; zero selects on ET only.')
    parser.add_argument('--et-validation-objective', choices=('balanced', 'prompt_generation'), default='balanced',
                        help='ET objective for checkpoint selection; prompt_generation matches native ET generation.')
    parser.add_argument('--replay-caption-missingness', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--caption-presence-conditioning', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0', help='cuda:N or cpu')
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    for name, value in (('epochs', 100 if args.from_scratch else 10),
                        ('train_batches', 0 if args.from_scratch else 100),
                        ('lr', 1e-4 if args.from_scratch else 1e-5),
                        ('simulation_validation_weight', 0.0 if args.from_scratch else 1.0)):
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ('base_config', 'data_dir', 'output'):
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required')
    if args.from_scratch:
        if args.checkpoint is not None:
            parser.error('--from-scratch cannot be combined with --checkpoint')
        if args.simulation_replay_ratio != 0 or args.simulation_validation_weight != 0:
            parser.error('--from-scratch requires zero SIM replay and zero SIM validation weight')
    elif args.checkpoint is None:
        parser.error('--checkpoint is required unless --from-scratch is specified')
    for name in ('epochs', 'batch_size', 'simulation_val_batches'):
        if getattr(args, name) < 1:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    if (args.train_batches < 0 or args.num_workers < 0 or args.val_batches < 0
            or args.conditioning_val_batches < 0 or not 0 <= args.seed < 2**32):
        parser.error('train-batches/workers/val-batches must be nonnegative and seed must be in [0, 2**32)')
    if not math.isfinite(args.simulation_replay_ratio) or not 0 <= args.simulation_replay_ratio < 1:
        parser.error('--simulation-replay-ratio must be finite and in [0, 1)')
    if not math.isfinite(args.simulation_validation_weight) or args.simulation_validation_weight < 0:
        parser.error('--simulation-validation-weight must be finite and nonnegative')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if args.device != 'cpu' and not (args.device.startswith('cuda:') and args.device[5:].isdigit()):
        parser.error('--device must be cpu or cuda:N')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def verify_prepared_data(directory):
    """Check inventory and provenance before importing or loading a model."""
    from prepare_et_adaptation import excluded_ids, video_id

    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest.get('output_sha256', {})
    required = ('et-data/pure_train_split.txt', 'et-data/pure_val_split.txt',
                'train_annotations.json', 'val_annotations.json', 'annotations.json')
    for name in required:
        if name not in hashes or sha256(directory / name) != hashes[name]:
            raise ValueError(f'Prepared ET output does not match its manifest: {name}')
    ids = {}
    for split, manifest_key in (('train', 'train_ids'), ('val', 'validation_ids')):
        raw = (directory / f'et-data/pure_{split}_split.txt').read_text()
        values = raw.split('\n')
        if not values or not all(values) or values != sorted(set(values)):
            raise ValueError(f'Prepared {split} IDs must be nonempty, sorted and unique without a trailing newline')
        if values != manifest.get(manifest_key):
            raise ValueError(f'Prepared {split} inventory differs from its manifest')
        rows = json.loads((directory / f'{split}_annotations.json').read_text())
        annotation_ids = [row.get('custom_id') for row in rows]
        if annotation_ids != ['et-' + value for value in values]:
            raise ValueError(f'Prepared {split} annotations must exactly match its ordered inventory')
        ids[split] = values
    training, validation = set(ids['train']), set(ids['val'])
    selected = training | validation
    held_out = manifest.get('held_out_ids')
    if not isinstance(held_out, list) or not held_out:
        raise ValueError('Preparation manifest must record nonempty held_out_ids')
    if training & validation or selected & set(held_out):
        raise ValueError('ET train/validation/held-out inventories overlap')
    if manifest.get('group_by') != 'video':
        raise ValueError('ET adaptation requires preparation with --group-by video for source-video isolation')
    # Recheck official source lists as well as the recorded collision checks.
    sources = manifest.get('source_splits', {})
    verified_held_out = set()
    for name in ('pure_train', 'pure_test', 'mixed_test', 'full_test', 'clatr_test'):
        info = sources.get(name, {})
        source = Path(info.get('path', ''))
        if not source.is_file() or sha256(source) != info.get('sha256'):
            raise ValueError(f'Original ET split provenance no longer verifies: {name}')
        original_ids = set(source.read_text().splitlines())
        if (name == 'pure_train' and not selected <= original_ids) or (name != 'pure_train' and selected & original_ids):
            raise ValueError(f'Prepared ET IDs violate source split: {name}')
        if name != 'pure_train':
            verified_held_out.update(original_ids)
    for info in manifest.get('explicit_excluded_cohorts', []):
        source = Path(info.get('path', ''))
        if not source.is_file() or sha256(source) != info.get('sha256'):
            raise ValueError('Explicit ET exclusion cohort provenance no longer verifies')
        verified_held_out.update(excluded_ids(source))
    if not verified_held_out <= set(held_out):
        raise ValueError('Prepared manifest omits IDs from official or explicit held-out cohorts')
    train_videos, val_videos = {video_id(value) for value in training}, {video_id(value) for value in validation}
    held_out_videos = {video_id(value) for value in held_out}
    if train_videos & val_videos or (train_videos | val_videos) & held_out_videos:
        raise ValueError('ET canonical source-video isolation failed (year aliases share one video)')
    return manifest, ids


def et_epoch_batch_limit(train_count, batch_size, requested_batches):
    """Cap replay cycling at one ET pass; zero means every ET train example."""
    if train_count < 1 or batch_size < 1 or requested_batches < 0:
        raise ValueError('ET batch budgeting needs positive sample/batch counts and nonnegative batch limit')
    full = (train_count + batch_size - 1) // batch_size
    return full if requested_batches == 0 else min(full, requested_batches)


def select_checkpoint(initial_path, initial_score, candidate_path, candidate_score):
    """The unmodified initial model remains eligible when adaptation regresses."""
    if not candidate_path or not all(math.isfinite(float(value)) for value in (initial_score, candidate_score)):
        raise ValueError('Checkpoint selection requires finite initial and fine-tuned scores and a candidate path')
    selected_initial = float(candidate_score) >= float(initial_score)
    return (str(initial_path), float(initial_score), True) if selected_initial else (
        str(candidate_path), float(candidate_score), False,
    )


def initialize_model(module_config, checkpoint, *, from_scratch, instantiate, load_checkpoint, device):
    """Make the generator's weight origin explicit and testable."""
    if from_scratch and checkpoint is not None:
        raise ValueError('Scratch initialization cannot load a generator checkpoint')
    if not from_scratch and checkpoint is None:
        raise ValueError('Fine-tuning requires a generator checkpoint')
    model = instantiate(module_config)
    return model if from_scratch else load_checkpoint(str(checkpoint), model, device)


def build_configs(base_config, args):
    """Return truthful ET training config and compatible SIM inference config."""
    from omegaconf import OmegaConf

    base = OmegaConf.to_container(base_config, resolve=True)
    simulation = deepcopy(base['data']['dataset'])
    if simulation['config'].get('_target_') != 'data.simulation.dataset.SimulationDataset':
        raise ValueError('--base-config must specify the original SimulationDataset normalization reference')
    if not simulation['config'].get('normalize', True):
        raise ValueError('ET training requires the original simulation normalization reference')
    embeddings = deepcopy(base['training']['loss_module']['clip_embeddings'])
    if embeddings.get('embedding_mode', 'default') != 'default':
        raise ValueError('ET adaptation mean filling requires default CLIP embedding mode')
    dataset = {
        '_target_': 'data.et.dataset.ETDataset',
        'project_config_dir': str(args.director_config),
        'dataset_dir': str(args.data_dir / 'et-data'),
        'et_cin_lang_path': str(args.data_dir / 'train_annotations.json'),
        'set_name': 'pure', 'split': 'train', 'normalize': True,
        'fill_none_with_mean': True, 'clip_embeddings': embeddings,
    }
    validation_dataset = deepcopy(dataset)
    validation_dataset.update(split='val', et_cin_lang_path=str(args.data_dir / 'val_annotations.json'))
    training = deepcopy(base)
    training['data'] = {
        'dataset': {'info': {'max_seq_length': 300}, 'config': dataset},
        'validation_dataset': {'info': {'max_seq_length': 300}, 'config': validation_dataset},
        'batch_size': args.batch_size, 'num_workers': args.num_workers,
        'split_strategy': 'explicit prepared ET pure_train and pure_val; no random split',
        'prepared_manifest': str(args.data_dir / 'manifest.json'),
    }
    settings = training['training']
    settings['_target_'] = ('training.autoencoder_trainer.LightningLensCraft' if args.from_scratch
                           else 'training.et_adaptation_trainer.ETAdaptationTrainer')
    # Simulation proxy augmentation belongs to TransferRobustTrainer only.
    # Passing these inherited keys through ETAdaptationTrainer's **kwargs
    # would fail in LightningLensCraft, and native ET must keep native inputs.
    settings.pop('domain_augmentation', None)
    settings.pop('validation_domain_augmentation', None)
    settings['dataset_mode'] = 'et'
    for key, value in (('simulation_replay_ratio', args.simulation_replay_ratio),
                       ('simulation_validation_weight', args.simulation_validation_weight),
                       ('et_validation_objective', args.et_validation_objective)):
        settings.pop(key, None)
        if not args.from_scratch:
            settings[key] = value
    settings['model']['module']['caption_presence_conditioning'] = args.caption_presence_conditioning
    if args.from_scratch:
        if settings['model']['module'].get('_target_') != 'models.camera_trajectory_model.LensCraft':
            raise ValueError('Scratch training requires the LensCraft generator constructor')
        settings['model']['inference']['checkpoint_path'] = None
    settings['optimizer']['lr'] = args.lr
    settings['lr_scheduler'] = None
    settings['conditioning'] = {
        'enabled': True,
        'mode_weights': {'prompt_generation': 3., 'reconstruction': 1., 'key_framing': 1., 'key_framing+prompt': 1.},
        'keyframe_counts': [1, 2, 4, 8, 26],
    }
    settings['validation_conditioning'] = {
        'enabled': True, 'keyframe_counts': [1, 4],
        'max_batches': args.conditioning_val_batches, 'seed': args.seed, 'known_pose_weight': 1.,
        'motion_score_weight': 1., 'rotation_baselines': True, 'caption_shuffle': True,
    }
    settings['caption_missingness'] = {
        'enabled': True, 'probability': .75, 'min_keep_ratio': .15, 'max_keep_ratio': .45,
        'endpoint_drop_probability': .75, 'preserve_movement_type': True,
        'fill': 'mean', 'mask_bank_path': None,
    } if args.simulation_replay_ratio and args.replay_caption_missingness else None
    settings['validation_caption_missingness'] = None
    # The source camera enters the encoder alignment task; the prompt decoder
    # remains caption-only through the existing explicit conditioning contract.
    settings['decode_mode'] = 'single_step'
    training['seed'] = args.seed
    training['initialize_from_checkpoint'] = None if args.from_scratch else str(args.checkpoint)
    training['initialization'] = {'kind': 'random' if args.from_scratch else 'checkpoint',
                                  'generator_checkpoint': training['initialize_from_checkpoint'],
                                  'seed': args.seed}
    training['resume_checkpoint'] = None
    training['compile'] = {'enabled': False, 'mode': 'default'}
    monitor = ('val_conditioning/prompt_generation/score'
               if args.et_validation_objective == 'prompt_generation' else 'val_conditioning_score')
    training['checkpoint_monitor'] = monitor if args.from_scratch else 'val_adaptation_score'
    training['output'] = {'directory': str(args.output)}
    training['logger'] = {
        '_target_': 'lightning.pytorch.loggers.CSVLogger',
        'save_dir': str(args.output), 'name': 'metrics',
    }
    training['trainer'] = {
        '_target_': 'lightning.Trainer', 'max_epochs': args.epochs,
        'accelerator': 'cpu' if args.device == 'cpu' else 'gpu',
        'devices': 1 if args.device == 'cpu' else [int(args.device[5:])],
        'precision': '32-true' if args.device == 'cpu' else 'bf16-mixed',
        'gradient_clip_val': 1., 'limit_train_batches': args.train_batches or 1.0,
        # Each validation dataset is bounded explicitly below, independently.
        'limit_val_batches': 1.0,
        'num_sanity_val_steps': 0, 'enable_progress_bar': False,
        'log_every_n_steps': 10, 'default_root_dir': str(args.output),
    }
    training['callbacks'] = {
        'val_checkpoint': {
            '_target_': 'lightning.pytorch.callbacks.ModelCheckpoint',
            'dirpath': str(args.output / 'checkpoints'), 'monitor': training['checkpoint_monitor'],
            'filename': ('best-et-val-{epoch:03d}' if args.from_scratch
                         else 'best-et-val-{epoch:03d}-{val_adaptation_score:.5f}'),
            'save_top_k': 1, 'save_last': True, 'mode': 'min',
        },
    }
    normalization = {
        'base_config': str(args.base_config),
        'dataset': simulation,
        'effective_data_path': os.environ.get('SIMULATION_NORMALIZATION_DATA_PATH') or simulation['config'].get('data_path'),
        'parameters': str(args.output / 'normalization.json'),
        'coordinate_contract': 'ET native standardization -> world conversion -> original simulation statistics',
    }
    training['normalization_reference'] = normalization
    training['simulation_replay'] = None if args.from_scratch else {
        'dataset': deepcopy(simulation),
        'original_data_config': deepcopy(base['data']),
        'loss_ratio': args.simulation_replay_ratio,
        'validation_batches': args.simulation_val_batches,
        'split_manifest': str(args.output / 'simulation_splits.json'),
        'et_validation_batches': args.val_batches,
    }
    inference = deepcopy(base)
    # Evaluation retains the architecture and normalization, while removing
    # inherited training initialization/resume claims from the template config.
    inference['initialize_from_checkpoint'] = training['initialize_from_checkpoint']
    inference['resume_checkpoint'] = None
    inference['initialization'] = deepcopy(training['initialization'])
    inference['training']['model']['module']['caption_presence_conditioning'] = args.caption_presence_conditioning
    if args.from_scratch:
        inference['training']['model']['inference']['checkpoint_path'] = None
    inference['normalization_reference'] = deepcopy(normalization)
    inference['adaptation'] = {
        'training_config': str(args.output / 'training_config.yaml'),
        'prepared_manifest': str(args.data_dir / 'manifest.json'),
        'training_dataset': 'ET pure_train prepared view',
        'training_mode': 'from_scratch' if args.from_scratch else 'fine_tuning',
        'selection_dataset': ('ET pure_val only; no SIM validation or test-based checkpoint selection'
                              if args.from_scratch else
                              'ET pure_val plus regression on original SIM validation; no test-based checkpoint selection'),
        'et_validation_objective': args.et_validation_objective,
        'inference_only_config': True,
    }
    inference['training']['model']['inference']['config'] = str(args.output / 'inference_config.yaml')
    return OmegaConf.create(training), OmegaConf.create(inference)


def self_test():
    from contextlib import redirect_stderr
    import importlib.util
    import io
    import unittest
    from unittest.mock import Mock

    class RunnerTests(unittest.TestCase):
        def test_full_et_epoch_does_not_expand_to_sim_replay_length(self):
            self.assertEqual(et_epoch_batch_limit(130, 64, 0), 3)
            self.assertEqual(et_epoch_batch_limit(130, 64, 100), 3)
            self.assertEqual(et_epoch_batch_limit(130, 64, 2), 2)

        def test_initial_checkpoint_survives_regression_or_tie(self):
            self.assertEqual(select_checkpoint('/initial', 1., '/candidate', 2.), ('/initial', 1., True))
            self.assertEqual(select_checkpoint('/initial', 1., '/candidate', 1.), ('/initial', 1., True))
            self.assertEqual(select_checkpoint('/initial', 1., '/candidate', .9), ('/candidate', .9, False))
            with self.assertRaises(ValueError):
                select_checkpoint('/initial', float('nan'), '/candidate', .9)

        def test_cli_accepts_complete_training_and_prompt_objective(self):
            args = parse_args(['--base-config', '/config', '--checkpoint', '/checkpoint',
                               '--data-dir', '/data', '--output', '/output', '--train-batches', '0',
                               '--et-validation-objective', 'prompt_generation'])
            self.assertEqual(args.train_batches, 0)
            self.assertEqual(args.et_validation_objective, 'prompt_generation')

        def test_scratch_defaults_and_forbidden_checkpoint_or_replay(self):
            required = ['--base-config', '/config', '--data-dir', '/data', '--output', '/output']
            args = parse_args(required + ['--from-scratch'])
            self.assertIsNone(args.checkpoint)
            self.assertEqual((args.epochs, args.train_batches, args.lr), (100, 0, 1e-4))
            self.assertEqual(args.simulation_validation_weight, 0)
            for invalid in (['--checkpoint', '/old'], ['--simulation-replay-ratio', '.25'],
                            ['--simulation-validation-weight', '1']):
                with self.subTest(invalid=invalid), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(required + ['--from-scratch'] + invalid)
            normal = parse_args(required + ['--checkpoint', '/old'])
            self.assertEqual((normal.epochs, normal.train_batches, normal.lr), (10, 100, 1e-5))
            self.assertEqual(normal.simulation_validation_weight, 1)

        def test_scratch_initialization_never_calls_checkpoint_loader(self):
            model = object()
            constructor, loader = Mock(return_value=model), Mock(side_effect=AssertionError('Checkpoint loaded'))
            self.assertIs(initialize_model({'architecture': 'lenscraft'}, None, from_scratch=True,
                                           instantiate=constructor, load_checkpoint=loader, device='cpu'), model)
            constructor.assert_called_once_with({'architecture': 'lenscraft'})
            loader.assert_not_called()
            with self.assertRaises(ValueError):
                initialize_model({}, '/old', from_scratch=True, instantiate=constructor,
                                 load_checkpoint=loader, device='cpu')

        def test_finetuning_still_loads_checkpoint(self):
            model, restored = object(), object()
            constructor, loader = Mock(return_value=model), Mock(return_value=restored)
            self.assertIs(initialize_model({}, '/old', from_scratch=False, instantiate=constructor,
                                           load_checkpoint=loader, device='cpu'), restored)
            loader.assert_called_once_with('/old', model, 'cpu')

        @unittest.skipUnless(importlib.util.find_spec('omegaconf'), 'OmegaConf required for config integration')
        def test_scratch_config_has_only_et_training_and_selection(self):
            from omegaconf import OmegaConf
            args = parse_args(['--base-config', '/config', '--from-scratch', '--data-dir', '/data',
                               '--output', '/output', '--et-validation-objective', 'prompt_generation'])
            args.director_config = Path('/director/configs/config.yaml')
            base = OmegaConf.create({
                'initialize_from_checkpoint': '/stale', 'resume_checkpoint': '/stale-resume',
                'data': {'dataset': {'config': {
                    '_target_': 'data.simulation.dataset.SimulationDataset', 'data_path': '/original',
                    'normalize': True,
                }}},
                'training': {
                    '_target_': 'training.transfer_robust_trainer.TransferRobustTrainer',
                    'domain_augmentation': {'enabled': True},
                    'validation_domain_augmentation': {'enabled': True},
                    'simulation_replay_ratio': .25, 'simulation_validation_weight': 1,
                    'et_validation_objective': 'balanced',
                    'loss_module': {'clip_embeddings': {'embedding_mode': 'default'}},
                    'model': {'module': {'_target_': 'models.camera_trajectory_model.LensCraft'},
                              'inference': {'config': '/config', 'checkpoint_path': '/stale-model'}},
                    'optimizer': {'lr': 1e-5},
                },
            })
            training, inference = build_configs(base, args)
            self.assertEqual(training.training._target_, 'training.autoencoder_trainer.LightningLensCraft')
            self.assertEqual(training.training.dataset_mode, 'et')
            self.assertEqual(training.checkpoint_monitor, 'val_conditioning/prompt_generation/score')
            self.assertEqual(training.callbacks.val_checkpoint.monitor, training.checkpoint_monitor)
            self.assertIsNone(training.simulation_replay)
            for key in ('domain_augmentation', 'validation_domain_augmentation', 'simulation_replay_ratio',
                        'simulation_validation_weight', 'et_validation_objective'):
                self.assertNotIn(key, training.training)
            for config in (training, inference):
                self.assertIsNone(config.initialize_from_checkpoint)
                self.assertIsNone(config.resume_checkpoint)
                self.assertIsNone(config.training.model.inference.checkpoint_path)
                self.assertEqual(config.initialization.kind, 'random')
            self.assertEqual(training.training.optimizer.lr, 1e-4)
            self.assertEqual(training.trainer.max_epochs, 100)
            self.assertEqual(training.trainer.limit_train_batches, 1.0)
            self.assertEqual(training.normalization_reference.dataset.config.data_path, '/original')
            self.assertEqual(inference.adaptation.training_mode, 'from_scratch')
            self.assertIn('ET pure_val only', inference.adaptation.selection_dataset)
            self.assertEqual(base.initialize_from_checkpoint, '/stale')

        @unittest.skipUnless(importlib.util.find_spec('omegaconf'), 'OmegaConf required for config integration')
        def test_transfer_config_initializes_native_et_without_proxy_kwargs(self):
            from omegaconf import OmegaConf
            args = parse_args(['--base-config', '/config', '--checkpoint', '/checkpoint',
                               '--data-dir', '/data', '--output', '/output', '--train-batches', '0',
                               '--et-validation-objective', 'prompt_generation'])
            args.director_config = Path('/director/configs/config.yaml')
            base = OmegaConf.create({
                'data': {'dataset': {'config': {
                    '_target_': 'data.simulation.dataset.SimulationDataset', 'data_path': '/original',
                    'normalize': True,
                }}},
                'training': {
                    '_target_': 'training.transfer_robust_trainer.TransferRobustTrainer',
                    'domain_augmentation': {'enabled': True},
                    'validation_domain_augmentation': {'enabled': True},
                    'loss_module': {'clip_embeddings': {'embedding_mode': 'default'}},
                    'model': {'module': {'caption_presence_conditioning': True}, 'inference': {'config': '/config'}},
                    'optimizer': {'lr': 1e-5},
                },
            })
            training, inference = build_configs(base, args)
            self.assertEqual(training.training._target_, 'training.et_adaptation_trainer.ETAdaptationTrainer')
            self.assertNotIn('domain_augmentation', training.training)
            self.assertNotIn('validation_domain_augmentation', training.training)
            self.assertEqual(training.training.et_validation_objective, 'prompt_generation')
            self.assertIsNone(training.training.validation_caption_missingness)
            self.assertEqual(training.trainer.limit_train_batches, 1.0)
            self.assertEqual(training.normalization_reference.dataset.config.data_path, '/original')
            self.assertTrue(inference.training.model.module.caption_presence_conditioning)
            self.assertIn('domain_augmentation', base.training)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    prepared, ids = verify_prepared_data(args.data_dir)
    if not args.from_scratch and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'training_config.yaml', 'status.json'):
        if (args.output / name).exists():
            raise FileExistsError(f'Use a fresh adaptation output directory: {args.output}')
    from dotenv import load_dotenv
    load_dotenv(args.repo / '.env', override=False)
    # Template inference interpolation must not retain an unrelated old model.
    os.environ['TEST_CHECKPOINT_PATH'] = '' if args.from_scratch else str(args.checkpoint)
    os.environ['TEST_CONFIG_PATH'] = str(args.base_config)
    os.environ.setdefault('CLIP_EMBEDDINGS_CACHE_DIR', str(args.repo))
    for name in ('SIMULATION_DATA_PATH', 'SIMULATION_NORMALIZATION_DATA_PATH',
                 'CLIP_EMBEDDINGS_CACHE_DIR', 'DIRECTOR_PROJECT_DIR'):
        if os.environ.get(name):
            path = Path(os.environ[name]).expanduser()
            os.environ[name] = str(path if path.is_absolute() else (args.repo / path).resolve())
    if args.director_config is None:
        args.director_config = Path(os.environ.get('DIRECTOR_PROJECT_DIR', str(args.repo / 'third_parties/DIRECTOR'))) / 'configs/config.yaml'
    sys.path.insert(0, str(args.repo / 'src'))
    import lightning as L
    import torch
    from hydra.utils import instantiate
    from lightning.pytorch.callbacks import ModelCheckpoint
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset
    from lightning.pytorch.utilities.combined_loader import CombinedLoader
    from data.et.dataset import collate_fn
    from data.et_replay import original_simulation_splits
    from data.simulation.dataset import SimulationDataset, collate_fn as simulation_collate_fn
    from utils.checkpoint import load_checkpoint

    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('high')
    # Resolve environment and interpolation before changing to an isolated
    # output directory where CLIP vocabulary moment sidecars are generated.
    base = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(args.base_config), resolve=True))
    cfg, inference_cfg = build_configs(base, args)
    simulation = cfg.normalization_reference.dataset.config
    os.chdir(args.output)
    normalization = SimulationDataset.get_normalization_parameters(
        str(simulation.data_path),
        target_frame_count=int(simulation.get('normalization_frame_count') or simulation.get('target_frame_count', 30)),
    )
    normalization_json = {
        key: {field: value.cpu().tolist() if torch.is_tensor(value) else value for field, value in fields.items()}
        for key, fields in normalization.items()
    }
    write_json(args.output / 'normalization.json', normalization_json)
    OmegaConf.save(base, args.output / 'base_config.yaml', resolve=True)
    OmegaConf.save(cfg, args.output / 'training_config.yaml', resolve=True)
    OmegaConf.save(inference_cfg, args.output / 'inference_config.yaml', resolve=True)
    manifest = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'base_config': {'path': str(args.base_config), 'sha256': sha256(args.base_config)},
        'training_mode': 'from_scratch' if args.from_scratch else 'fine_tuning',
        'initialization': dict(cfg.initialization),
        'initial_checkpoint': (None if args.from_scratch else
                               {'path': str(args.checkpoint), 'sha256': sha256(args.checkpoint)}),
        'prepared_manifest': {'path': str(args.data_dir / 'manifest.json'), 'sha256': sha256(args.data_dir / 'manifest.json')},
        'prepared_output_sha256': prepared['output_sha256'],
        'normalization_sha256': sha256(args.output / 'normalization.json'),
        'training_config_sha256': sha256(args.output / 'training_config.yaml'),
        'source_sha256': {str(path.relative_to(args.repo)): sha256(path) for path in sorted((args.repo / 'src').rglob('*.py'))},
        'script_sha256': sha256(__file__), 'torch_version': torch.__version__,
        'train_count': len(ids['train']), 'validation_count': len(ids['val']),
        'split_grouping': prepared.get('group_by'),
        'protocol': ('Randomly initialized LensCraft trained only on explicit prepared ET training IDs. '
                    'Checkpoint selection uses ET validation only; no simulation replay or validation. '
                    'Original simulation coordinate statistics and frozen CLIP conditioning vocabulary are retained '
                    'for transfer evaluation compatibility; no generator checkpoint is loaded. '
                    'Reviewed/test cohorts are excluded.' if args.from_scratch else
                    'Supervised ET adaptation on explicit prepared train/validation lists; original simulation normalization. '
                    'Optional weighted SIM replay reproduces the original training split; sparse augmentation only on replay. '
                    'Checkpoint score includes ET conditioning/motion and regression from pre-adaptation SIM validation. '
                    'Reviewed/test cohorts are excluded.'),
        'checkpoint_selection': {'monitor': cfg.checkpoint_monitor, 'mode': 'min',
                                 'et_validation_objective': args.et_validation_objective,
                                 'initial_checkpoint_eligible': not args.from_scratch,
                                 'conditioning_validation_max_batches': args.conditioning_val_batches,
                                 'et_validation_max_batches': args.val_batches,
                                 'simulation_validation_max_batches': 0 if args.from_scratch else args.simulation_val_batches,
                                 'simulation_regression_weight': args.simulation_validation_weight},
    }
    write_json(args.output / 'manifest.json', manifest)
    started = time.monotonic()
    try:
        # Instantiate once per explicit split; passing loaders directly avoids
        # generic datamodule setup or accidental train/validation repartition.
        train_dataset = instantiate(cfg.data.dataset.config)
        val_dataset = instantiate(cfg.data.validation_dataset.config)
        for split, dataset in (('train', train_dataset), ('val', val_dataset)):
            if list(dataset.original_dataset.root_filenames) != ids[split]:
                raise ValueError(f'ET loader changed the prepared {split} inventory')
        loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.device != 'cpu', persistent_workers=args.num_workers > 0)
        train_loader = DataLoader(train_dataset, shuffle=True,
                                  collate_fn=collate_fn,
                                  generator=torch.Generator().manual_seed(args.seed), **loader_kwargs)
        et_batches = len(train_loader)
        effective_batches = et_epoch_batch_limit(len(train_dataset), args.batch_size, args.train_batches)
        # CombinedLoader(max_size_cycle) may have a much longer SIM stream.
        # Explicitly cap updates so an epoch always ends after at most one ET
        # pass while the smaller SIM stream can cycle when necessary.
        cfg.trainer.limit_train_batches = effective_batches
        manifest['training_budget'] = {
            'epochs': args.epochs, 'requested_train_batches': args.train_batches,
            'et_batches_per_full_epoch': et_batches,
            'effective_batches_per_epoch': effective_batches,
            'full_et_epoch': effective_batches == et_batches,
            'expected_global_steps': args.epochs * effective_batches,
            'replay_epoch_rule': 'At most one shuffled ET pass; cycle SIM as needed without extending the ET epoch',
        }
        OmegaConf.save(cfg, args.output / 'training_config.yaml', resolve=True)
        manifest['training_config_sha256'] = sha256(args.output / 'training_config.yaml')
        selected_val = (Subset(val_dataset, range(min(len(val_dataset), args.val_batches * args.batch_size)))
                        if args.val_batches else val_dataset)
        val_loader = DataLoader(selected_val, shuffle=False, collate_fn=collate_fn, **loader_kwargs)
        validation_loaders = val_loader
        sim_val_indices = []
        if not args.from_scratch:
            simulation_dataset = instantiate(cfg.simulation_replay.dataset.config)
            simulation_splits = original_simulation_splits(simulation_dataset, base.data)
            split_manifest = {
                'base_config_sha256': sha256(args.base_config),
                'dataset_fingerprint': simulation_dataset.dataset_fingerprint,
                'inventory': [path.name for path in simulation_dataset.simulation_files],
                'split_seed': int(base.data.get('split_seed', 42)),
                'val_size': float(base.data.get('val_size', .1)),
                'test_size': float(base.data.get('test_size', .1)),
                'indices': {name: list(subset.indices) for name, subset in simulation_splits.items()},
                'note': 'Original deterministic CameraTrajectoryDataModule split; test indices are never loaded.',
            }
            write_json(args.output / 'simulation_splits.json', split_manifest)
            manifest['simulation_splits_sha256'] = sha256(args.output / 'simulation_splits.json')
            manifest['simulation_split_counts'] = {name: len(subset) for name, subset in simulation_splits.items()}
            sim_val_indices = simulation_splits['val'].indices[:args.simulation_val_batches * args.batch_size]
            sim_val_loader = DataLoader(Subset(simulation_dataset, sim_val_indices), shuffle=False,
                                        collate_fn=simulation_collate_fn, **loader_kwargs)
            if args.simulation_replay_ratio:
                sim_train_loader = DataLoader(simulation_splits['train'], shuffle=True,
                                               collate_fn=simulation_collate_fn,
                                               generator=torch.Generator().manual_seed(args.seed + 1), **loader_kwargs)
                train_loader = CombinedLoader({'et': train_loader, 'simulation': sim_train_loader}, mode='max_size_cycle')
            validation_loaders = [val_loader, sim_val_loader]
        # Generator initialization is independent of any RNG consumed while
        # constructing datasets or frozen conditioning vocabularies.
        if args.from_scratch:
            L.seed_everything(args.seed, workers=True)
        model = initialize_model(cfg.training.model.module, args.checkpoint, from_scratch=args.from_scratch,
                                 instantiate=instantiate, load_checkpoint=load_checkpoint, device=torch.device('cpu'))
        module = instantiate(cfg.training, model=model, compile_enabled=False, compile_mode='default')
        checkpoint = instantiate(cfg.callbacks.val_checkpoint)
        assert isinstance(checkpoint, ModelCheckpoint)
        logger = instantiate(cfg.logger)
        trainer = instantiate(cfg.trainer, callbacks=[checkpoint], logger=logger)
        # Establish forgetting relative to this exact checkpoint, model behavior,
        # validation subset and metric contract before any optimization occurs.
        initial_selection_score = None
        if not args.from_scratch:
            trainer.validate(module, dataloaders=validation_loaders, verbose=False)
            initial_scores = {name: float(value) for name, value in module._domain_scores.items()}
            initial_selection_score = float(module.et_validation_score)
            if not math.isfinite(initial_selection_score):
                raise RuntimeError('Initial ET selection objective must be finite')
            module.set_reference_simulation_score(initial_scores['simulation'])
            manifest['initial_validation_scores'] = initial_scores
            manifest['conditioning_validation_counts'] = {
                name: state['_conditioning_sample_offset'] for name, state in module._domain_states.items()
            }
        manifest['initial_selection_score'] = initial_selection_score
        manifest['evaluated_validation_counts'] = {'et': len(selected_val), 'simulation': len(sim_val_indices)}
        write_json(args.output / 'manifest.json', manifest)
        # Baseline data workers and diagnostic passes must not change training RNG.
        L.seed_everything(args.seed, workers=True)
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=validation_loaders)
        candidate = checkpoint.best_model_path
        if not candidate or checkpoint.best_model_score is None or not math.isfinite(float(checkpoint.best_model_score)):
            raise RuntimeError('ET training produced no finite validation-selected checkpoint')
        if args.from_scratch:
            best, best_score, selected_initial = candidate, float(checkpoint.best_model_score), False
        else:
            best, best_score, selected_initial = select_checkpoint(
                args.checkpoint, initial_selection_score, candidate, checkpoint.best_model_score,
            )
        inference_cfg.training.model.inference.checkpoint_path = best
        inference_cfg.adaptation.selected_checkpoint = best
        inference_cfg.adaptation.selected_checkpoint_sha256 = sha256(best)
        inference_cfg.adaptation.selected_initial_checkpoint = selected_initial
        OmegaConf.save(inference_cfg, args.output / 'inference_config.yaml', resolve=True)
        status = {
            'exit_code': 0, 'completed_at': datetime.now(timezone.utc).isoformat(),
            'training_mode': manifest['training_mode'], 'initialization': manifest['initialization'],
            'initial_checkpoint': manifest['initial_checkpoint'],
            'elapsed_seconds': time.monotonic() - started,
            'best_checkpoint': best, 'best_checkpoint_sha256': sha256(best),
            'best_validation_score': best_score,
            'selected_initial_checkpoint': selected_initial,
            'initial_selection_score': initial_selection_score,
            'trained_candidate_checkpoint': candidate,
            'trained_candidate_score': float(checkpoint.best_model_score),
            'expected_global_steps': manifest['training_budget']['expected_global_steps'],
            'epochs_completed': trainer.current_epoch, 'global_step': trainer.global_step,
            'inference_config': str(args.output / 'inference_config.yaml'),
            'inference_config_sha256': sha256(args.output / 'inference_config.yaml'),
        }
        if not args.from_scratch:
            # Retain historical receipt fields for existing fine-tuning pipelines.
            status.update(fine_tuned_candidate_checkpoint=candidate,
                          fine_tuned_candidate_score=float(checkpoint.best_model_score))
        write_json(args.output / 'status.json', status)
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except BaseException as error:
        write_json(args.output / 'status.json', {
            'exit_code': 1, 'error': f'{type(error).__name__}: {error}',
            'elapsed_seconds': time.monotonic() - started,
            'completed_at': datetime.now(timezone.utc).isoformat(),
        })
        raise


if __name__ == '__main__':
    raise SystemExit(main())
