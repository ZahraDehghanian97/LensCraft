"""The shipped project has one categorical generator and separate frozen judges."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import pytest
import torch

from models.camera_trajectory_model import LensCraft
import utils.load_lens_craft as loader


CONFIG_DIRECTORY = str(Path(__file__).resolve().parents[1] / 'config')
REMOVED = {'conditioning_representation', 'use_merged_memory', 'denormalize_memory',
           'camera_memory_norms', 'caption_presence_conditioning'}
DATASET_REMOVED = {'conditioning_representation', 'embedding_dim', 'clip_embeddings',
                   'fill_none_with_mean', 'clip_model_name'}


def configuration(name='config', overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIRECTORY, version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def test_default_entry_is_the_complete_categorical_contract():
    cfg = configuration()
    module = cfg.training.model.module
    assert module._target_ == 'models.camera_trajectory_model.LensCraft'
    assert (module.latent_dim, module.nhead, module.dim_feedforward) == (256, 8, 1024)
    assert (module.num_encoder_layers, module.num_decoder_layers) == (4, 4)
    assert REMOVED.isdisjoint(module)
    assert set(cfg.training.model.inference) == {'checkpoint_path', 'config'}
    assert 'clip' not in cfg
    assert DATASET_REMOVED.isdisjoint(cfg.data.dataset.config)
    loss = cfg.training.loss_module
    assert loss._target_ == 'training.losses.categorical_loss.CategoricalCameraTrajectoryLoss'
    assert set(loss) == {'_target_', 'losses_list', 'rotation_weight'}
    assert loss.losses_list.categorical == loss.losses_list.cycle == 1
    assert set(loss.losses_list) == {'categorical', 'cycle', 'first_frame', 'relative', 'speed',
                                    'rotation_absolute', 'rotation_raw', 'keyframe_position', 'keyframe_rotation'}
    assert 'use_merged_memory' not in cfg.training


@pytest.mark.parametrize('recipe', ['default', 'multi', 'et_robust', 'transfer_robust',
                                    'transfer_canonical', 'high_level_sparse', 'high_level_scratch',
                                    'high_level_ccdm', 'keyframe_pose'])
def test_every_shipped_training_recipe_uses_the_same_semantic_model_and_objective(recipe):
    overrides = [f'training={recipe}']
    if recipe == 'multi':
        overrides.append('data=multi')
    cfg = configuration(overrides=overrides)
    assert cfg.training.model.module.latent_dim == 256
    assert REMOVED.isdisjoint(cfg.training.model.module)
    assert cfg.training.loss_module._target_ == 'training.losses.categorical_loss.CategoricalCameraTrajectoryLoss'
    assert cfg.training.loss_module.losses_list.categorical == cfg.training.loss_module.losses_list.cycle == 1
    assert 'use_merged_memory' not in cfg.training
    for field in ('caption_missingness', 'domain_augmentation'):
        if field in cfg.training and 'fill' in cfg.training[field]:
            assert cfg.training[field].fill == 'zero'


@pytest.mark.parametrize('dataset', ['default', 'et', 'ccdm', 'multi'])
def test_every_dataset_configuration_is_free_of_generator_clip_factories(dataset):
    overrides = [f'data/dataset={dataset}'] if dataset != 'multi' else ['data=multi', 'training=multi']
    cfg = configuration(overrides=overrides)
    nodes = ([cfg.data.dataset.simulation_config, cfg.data.dataset.ccdm_config]
             if dataset == 'multi' else [cfg.data.dataset.config])
    assert all(DATASET_REMOVED.isdisjoint(node) for node in nodes)
    if dataset == 'multi':
        assert cfg.training._target_ == 'training.multi_dataset_trainer.MultiDatasetTrainer'


@pytest.mark.parametrize('entry', ['config', 'inference', 'test', 'annotator', 'clatr/train', 'tsne'])
def test_top_level_configs_compose(entry):
    configuration(entry)


@pytest.mark.parametrize('baseline', ['et', 'ccdm', 'gendop'])
def test_external_baselines_remain_evaluation_options(baseline):
    cfg = configuration('test', [f'training/model={baseline}'])
    assert cfg.training.model.data_format.type == baseline
    assert cfg.ref_model.module._target_ == 'testing.metrics.semantic_evaluator.FrozenSemanticEvaluator'
    assert cfg.ref_model.module.latent_dim == 512
    assert cfg.ref_model.module.num_query_tokens == 46
    assert 'num_decoder_layers' not in cfg.ref_model.module
    assert REMOVED.isdisjoint(cfg.ref_model.module)


@pytest.mark.parametrize('optimizer', ['adam', 'adamW'])
def test_optimizer_options_compose(optimizer):
    configuration(overrides=[f'training/optimizer={optimizer}'])


@pytest.mark.parametrize('scheduler', ['root_warmup_cosine', 'linear_warmup_cosine'])
def test_scheduler_options_compose(scheduler):
    configuration(overrides=[f'training/lr_scheduler={scheduler}'])


@pytest.mark.parametrize('saved_old_categorical_config', [False, True])
def test_current_checkpoint_loads_strictly_and_saved_categorical_flags_are_migrated(
        tmp_path, saved_old_categorical_config):
    cfg = configuration()
    with open_dict(cfg):
        cfg.training.model.module.latent_dim = 16
        cfg.training.model.module.nhead = 2
        cfg.training.model.module.dim_feedforward = 32
        cfg.training.model.module.num_encoder_layers = 1
        cfg.training.model.module.num_decoder_layers = 1
        cfg.training.model.module.seq_length = 5
        cfg.training.model.module.dropout_rate = 0.0
    original_model = LensCraft(**{key: value for key, value in cfg.training.model.module.items()
                                 if key != '_target_'})
    checkpoint = tmp_path / 'categorical.pt'
    torch.save({'model': original_model.state_dict()}, checkpoint)
    saved_config = None
    if saved_old_categorical_config:
        with open_dict(cfg):
            cfg.training.model.module.conditioning_representation = 'categorical'
            cfg.training.model.module.use_merged_memory = False
            cfg.training.model.module.denormalize_memory = False
            cfg.training.model.module.caption_presence_conditioning = False
            # A recursive Hydra initialization here would fail. Migration must
            # remove it before instantiate rather than load any CLIP asset.
            cfg.training.model.module.camera_memory_norms = {'_target_': 'forbidden.CLIPFactory'}
        saved_config = tmp_path / 'saved.yaml'
        OmegaConf.save(cfg, saved_config)
    caller = configuration().training.model.module if saved_config else cfg.training.model.module
    before = OmegaConf.to_container(caller, resolve=False)
    inference = OmegaConf.create({'checkpoint_path': str(checkpoint),
                                 'config': str(saved_config) if saved_config else None})
    restored = loader.load_lens_craft_model(caller, inference, torch.device('cpu'))
    assert type(restored) is LensCraft and not restored.training
    assert restored.state_dict().keys() == original_model.state_dict().keys()
    assert all(torch.equal(value, restored.state_dict()[key])
               for key, value in original_model.state_dict().items())
    assert OmegaConf.to_container(caller, resolve=False) == before


def test_legacy_generator_fails_before_model_construction(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'old512.pt'
    torch.save({'model': {'encoder.query_tokens': torch.zeros(46, 1, 512)}}, checkpoint)

    def forbidden_construct(*args, **kwargs):
        pytest.fail('Legacy weights must fail before model construction')

    monkeypatch.setattr(loader, 'instantiate', forbidden_construct)
    inference = OmegaConf.create({'checkpoint_path': str(checkpoint), 'config': None})
    with pytest.raises(ValueError, match='no longer supported'):
        loader.load_lens_craft_model(configuration().training.model.module, inference, torch.device('cpu'))
