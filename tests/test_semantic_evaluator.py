"""The fixed semantic judge stays independent of categorical generation."""
import hashlib
import importlib
from pathlib import Path
import pickle
import sys
from types import ModuleType

import pytest
import torch
from omegaconf import OmegaConf

from data.categorical_conditioning import metadata_from_prompt
from data.simulation.caption import enum_descriptions
from testing import evaluator as evaluator_loader
from testing.metrics.semantic_evaluator import (
    FROZEN_ENCODER_SOURCE_SHA256,
    FROZEN_POSITION_SOURCE_SHA256,
    FrozenSemanticEvaluator,
    _METRIC_FIELDS,
    metric_conditioning_from_batch,
)
from testing.metrics import semantic_vocabulary


def tiny_judge():
    return FrozenSemanticEvaluator(latent_dim=8, nhead=2, num_encoder_layers=1,
                                   dim_feedforward=16, seq_length=3)


def metric_vocab(width=8):
    result = {}
    for family, descriptions in enum_descriptions.items():
        result[family] = {raw: torch.arange(width, dtype=torch.float32) + index + 1
                          for index, raw in enumerate(descriptions)}
    result['boolean'] = {True: torch.ones(width), False: -torch.ones(width)}
    return result


def historical_payload(model):
    state = {'model.' + name: value.clone() for name, value in model.state_dict().items()}
    state['model.decoder.unused_weight'] = torch.randn(17, 3)
    return {'state_dict': state}


def test_strict_encoder_subset_is_frozen_and_cannot_generate():
    original = tiny_judge()
    payload = historical_payload(original)
    # These aliases occur in older fixed checkpoints.
    state = payload['state_dict']
    state['model.subject_projection_loc_rot.weight'] = state.pop('model.subject_trajectory_projection.weight')
    state['model.subject_projection_loc_rot.bias'] = state.pop('model.subject_trajectory_projection.bias')
    loaded = tiny_judge().load_metric_checkpoint(payload)
    camera, subject, volume = torch.randn(2, 3, 6), torch.randn(2, 3, 6), torch.randn(2, 1, 3)
    mask = torch.tensor([[False, False, True], [False, True, True]])
    expected = original.embed_trajectory(camera, subject, volume, mask)
    actual = loaded.embed_trajectory(camera, subject, volume, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (46, 2, 8) and not actual.requires_grad
    loaded.train()
    assert not loaded.training and all(not module.training for module in loaded.modules())
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.memory_tokens_count == loaded.num_query_tokens == 46
    assert loaded.decoder.seq_length == 3 and not isinstance(loaded.decoder, torch.nn.Module)
    assert not hasattr(loaded, 'generate_camera_trajectory')
    assert not hasattr(loaded, 'decode')
    assert loaded.checkpoint_load_info['ignored_nonencoder_tensor_count'] == 1
    del state['model.encoder.query_tokens']
    with pytest.raises(RuntimeError, match='query_tokens'):
        tiny_judge().load_metric_checkpoint(payload)


def test_encoder_matches_pinned_historical_source(monkeypatch):
    source = (Path(__file__).resolve().parents[2] / 'performance_reports' /
              'ccdm_review_20260918' / 'execution' / 'src' / 'models')
    if not source.is_dir():
        pytest.skip('Historical read-only source snapshot is not in this deployment')
    assert hashlib.sha256((source / 'encoder.py').read_bytes()).hexdigest() == FROZEN_ENCODER_SOURCE_SHA256
    assert hashlib.sha256((source / 'positional_encoding.py').read_bytes()).hexdigest() == FROZEN_POSITION_SOURCE_SHA256
    package_name = '_lenscraft_pinned_metric_reference'
    package = ModuleType(package_name)
    package.__path__ = [str(source)]
    monkeypatch.setitem(sys.modules, package_name, package)
    reference_class = importlib.import_module(package_name + '.encoder').Encoder
    reference = reference_class(6, 8, 2, 1, 16, .3, 46).eval()
    model = tiny_judge()
    model.encoder.load_state_dict(reference.state_dict(), strict=True)
    camera = torch.randn(2, 3, 6)
    embedded_subject = torch.randn(2, 4, 8)
    for camera_mask, subject_mask in (
        (None, None),
        (torch.tensor([[False, False, True], [False, True, True]]), None),
        (None, torch.tensor([[False, True, False], [True, False, True]])),
    ):
        with torch.no_grad():
            expected = reference(camera, embedded_subject, camera_mask, subject_mask)
            actual = model.encoder(camera, embedded_subject, camera_mask, subject_mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_metric_targets_use_full_raw_metadata_and_independent_presence():
    sample = metadata_from_prompt(
        {'initial': {'cameraAngle': 'low'}, 'movement': {'type': 'nativeOrbit'}},
        {'constraints': {'staticDistance': False, 'maxSpeed': 4.0}},
    )
    batch = {key: (value.unsqueeze(0) if torch.is_tensor(value) else [value])
             for key, value in sample.items()}
    # A generator's semantic probabilities and obsolete tuple vectors cannot
    # supply frozen metric targets.
    batch['caption_raw'] = torch.randn(10, 1, 21)
    batch['cinematography_prompt_parameters'][0] = [
        (*entry[:3], torch.full((21,), float('nan')))
        for entry in batch['cinematography_prompt_parameters'][0]
    ]
    vocabulary = metric_vocab()
    target, present = metric_conditioning_from_batch(batch, vocabulary)
    fields = [name for name, _ in _METRIC_FIELDS]
    angle, movement = fields.index('initial_cameraAngle'), fields.index('movement_type')
    boolean, numeric = fields.index('constraints_staticDistance'), fields.index('constraints_maxSpeed')
    assert target.shape == (46, 1, 8) and present.shape == (1, 46)
    assert present.sum() == 3 and not present[0, movement]
    torch.testing.assert_close(target[angle, 0], vocabulary['CameraVerticalAngle']['low'])
    torch.testing.assert_close(target[boolean, 0], vocabulary['boolean'][False])
    phase = torch.arange(1, 5) * (torch.log1p(torch.tensor(4.)) / torch.log1p(torch.tensor(20.)) * torch.pi)
    torch.testing.assert_close(target[numeric, 0], torch.stack((phase.sin(), phase.cos()), -1).flatten())
    batch['prompt_none_mask'][0, angle] = False
    batch['structured_presence_mask'][0, numeric] = False
    masked_target, masked_presence = metric_conditioning_from_batch(batch, vocabulary)
    assert masked_presence.sum() == 1 and masked_presence[0, boolean]
    assert torch.isfinite(masked_target).all() and not masked_target[~masked_presence.T].any()
    # Native support is explicit in a metric vocabulary; no directional class
    # is substituted when that extension is absent.
    vocabulary['CameraMovementType']['nativeOrbit'] = torch.full((8,), 17.)
    _, extended_presence = metric_conditioning_from_batch(batch, vocabulary)
    assert extended_presence[0, movement]


def test_loader_uses_metric_encoder_instead_of_saved_generator_target(tmp_path, monkeypatch):
    original = tiny_judge()
    checkpoint = tmp_path / 'semantic.ckpt'
    torch.save(historical_payload(original), checkpoint)
    definition = {'_target_': 'models.camera_trajectory_model.LensCraft',
                  'latent_dim': 8, 'nhead': 2, 'num_encoder_layers': 1,
                  'dim_feedforward': 16, 'seq_length': 3,
                  'num_decoder_layers': 99, 'use_merged_memory': True}
    saved_config = tmp_path / 'historical.yaml'
    clip = {'model_name': 'clip-vit-base-patch32', 'latent_dim': 8}
    OmegaConf.save(OmegaConf.create({'training': {'model': {'module': definition}},
                                   'clip': clip}), saved_config)
    config = OmegaConf.create({'module': definition, 'clip': clip, 'inference': {
        'checkpoint_path': str(checkpoint), 'config': str(saved_config)}})
    vocabulary = metric_vocab()
    monkeypatch.setattr(evaluator_loader, 'load_metric_clip_embeddings', lambda **_: vocabulary)
    monkeypatch.setattr(evaluator_loader.hydra.utils, 'instantiate',
                        lambda *_, **__: pytest.fail('The saved generator target must never be instantiated'))
    model = evaluator_loader.load_semantic_evaluator(config, 'cpu', input_clip=OmegaConf.create(clip))
    assert isinstance(model, FrozenSemanticEvaluator) and model.clip_embeddings is vocabulary
    assert model.evaluation_provenance['model_definition'] == definition
    implementation = model.evaluation_provenance['implementation']
    assert implementation['generation_supported'] is False
    assert implementation['checkpoint_load']['strict_encoder_and_projection_load'] is True
    for name, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_valid_metric_cache_never_initializes_clip_or_creates_generator_statistics(tmp_path, monkeypatch):
    cache = tmp_path / 'fixed_metric.pkl'
    vocabulary = metric_vocab()
    with cache.open('wb') as stream:
        pickle.dump(vocabulary, stream)
    semantic_vocabulary._write_embedding_cache_metadata(str(cache), 'clip-vit-base-patch32', 8)
    monkeypatch.setattr(semantic_vocabulary, '_create_clip_embedder',
                        lambda *_, **__: pytest.fail('A valid frozen cache must not load CLIP'))
    actual = semantic_vocabulary.load_metric_clip_embeddings(
        clip_model_name='clip-vit-base-patch32', cache_file=str(cache), embedding_dimension=8)
    for family, values in vocabulary.items():
        for raw, expected in values.items():
            torch.testing.assert_close(actual[family][raw], expected, rtol=0, atol=0)
    assert {path.name for path in tmp_path.iterdir()} == {'fixed_metric.pkl', 'fixed_metric.pkl.meta.json'}
