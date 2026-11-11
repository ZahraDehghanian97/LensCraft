"""Metadata-only conditioning must preserve geometry and avoid CLIP setup."""
from __future__ import annotations

import builtins
from contextlib import nullcontext
import inspect
import json
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch

from data.ccdm.dataset import CCDMDataset, collate_fn as ccdm_collate
from data.simulation.constants import cinematography_struct, simulation_struct
from data.simulation.dataset import SimulationDataset, collate_fn as simulation_collate


def null_struct(struct):
    return {name: null_struct(kind) if isinstance(kind, list) else None for name, kind in struct}


@pytest.fixture
def no_clip_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"models.clip_embeddings", "data.simulation.init_embeddings",
                    "data.simulation.memory_norms", "testing.metrics.semantic_vocabulary",
                    "testing.metrics.semantic_evaluator"}:
            raise AssertionError("Dataset conditioning imported a CLIP dependency: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def simulation_export(tmp_path):
    root = tmp_path / 'simulation'
    root.mkdir()
    dictionary = {
        'keys': ['cinematography__initial__cameraAngle', 'cinematography__initial__shotSize',
                 'cinematography__movement__type', 'cinematography__final__cameraAngle'],
        'values': [['low'], ['closeUp'], ['nativePan'], ['high']],
    }
    camera = [[float(index), 2., 3., 0., .1, 0., 35., 16 / 9] for index in range(4)]
    subject = [[0., 1., 0., 0., 0., 0.] for _ in range(4)]
    payload = [[[index, 0] for index in range(4)], [],
               [{'i': 'one', 'c': 'human', 'd': [.5, 1.7, .3], 'm': 'static', 'f': subject}], camera]
    (root / 'manifest.json').write_text(json.dumps({'config': {'subjectCount': 1, 'instructionCount': 1}}))
    (root / 'parameter_dictionary.msgpack').write_bytes(msgpack.packb(dictionary, use_bin_type=True))
    path = root / 'simulation_000.msgpack'
    path.write_bytes(msgpack.packb(payload, use_bin_type=True))
    return root, path, payload


def test_simulation_categorical_path_never_loads_clip_and_keeps_geometry(simulation_export, no_clip_dependencies):
    root, path, payload = simulation_export
    dataset = SimulationDataset(str(root), normalize=False, target_frame_count=4)
    item = dataset[0]
    assert item['cinematography_ids'].dtype == torch.long
    assert item['cinematography_ids'].tolist() == [0, 1, -1, -1, 19, -1, 2, -1, -1, -1]
    assert item['prompt_none_mask'].shape == (10,)
    assert item['prompt_none_mask'].sum() == 4
    assert 'cinematography_prompt' not in item and 'simulation_instruction' not in item
    assert all(entry[3] is None for entry in item['cinematography_prompt_parameters'])
    assert len(item['simulation_instruction_parameters']) == 36
    assert all(entry[3] is None for entry in item['simulation_instruction_parameters'])
    assert item['structured_presence_mask'].shape == (46,)
    assert torch.equal(item['camera_trajectory'], torch.tensor(payload[3])[:, :6])
    assert torch.equal(item['subject_trajectory'], torch.tensor(payload[2][0]['f']))

    # Annotation IDs cannot depend on reference-camera values.
    payload[3][0][0] += 100.
    path.write_bytes(msgpack.packb(payload, use_bin_type=True))
    changed = dataset[0]
    assert torch.equal(changed['cinematography_ids'], item['cinematography_ids'])
    assert not torch.equal(changed['camera_trajectory'], item['camera_trajectory'])


def test_unknown_simulation_semantics_remain_missing(simulation_export):
    root, _, _ = simulation_export
    dictionary_path = root / 'parameter_dictionary.msgpack'
    dictionary = msgpack.unpackb(dictionary_path.read_bytes(), raw=False)
    dictionary['values'][2] = ['unrecognized-camera-motion']
    dictionary_path.write_bytes(msgpack.packb(dictionary, use_bin_type=True))
    dataset = SimulationDataset(str(root), normalize=False, target_frame_count=4)
    item = dataset[0]
    assert item['cinematography_ids'][4] == -1
    assert not item['prompt_none_mask'][4]


def test_categorical_ids_respect_declared_observation_availability(simulation_export):
    root, _, _ = simulation_export
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['modelObservation'] = {
        'profile': 'et_compatible_v1', 'subjectOrientation': 'unavailable',
        'subjectDimensions': 'default', 'defaultSubjectDimensions': [.5, 1.7, .3],
        'simulationInstructions': 'unavailable', 'captionPolicy': 'motion_only',
    }
    manifest_path.write_text(json.dumps(manifest))
    dataset = SimulationDataset(str(root), normalize=False, target_frame_count=4)
    item = dataset[0]
    assert item['raw_prompt']['initial']['cameraAngle'] == 'low'
    assert item['cinematography_ids'].tolist() == [-1, -1, -1, -1, 19, -1, -1, -1, -1, -1]
    assert item['prompt_none_mask'].sum() == 1
    assert item['geometry_reference']['camera_trajectory'].shape == (4, 6)


def test_compaction_preserves_ten_ids_after_dropping_annotation_tuples(simulation_export):
    from data.datamodule import compact_simulation_collate_fn

    root, _, _ = simulation_export
    dataset = SimulationDataset(str(root), normalize=False, target_frame_count=4)
    item = dataset[0]
    regular = simulation_collate([item, item])
    compact = compact_simulation_collate_fn([item, item])
    assert compact['cinematography_ids'].shape == (2, 10)
    assert compact['prompt_none_mask'].shape == (2, 10)
    assert torch.equal(compact['cinematography_ids'], regular['cinematography_ids'])
    assert torch.equal(compact['prompt_none_mask'], compact['cinematography_ids'] >= 0)
    assert 'cinematography_prompt_parameters' not in compact
    assert 'cinematography_prompt' not in compact


def test_default_datamodule_needs_only_categorical_data_config(simulation_export, no_clip_dependencies):
    from data.datamodule import CameraTrajectoryDataModule
    from omegaconf import OmegaConf

    root, _, _ = simulation_export
    configuration = OmegaConf.create({
        '_target_': 'data.simulation.dataset.SimulationDataset', 'data_path': str(root),
        'normalize': False, 'target_frame_count': 4,
    })
    module = CameraTrajectoryDataModule(configuration, batch_size=1, num_workers=0,
                                         val_size=0., test_size=0., compact_batches=True)
    module.setup()
    batch = next(iter(module.train_dataloader()))
    assert batch['cinematography_ids'].shape == (1, 10)
    assert 'cinematography_prompt_parameters' not in batch
    assert batch['structured_presence_mask'].shape == (1, 46)


@pytest.fixture
def ccdm_export(tmp_path, monkeypatch):
    monkeypatch.setattr(CCDMDataset, '_normalization_parameters', None)
    np.save(tmp_path / 'Mean_Std.npy', {'Mean': np.zeros(5), 'Std': np.ones(5)})
    np.save(tmp_path / 'data.npy', {'cam': [np.zeros((3, 5)), np.ones((4, 5))],
                                     'info': [['native pan caption'], ['native orbit caption']]})
    rows = []
    for index, value in enumerate(('nativePan', 'nativeOrbit')):
        prompt = null_struct(cinematography_struct)
        prompt['movement']['type'] = value
        rows.append({'custom_id': f'ccdm-{index:06d}',
                     'source_prompt': ('native pan caption', 'native orbit caption')[index],
                     'cinematographyPrompts': [prompt],
                     'simulationInstructions': [null_struct(simulation_struct)]})
    (tmp_path / 'annotations.json').write_text(json.dumps(rows))
    (tmp_path / 'indices.json').write_text(json.dumps([1, 0]))
    return tmp_path


def test_ccdm_native_ids_and_cohort_order_survive_without_clip(ccdm_export, no_clip_dependencies):
    before = (ccdm_export / 'annotations.json').read_bytes()
    dataset = CCDMDataset(str(ccdm_export / 'data.npy'), device='cpu', normalize=False,
        original_seq_len=5, annotations_path=str(ccdm_export / 'annotations.json'),
        sample_indices_path=str(ccdm_export / 'indices.json'))
    batch = ccdm_collate([dataset[0], dataset[1]])
    assert batch['item_ids'] == ['ccdm-000001', 'ccdm-000000']
    assert batch['text_prompts'] == ['native orbit caption', 'native pan caption']
    assert batch['cinematography_ids'][:, 4].tolist() == [20, 19]
    assert batch['prompt_none_mask'].sum(1).tolist() == [1, 1]
    assert (~batch['padding_mask']).sum(1).tolist() == [4, 3]
    assert 'caption_feat' not in batch and 'cinematography_prompt' not in batch
    assert (ccdm_export / 'annotations.json').read_bytes() == before


def test_unannotated_categorical_ccdm_exposes_unknown_ids_without_guessing(ccdm_export):
    dataset = CCDMDataset(str(ccdm_export / 'data.npy'), device='cpu', normalize=False)
    item = dataset[0]
    assert item['cinematography_ids'].tolist() == [-1] * 10
    assert not item['prompt_none_mask'].any()
    assert 'caption_feat' not in item


def test_et_categorical_loader_needs_no_stored_clip_features(tmp_path, monkeypatch):
    from data.et import dataset as module

    item = {'traj_filename': 'sample.npz', 'intrinsics': [35., 16 / 9],
            'traj_feat': torch.arange(36, dtype=torch.float32).reshape(9, 4),
            'char_feat': torch.ones(3, 4), 'padding_mask': torch.tensor([True] * 4),
            'caption_raw': {'caption': 'Native orbit description'}}
    monkeypatch.setattr(module, 'load_et_dataset', lambda *args: [item])
    prompt = null_struct(cinematography_struct)
    prompt['movement']['type'] = 'nativeOrbit'
    path = tmp_path / 'et_annotations.json'
    path.write_text(json.dumps([{'custom_id': 'et-sample', 'cinematographyPrompts': [prompt],
                                 'simulationInstructions': [null_struct(simulation_struct)]}]))
    dataset = module.ETDataset('unused', 'unused', str(path), 'mixed', 'test', True)
    result = dataset[0]
    batch = module.collate_fn([result, result])
    assert batch['cinematography_ids'][:, 4].tolist() == [20, 20]
    assert torch.equal(result['camera_trajectory'], item['traj_feat'].permute(1, 0))
    assert torch.equal(result['subject_trajectory'], item['char_feat'].permute(1, 0))
    assert 'caption_feat' not in batch and 'cinematography_prompt' not in batch


def test_native_et_loader_never_instantiates_cached_caption_modality(tmp_path, monkeypatch):
    from data.et import load as module
    from omegaconf import OmegaConf

    captions = tmp_path / 'caption'
    captions.mkdir()
    (captions / 'one.txt').write_text('A camera moves to the left.\n')
    configuration = OmegaConf.create({'dataset': {
        '_target_': 'unused.GeometryDataset', 'name': '${dataset.caption.name}',
        'num_cond_feats': [3, 512], 'char': {'num_feats': 3},
        'caption': {'_target_': 'forbidden.CaptionCache', 'name': 'caption',
                    'raw_caption_dir': str(captions)},
    }})

    class NativeGeometry:
        root_filenames = ['one']

        def set_split(self, split):
            assert split == 'test'
            return self

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {'traj_filename': 'one.txt', 'traj_feat': torch.zeros(9, 4),
                    'char_feat': torch.zeros(3, 4)}

    def instantiate_geometry(configuration):
        assert 'caption' not in configuration
        assert configuration['num_cond_feats'] == [3]
        assert configuration['name'] == 'caption'
        return NativeGeometry()

    monkeypatch.setattr(module, 'load_et_config', lambda *args: configuration)
    monkeypatch.setattr(module, 'instantiate', instantiate_geometry)
    monkeypatch.setattr(module, 'ModuleImporter', SimpleNamespace(
        temporary_module=lambda *args: nullcontext()))
    native = module.load_et_dataset('unused/configs', 'unused', 'mixed', 'test')
    assert len(native) == 1 and native.root_filenames == ['one']
    assert native[0]['caption_raw'] == {'caption': 'A camera moves to the left.\n'}
    assert 'caption_feat' not in native[0]


def test_dataset_api_exposes_no_embedding_or_representation_mode_arguments():
    from data.et.dataset import ETDataset

    removed = {'conditioning_representation', 'clip_embeddings', 'embedding_dim',
               'fill_none_with_mean', 'clip_model_name'}
    for dataset in (SimulationDataset, CCDMDataset, ETDataset):
        assert not removed.intersection(inspect.signature(dataset).parameters)
