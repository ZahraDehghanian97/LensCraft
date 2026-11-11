"""Caption collection needs no fitted camera transform or training split."""
import json

import msgpack
from omegaconf import OmegaConf

import annotate_data
from data.datamodule import CameraTrajectoryDataModule


def test_all_inventory_annotation_does_not_fit_normalization(tmp_path, monkeypatch):
    source = tmp_path / 'simulation'
    source.mkdir()
    (source / 'parameter_dictionary.msgpack').write_bytes(msgpack.packb({
        'keys': ['cinematography__movement__type'], 'values': [['static']],
    }, use_bin_type=True))
    for index in range(2):
        payload = [[[0, 0]], [], [{'i': 'subject', 'c': 'human', 'd': [.5, 1.7, .3],
                                 'm': 'static', 'f': [[0., 0., 0., 0., 0., 0.]] * 4}],
                   [[float(index), 2., 3., 0., 0., 0., 35., 16 / 9]] * 4]
        (source / f'simulation_{index:03d}.msgpack').write_bytes(msgpack.packb(payload, use_bin_type=True))
    cfg = OmegaConf.create({
        'data': {'dataset': {'config': {'_target_': 'data.simulation.dataset.SimulationDataset',
                                      'data_path': str(source), 'normalize': True,
                                      'target_frame_count': 4}}, 'batch_size': 2},
        'output': {name: str(tmp_path / name) for name in ('batch_dir', 'results_dir', 'annotations_dir')},
        'num_batches': 1,
    })

    def make_module(**kwargs):
        kwargs['num_workers'] = 0
        return CameraTrajectoryDataModule(**kwargs)

    monkeypatch.setattr(annotate_data, 'CameraTrajectoryDataModule', make_module)
    annotate_data.main.__wrapped__(cfg)
    tasks = [json.loads(line) for line in
             (tmp_path / 'batch_dir' / 'simulation_dataset_batch_1.jsonl').read_text().splitlines()]
    assert len(tasks) == 2
    assert cfg.data.dataset.config.normalize is True
    assert not (source / 'normalization_parameters.json').exists()
