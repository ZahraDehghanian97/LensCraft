"""Single-field native motion interface and sparse-caption invariants."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from training.robust_scratch import native_motion_tokens as native


class PackagedNativeMotionTests(unittest.TestCase):
    def test_frozen_parser_recognizes_complete_text_and_marks_unknown_sentences(self):
        bridge = native._load_bridge()
        for sentence, family in (
            ('The camera pans to the character.', 'pan'),
            ('The camera rotates around the character.', 'orbit'),
        ):
            _, parsed = bridge.parse_with_provenance(sentence)
            self.assertEqual(parsed['motion_families'], [family])
            self.assertTrue(parsed['all_sentences_recognized'])
        _, parsed = bridge.parse_with_provenance(
            'The camera rotates around the character. Then it teleports.'
        )
        self.assertFalse(parsed['all_sentences_recognized'])
        self.assertIn('then it teleports', parsed['unknown_sentences'])

    def test_packaged_artifacts_preserve_frozen_bytes_and_vector_identity(self):
        artifacts = (
            (native.NATIVE_PAN_EMBEDDING_PATH,
             'cf5eb09b03ba40c78eb96831026a42355a37d6974f0f801727e74bc3c0dc3cef'),
            (native.NATIVE_ORBIT_EMBEDDING_PATH,
             '34f1eb9e3bb865419502746aa5f1bafe2bfb1690f1adfa7d9c77882bda2970f3'),
        )
        for path, expected_digest in artifacts:
            with self.subTest(artifact=path.name):
                self.assertEqual(native.file_sha256(path), expected_digest)
                payload = json.loads(path.read_text())
                vector = payload['embedding']
                self.assertEqual(len(vector), 512)
                self.assertTrue(all(math.isfinite(value) for value in vector))
                self.assertEqual(hashlib.sha256(struct.pack('<512f', *vector)).hexdigest(),
                                 payload['embedding_sha256'])
                self.assertEqual(payload['base_vocabulary']['sha256'],
                                 native.BASE_VOCABULARY_SHA256)


def vocabulary():
    from data.simulation.constants import NumericFeature
    from data.simulation.utils import CLIP_PARAMETERS
    result = {}
    for i, (_, kind) in enumerate(CLIP_PARAMETERS):
        if isinstance(kind, NumericFeature):
            continue
        key = 'boolean' if kind is bool else kind.__name__
        values = (False, True) if kind is bool else [member.value for member in kind]
        if key not in result:
            result[key] = {value: torch.arange(512).float()*.001+i+j for j, value in enumerate(values)}
    return result


def make_batch(texts, *, batch_first=False):
    from data.simulation.utils import CLIP_PARAMETERS, cinematography_struct_parameters
    fields = tuple(name for name, _ in CLIP_PARAMETERS)
    split, size = len(cinematography_struct_parameters), len(texts)
    values = torch.arange(46*size*512).float().reshape(46, size, 512)/1000
    parameters = [[(field, 'preserved-original', 0, values[i, row].clone())
                   for i, field in enumerate(fields)] for row in range(size)]
    return {'text_prompts': list(texts), 'item_ids': [f'item-{i:06d}' for i in range(size)],
            'cinematography_prompt': values[:split].transpose(0, 1) if batch_first else values[:split],
            'simulation_instruction': values[split:].transpose(0, 1) if batch_first else values[split:],
            'cinematography_prompt_parameters': [row[:split] for row in parameters],
            'simulation_instruction_parameters': [row[split:] for row in parameters],
            'prompt_none_mask': torch.ones(size, 46, dtype=torch.bool),
            'raw_prompt': [{'initial': {'shotSize': 'fullShot'},
                            'movement': {'type': 'panLeft', 'speed': 'constant'},
                            'final': {'shotSize': 'mediumShot'}} for _ in range(size)],
            'raw_instruction': [{'dynamic': {'easing': 'easeInSine'}, 'private-extra': [1, 2]} for _ in range(size)],
            'camera_trajectory': object(), 'subject_trajectory': object(), 'subject_volume': object(),
            'opaque_payload': object(), 'domain_augmentation_view': torch.full((size,), 2)}


def tokens(batch):
    first, second = batch['cinematography_prompt'], batch['simulation_instruction']
    size = len(batch['prompt_none_mask'])
    if first.shape[0] == size and first.shape[1] == 10:
        first, second = first.transpose(0, 1), second.transpose(0, 1)
    return torch.cat((first, second), 0)


@unittest.skipIf(torch is None, 'Torch is required for actual native feature contracts')
class NativeMotionTokenTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = vocabulary()
        self.pan = torch.arange(512).float()/512
        self.orbit = torch.arange(512).float()/1024+1
        self.adapter = native.NativeMotionTokens(self.vocabulary, self.pan, self.orbit)
        self.fields = self.adapter.fields
        self.move = self.fields.index('movement_type')

    def assert_preserved(self, before, after, changed_rows):
        old_tokens, new_tokens = tokens(before), tokens(after)
        old_params = [a+b for a, b in zip(before['cinematography_prompt_parameters'], before['simulation_instruction_parameters'])]
        new_params = [a+b for a, b in zip(after['cinematography_prompt_parameters'], after['simulation_instruction_parameters'])]
        for field in range(46):
            for row in range(len(old_params)):
                if field == self.move and row in changed_rows:
                    continue
                self.assertTrue(torch.equal(old_tokens[field, row], new_tokens[field, row]), (field, row))
                self.assertEqual(bool(before['prompt_none_mask'][row, field]), bool(after['prompt_none_mask'][row, field]))
                self.assertEqual(old_params[row][field][:3], new_params[row][field][:3])
                self.assertTrue(torch.equal(old_params[row][field][3], new_params[row][field][3]))
        for key in ('camera_trajectory', 'subject_trajectory', 'subject_volume', 'opaque_payload', 'raw_instruction'):
            self.assertIs(before[key], after[key], key)
        self.assertIs(before['text_prompts'], after['text_prompts'])
        self.assertIs(before['item_ids'], after['item_ids'])
        for old, new in zip(before['raw_prompt'], after['raw_prompt']):
            self.assertEqual(old['initial'], new['initial'])
            self.assertEqual(old['final'], new['final'])
            self.assertEqual(old['movement']['speed'], new['movement']['speed'])

    def test_text_only_pan_and_orbit_change_only_one_field(self):
        batch = make_batch(['The camera pans to the character. The camera shoots in medium shot.',
                            'The camera rotates around the character. The camera moves in clockwise direction.',
                            'The camera pulls out.'])
        before = tokens(batch).clone()
        raw = deepcopy(batch['raw_prompt'])
        result, provenance = self.adapter.adapt_native_batch(batch)
        self.assertEqual([row['changed'] for row in provenance], [True, True, False])
        self.assertEqual([result['cinematography_prompt_parameters'][row][self.move][1]
                          for row in (0, 1)], ['nativePan', 'nativeOrbit'])
        self.assertEqual(result['cinematography_prompt_parameters'][1][self.move][2], 20)
        self.assertTrue(torch.equal(tokens(result)[self.move, 0], self.pan))
        self.assertTrue(torch.equal(tokens(result)[self.move, 1], self.orbit))
        self.assert_preserved(batch, result, {0, 1})
        self.assertTrue(torch.equal(tokens(batch), before))
        self.assertEqual(batch['raw_prompt'], raw)

    def test_unsupported_and_conflicting_captions_do_not_relabel_or_clear_fields(self):
        batch = make_batch(['The camera pans to the character. The camera pulls out.',
                            'The camera rotates around the character. Then it teleports.',
                            'A completely unsupported description.'])
        result, provenance = self.adapter.adapt_native_batch(batch)
        self.assertFalse(any(row['changed'] for row in provenance))
        self.assertTrue(all(row['native_family'] == 'unknown' for row in provenance))
        self.assert_preserved(batch, result, set())

    def test_batch_first_axes_retained(self):
        batch = make_batch(['The camera pans to the character.', 'The camera rotates around the character.'], batch_first=True)
        result, _ = self.adapter.adapt_native_batch(batch)
        self.assertEqual(result['cinematography_prompt'].shape, (2, 10, 512))
        self.assertEqual(result['simulation_instruction'].shape, (2, 36, 512))
        self.assert_preserved(batch, result, {0, 1})

    def test_training_orbit_requires_ccdm_and_known_nonconflicting_metadata(self):
        batch = make_batch(['training']*4)
        split = self.adapter.split
        mode, dynamic = self.fields.index('dynamic_movementMode'), self.fields.index('dynamic_type')
        for row in range(4):
            all_params = batch['cinematography_prompt_parameters'][row]+batch['simulation_instruction_parameters'][row]
            all_params[mode] = (self.fields[mode], 'arc', 2, tokens(batch)[mode, row])
            all_params[dynamic] = (self.fields[dynamic], 'simple', 0, tokens(batch)[dynamic, row])
            all_params[self.move] = (self.fields[self.move], None, -1, None)
            batch['prompt_none_mask'][row, self.move] = False
            if row == 2:
                all_params[self.move] = (self.fields[self.move], 'arcLeft', 13, tokens(batch)[self.move, row])
                batch['prompt_none_mask'][row, self.move] = True
            batch['cinematography_prompt_parameters'][row] = all_params[:split]
            batch['simulation_instruction_parameters'][row] = all_params[split:]
        batch['domain_augmentation_view'][1] = 0
        batch['prompt_none_mask'][3, mode] = False
        result = self.adapter.augment_training_batch(batch)
        self.assertEqual(result['native_motion_tokens']['changed_rows'], 1)
        self.assertEqual(result['cinematography_prompt_parameters'][0][self.move][1], 'nativeOrbit')
        self.assertFalse(bool(batch['prompt_none_mask'][0, self.move]))
        self.assertTrue(torch.equal(tokens(result)[self.move, 0], self.orbit))
        # Metadata entries which are unknown legitimately hold no vector.
        for row in (1, 2, 3):
            self.assertEqual(result['cinematography_prompt_parameters'][row][self.move][:3],
                             batch['cinematography_prompt_parameters'][row][self.move][:3])
        for field in range(46):
            if field != self.move:
                self.assertTrue(torch.equal(tokens(result)[field], tokens(batch)[field]))
        self.assertIs(result['camera_trajectory'], batch['camera_trajectory'])

    def test_native_orbit_survives_actual_high_level_only_missingness(self):
        from training.caption_missingness import CaptionMissingness
        batch = make_batch(['The camera rotates around the character.'])
        result, _ = self.adapter.adapt_native_batch(batch)
        missingness = CaptionMissingness({'enabled': True, 'profile': 'high_level_only',
                                         'probability': 1., 'preserve_movement_type': True}, self.vocabulary)
        sparse, present = missingness.apply(tokens(result), result['prompt_none_mask'], sample_seeds=[81], force=True)
        self.assertTrue(bool(present[0, self.move]))
        self.assertTrue(torch.equal(sparse[self.move, 0], self.orbit))
        self.assertFalse(bool(present[0, self.fields.index('dynamic_movementMode')]))

    def test_certified_training_orbit_accepts_neutral_synthetic_raw_prompt(self):
        # The safe procedural generator stores empty raw prompt dictionaries;
        # the old semantic/motion augmentation adds only structured metadata.
        batch = make_batch(['training'])
        batch['raw_prompt'] = [{}]
        split = self.adapter.split
        params = batch['cinematography_prompt_parameters'][0]+batch['simulation_instruction_parameters'][0]
        for field, value, index in (('dynamic_movementMode', 'arc', 2), ('dynamic_type', 'simple', 0)):
            number = self.fields.index(field)
            params[number] = (field, value, index, tokens(batch)[number, 0])
        params[self.move] = ('movement_type', None, -1, None)
        batch['prompt_none_mask'][0, self.move] = False
        batch['cinematography_prompt_parameters'][0] = params[:split]
        batch['simulation_instruction_parameters'][0] = params[split:]
        result = self.adapter.augment_training_batch(batch)
        self.assertEqual(result['native_motion_tokens']['changed_rows'], 1)
        self.assertEqual(result['raw_prompt'], [{}])
        self.assertEqual(batch['raw_prompt'], [{}])
        self.assertEqual(result['cinematography_prompt_parameters'][0][self.move][1], 'nativeOrbit')
        for field in range(46):
            if field != self.move:
                self.assertTrue(torch.equal(tokens(result)[field], tokens(batch)[field]))
                self.assertEqual(bool(result['prompt_none_mask'][0, field]), bool(batch['prompt_none_mask'][0, field]))
        self.assertIs(result['camera_trajectory'], batch['camera_trajectory'])

    def test_base_vocabulary_and_core_enum_unchanged_and_native_vectors_distinct(self):
        from data.simulation.constants import CameraMovementType
        before = deepcopy(self.vocabulary)
        self.adapter.adapt_native_batch(make_batch(['The camera pans to the character.']))
        self.assertEqual(len(CameraMovementType), 19)
        self.assertEqual((native.NATIVE_PAN_INDEX, native.NATIVE_ORBIT_INDEX), (19, 20))
        for key, values in before.items():
            self.assertEqual(set(values), set(self.vocabulary[key]))
            for value, vector in values.items():
                self.assertTrue(torch.equal(vector, self.vocabulary[key][value]))
        with self.assertRaises(ValueError):
            native.NativeMotionTokens(self.vocabulary, self.pan, self.pan)
        bad = deepcopy(self.vocabulary)
        bad['CameraMovementType']['nativeOrbit'] = self.orbit
        with self.assertRaises(ValueError):
            native.NativeMotionTokens(bad, self.pan, self.orbit)

    def test_source_ids_required_and_target_fields_never_needed(self):
        batch = make_batch(['The camera pans to the character.'])
        for key in ('camera_trajectory', 'subject_trajectory', 'subject_volume'):
            del batch[key]
        result, _ = self.adapter.adapt_native_batch(batch)
        self.assertTrue(torch.equal(tokens(result)[self.move, 0], self.pan))
        batch['item_ids'] = []
        with self.assertRaises(ValueError):
            self.adapter.adapt_native_batch(batch)

    def artifact(self):
        return {'protocol': native.ORBIT_VOCABULARY_PROTOCOL, 'field': 'movement_type',
                'value': native.NATIVE_ORBIT_VALUE, 'value_index': native.NATIVE_ORBIT_INDEX,
                'text': native.NATIVE_ORBIT_TEXT, 'embedding_width': 512,
                'representation': 'CLIPTextModel.pooler_output', 'model_name': 'openai/clip-vit-base-patch32',
                'normalization': 'none', 'projection': 'none', 'embedding': self.orbit.tolist(),
                'embedding_sha256': native.embedding_sha256(self.orbit),
                'base_vocabulary': {'sha256': native.BASE_VOCABULARY_SHA256}}

    def test_artifact_validates_vector_index_text_and_base_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)/'orbit.json'
            payload = self.artifact()
            artifact.write_text(json.dumps(payload))
            self.assertTrue(torch.equal(native.load_native_motion_embedding(artifact), self.orbit))
            for key, value in [('value_index', 19), ('text', native.NATIVE_PAN_TEXT),
                               ('embedding_sha256', '0'*64), ('base_vocabulary', {'sha256': 'wrong'}),
                               ('embedding', [0.]*511)]:
                artifact.write_text(json.dumps(dict(payload, **{key: value})))
                with self.assertRaises(ValueError):
                    native.load_native_motion_embedding(artifact)
            artifact.write_text(json.dumps(payload))
            wrong_cache = Path(temporary)/'cache.pkl'
            wrong_cache.write_bytes(b'wrong bytes')
            with self.assertRaises(ValueError):
                native.load_native_motion_embedding(artifact, base_vocabulary_path=wrong_cache)
            with self.assertRaises(ValueError):
                native.load_native_motion_embedding(artifact, kind='nativePan')

    def test_packaged_vectors_load_without_access_to_historical_server_paths(self):
        pan = native.load_native_motion_embedding(
            native.NATIVE_PAN_EMBEDDING_PATH, kind=native.NATIVE_PAN_VALUE,
        )
        orbit = native.load_native_motion_embedding(
            native.NATIVE_ORBIT_EMBEDDING_PATH, kind=native.NATIVE_ORBIT_VALUE,
        )
        adapter = native.NativeMotionTokens(self.vocabulary, pan, orbit)
        result, provenance = adapter.adapt_native_batch(make_batch([
            native.NATIVE_PAN_TEXT, native.NATIVE_ORBIT_TEXT,
        ]))
        self.assertEqual([row['changed'] for row in provenance], [True, True])
        self.assertTrue(torch.equal(tokens(result)[self.move, 0], pan))
        self.assertTrue(torch.equal(tokens(result)[self.move, 1], orbit))

    def test_frozen_parser_bytes_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrong_bridge = Path(temporary)/'bridge.py'
            wrong_bridge.write_text('raise RuntimeError("must not execute")')
            with self.assertRaises(ValueError):
                native.NativeMotionTokens(self.vocabulary, self.pan, self.orbit, bridge_path=wrong_bridge)


if __name__ == '__main__':
    unittest.main()
