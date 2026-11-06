"""Two fixed native motion features in the existing movement_type slot.

Training: call ``augment_training_batch`` after the frozen unpaired CCDM motion
repair. Its certified, nondirectional orbit is encoded as nativeOrbit, so the
high-level-only missingness profile retains the motion cue.

Inference: ``adapt_native_batch`` reads complete caption text with the frozen
text-only parser and changes ONLY movement_type on unambiguous native Pan or
Orbit captions. The remaining 45 fields, including available speed/easing and
all endpoints, stay untouched. Unknown/conflicting captions remain unchanged.
No target camera or private geometry is read by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


PROTOCOL = 'robust-scratch-single-slot-native-motion-v1'
PAN_VOCABULARY_PROTOCOL = 'ccdm-native-pan-fixed-clip-pooler-v1'
ORBIT_VOCABULARY_PROTOCOL = 'robust-scratch-native-orbit-fixed-clip-pooler-v1'
NATIVE_PAN_VALUE = 'nativePan'
NATIVE_ORBIT_VALUE = 'nativeOrbit'
NATIVE_PAN_TEXT = 'The camera pans to the character.'
NATIVE_ORBIT_TEXT = 'The camera rotates around the character.'
NATIVE_PAN_INDEX = 19
NATIVE_ORBIT_INDEX = 20
BASE_VOCABULARY_SHA256 = '6e27c50dbeed112c03a4967958e2361d6ceb45bcc6e1a766f318f985c0aea8a0'
BRIDGE_SHA256 = '37f37ac51632b4b273445c39e5251585dc3bd634db386f201ad742397da3bdf6'

# Historical server paths inside these artifacts are provenance, not runtime inputs.
_ASSETS_DIRECTORY = Path(__file__).with_name('assets')
NATIVE_PAN_EMBEDDING_PATH = _ASSETS_DIRECTORY / 'native_pan_embedding.json'
NATIVE_ORBIT_EMBEDDING_PATH = _ASSETS_DIRECTORY / 'native_orbit_embedding.json'


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def embedding_sha256(value):
    import torch
    vector = torch.as_tensor(value, dtype=torch.float32).detach().cpu().flatten()
    if vector.shape != (512,) or not torch.isfinite(vector).all():
        raise ValueError('Native motion features require a finite unprojected512-D vector')
    return hashlib.sha256(struct.pack('<512f', *vector.tolist())).hexdigest()


def load_native_motion_embedding(path, *, kind=NATIVE_ORBIT_VALUE, base_vocabulary_path=None):
    """Load an additive artifact without modifying the legacy cache/statistics."""
    import torch
    definitions = {
        NATIVE_PAN_VALUE: (PAN_VOCABULARY_PROTOCOL, NATIVE_PAN_INDEX, NATIVE_PAN_TEXT),
        NATIVE_ORBIT_VALUE: (ORBIT_VOCABULARY_PROTOCOL, NATIVE_ORBIT_INDEX, NATIVE_ORBIT_TEXT),
    }
    if kind not in definitions:
        raise ValueError('Unknown native motion artifact kind')
    protocol, index, text = definitions[kind]
    payload = json.loads(Path(path).read_text())
    if type(payload.get('value_index')) is not int:
        raise ValueError('Native motion artifact value_index must be an explicit integer')
    expected = {'protocol': protocol, 'field': 'movement_type', 'value': kind,
                'value_index': index, 'text': text, 'embedding_width': 512,
                'representation': 'CLIPTextModel.pooler_output',
                'model_name': 'openai/clip-vit-base-patch32',
                'normalization': 'none', 'projection': 'none'}
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError('Native motion artifact identity differs: ' + name)
    if payload.get('base_vocabulary', {}).get('sha256') != BASE_VOCABULARY_SHA256:
        raise ValueError('Native motion artifact uses a different base vocabulary')
    if base_vocabulary_path is not None and file_sha256(base_vocabulary_path) != BASE_VOCABULARY_SHA256:
        raise ValueError('Actual base vocabulary differs from the frozen artifact identity')
    vector = torch.as_tensor(payload.get('embedding'), dtype=torch.float32)
    if vector.shape != (512,) or embedding_sha256(vector) != payload.get('embedding_sha256'):
        raise ValueError('Native motion artifact vector failed identity validation')
    return vector


def _load_bridge(path=None):
    path = Path(path) if path else Path(__file__).with_name('caption_bridge.py')
    if file_sha256(path) != BRIDGE_SHA256:
        raise ValueError('Native motion parsing requires the frozen text-only bridge')
    spec = importlib.util.spec_from_file_location('_robust_scratch_frozen_caption_bridge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeMotionTokens:
    def __init__(self, base_vocabulary, native_pan_embedding, native_orbit_embedding, *, bridge_path=None):
        import torch
        from data.simulation.constants import CameraMovementType
        from data.simulation.utils import CLIP_PARAMETERS, cinematography_struct_parameters
        if len(CameraMovementType) != NATIVE_PAN_INDEX:
            raise ValueError('Native motion token indices disagree with the frozen base enum')
        self.fields = tuple(name for name, _ in CLIP_PARAMETERS)
        self.split = len(cinematography_struct_parameters)
        if len(self.fields) != 46:
            raise ValueError('Native motion features require the frozen46-field schema')
        if any(name in base_vocabulary.get('CameraMovementType', {})
               for name in (NATIVE_PAN_VALUE, NATIVE_ORBIT_VALUE)):
            raise ValueError('Native features must remain separate from the base vocabulary/category means')
        self.vectors = {NATIVE_PAN_VALUE: torch.as_tensor(native_pan_embedding).detach().float().cpu().flatten().clone(),
                        NATIVE_ORBIT_VALUE: torch.as_tensor(native_orbit_embedding).detach().float().cpu().flatten().clone()}
        for vector in self.vectors.values():
            embedding_sha256(vector)
        if torch.equal(self.vectors[NATIVE_PAN_VALUE], self.vectors[NATIVE_ORBIT_VALUE]):
            raise ValueError('Native Pan and Orbit must have distinct fixed text features')
        self.indices = {NATIVE_PAN_VALUE: NATIVE_PAN_INDEX, NATIVE_ORBIT_VALUE: NATIVE_ORBIT_INDEX}
        self.bridge = _load_bridge(bridge_path)

    def _validate(self, batch):
        import torch
        mask = batch.get('prompt_none_mask')
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != 46:
            raise ValueError('Caption presence must be boolean[B,46]')
        size = mask.shape[0]
        rows = [list(a)+list(b) for a, b in zip(batch.get('cinematography_prompt_parameters', []),
                                               batch.get('simulation_instruction_parameters', []))]
        if len(rows) != size or any(tuple(e[0] for e in row) != self.fields for row in rows):
            raise ValueError('Caption metadata must contain the ordered46-field schema for every row')
        tokens, axes = [], []
        for key, count in (('cinematography_prompt', self.split), ('simulation_instruction', 46-self.split)):
            value = batch.get(key)
            if not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] != 512:
                raise ValueError('Structured native captions require512-D tokens')
            if value.shape[:2] == (count, size):
                tokens.append(value)
                axes.append(False)
            elif value.shape[:2] == (size, count):
                tokens.append(value.transpose(0, 1))
                axes.append(True)
            else:
                raise ValueError('Structured caption axes disagree with metadata')
        if tokens[0].device != tokens[1].device or tokens[0].dtype != tokens[1].dtype:
            raise ValueError('Structured caption blocks require a common dtype/device')
        return mask, rows, torch.cat(tokens, 0), axes

    def _apply(self, batch, changes, *, receipt):
        mask, rows, tokens, axes = self._validate(batch)
        result = dict(batch)
        new_mask, new_tokens = mask.clone(), tokens.clone()
        field = self.fields.index('movement_type')
        raw = batch.get('raw_prompt')
        raw_copy = deepcopy(raw) if raw is not None else None
        for row, value in changes.items():
            if type(row) is not int or not 0 <= row < len(rows) or value not in self.vectors:
                raise ValueError('Invalid native movement replacement')
            vector = self.vectors[value].to(device=tokens.device, dtype=tokens.dtype)
            rows[row][field] = ('movement_type', value, self.indices[value], vector)
            new_tokens[field, row] = vector
            new_mask[row, field] = True
            # Procedural native-style samples deliberately store neutral raw
            # captions (often {}). Their certified conditioning lives in the
            # parameter metadata after augmentation. Never invent a raw schema
            # or overwrite an absent movement object just to mirror that cue.
            if (raw_copy is not None and isinstance(raw_copy[row], dict)
                    and isinstance(raw_copy[row].get('movement'), dict)
                    and 'type' in raw_copy[row]['movement']):
                raw_copy[row]['movement']['type'] = value
        first, second = new_tokens[:self.split], new_tokens[self.split:]
        result['cinematography_prompt'] = first.transpose(0, 1) if axes[0] else first
        result['simulation_instruction'] = second.transpose(0, 1) if axes[1] else second
        result['prompt_none_mask'] = new_mask
        result['cinematography_prompt_parameters'] = [row[:self.split] for row in rows]
        result['simulation_instruction_parameters'] = [row[self.split:] for row in rows]
        if raw_copy is not None:
            result['raw_prompt'] = raw_copy
        result['native_motion_tokens'] = {'protocol': PROTOCOL, 'changed_rows': len(changes),
                                          'changed_fields': ['movement_type'] if changes else [],
                                          'other45fields_preserved': True, **receipt}
        return result

    def augment_training_batch(self, batch):
        """Promote only the frozen repair's known, unambiguous native orbit."""
        import torch
        mask, rows, _, _ = self._validate(batch)
        views = batch.get('domain_augmentation_view')
        if not torch.is_tensor(views) or views.shape != (len(rows),):
            raise ValueError('Training native motion tokens require explicit domain views')
        mode = self.fields.index('dynamic_movementMode')
        movement = self.fields.index('movement_type')
        dynamic = self.fields.index('dynamic_type')
        changes = {}
        for row, entries in enumerate(rows):
            if (int(views[row]) == 2 and bool(mask[row, mode]) and entries[mode][1] == 'arc'
                    and entries[mode][2] >= 0 and bool(mask[row, dynamic]) and entries[dynamic][1] == 'simple'
                    and entries[dynamic][2] >= 0
                    and (not bool(mask[row, movement]) or entries[movement][1] == NATIVE_ORBIT_VALUE)):
                changes[row] = NATIVE_ORBIT_VALUE
        return self._apply(batch, changes, receipt={'source': 'certified synthetic motion metadata',
                                                   'target_camera_read': False})

    def adapt_native_batch(self, batch):
        """Read full text, changing only known Pan/Orbit movement_type values."""
        mask, rows, _, _ = self._validate(batch)
        texts, ids = batch.get('text_prompts'), batch.get('item_ids')
        if (not isinstance(texts, (list, tuple)) or len(texts) != len(rows)
                or not isinstance(ids, (list, tuple)) or len(ids) != len(rows)
                or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids))):
            raise ValueError('One unchanged complete caption and unique source ID is required per row')
        changes, provenance = {}, []
        for row, (text, identity) in enumerate(zip(texts, ids)):
            _, parsed = self.bridge.parse_with_provenance(text)
            families = parsed['motion_families']
            accepted = (parsed['all_sentences_recognized'] and len(families) == 1
                        and not any('principal movement:' in reason for reason in parsed['masked_semantics']))
            value = ({'pan': NATIVE_PAN_VALUE, 'orbit': NATIVE_ORBIT_VALUE}.get(families[0])
                     if accepted else None)
            if value is not None:
                changes[row] = value
            provenance.append({'item_id': identity, 'source_prompt_sha256': parsed['source_prompt_sha256'],
                               'parser_sha256': BRIDGE_SHA256, 'native_family': families[0] if accepted else 'unknown',
                               'movement_value': value, 'changed': value is not None,
                               'changed_fields': ['movement_type'] if value is not None else [],
                               'all_sentences_recognized': parsed['all_sentences_recognized'],
                               'unknown_sentences': parsed['unknown_sentences'],
                               'masked_semantics': parsed['masked_semantics'],
                               'target_camera_used': False, 'other45fields_preserved': True})
        return self._apply(batch, changes, receipt={'source': 'complete caption text only',
                                                   'target_camera_read': False}), provenance
