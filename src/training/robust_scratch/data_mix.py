"""Opt-in experimental robust training component; full-run validation is pending.

Prospective natural replay plus additive, balanced primitive exposures.

No Torch dependency. ``EpochSchedule`` visits every natural reference exactly
once per epoch, shuffles it deterministically, and appends 16 separately sourced
primitive examples to each block of 112 natural examples. The last partial
natural block is retained. Primitive rows must be forced to the CCDM domain by
the trainer; natural rows continue to use its ordinary domain sampler.

``batch.references``, ``batch.seeds`` and ``batch.primitive_mask`` have identical
ordering. Save ``resume_identity(epoch, next_batch)`` in a durable checkpoint
and validate it before calling ``iter_batches(epoch, start_batch=next_batch)``.
Reference metadata, including source-scene identities, is preserved verbatim.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import random


PROTOCOL = 'robust-scratch-additive-natural-primitive-mix-v1'
FAMILIES = ('static', 'push_in', 'pull_out', 'pan', 'boom_up', 'boom_down', 'orbit')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def reference_id(ref):
    if (not isinstance(ref, dict) or not isinstance(ref.get('dataset'), str)
            or not ref['dataset'] or type(ref.get('index')) is not int or ref['index'] < 0):
        raise ValueError('A source reference requires a dataset and nonnegative integer index')
    return ref['dataset'], ref['index']


def deterministic_seed(seed, *parts):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Seed must be an integer in [0,2**63)')
    return int(canonical_hash([PROTOCOL, seed, *parts])[:16], 16) % (2**63)


def primitive_references_from_pairs(pairs):
    """Convert the preceding experiment's pair manifest to a separate stream."""
    result = []
    for pair in pairs:
        if not isinstance(pair, dict) or pair.get('native_family') not in FAMILIES:
            raise ValueError('A primitive pair needs its declared native family')
        ref = deepcopy(pair.get('primitive'))
        reference_id(ref)
        ref['native_family'] = pair['native_family']
        result.append(ref)
    return result


def natural_references_from_manifest(manifest, original_indices):
    """Build the full 60k original + 4096 narrow + 4096 broad natural pool.

    ``original_indices`` must come from the frozen original training split,
    never from validation/test membership or a native dataset.
    """
    indices = list(original_indices)
    if len(indices) != 60000 or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('The complete frozen original 60000-example train split is required')
    result = [{'dataset': 'original', 'index': i} for i in indices]
    for name in ('train_narrow', 'train_broad'):
        fresh = [deepcopy(ref) for ref in manifest[name] if ref.get('dataset') != 'original']
        if len(fresh) != 4096:
            raise ValueError('Each fresh natural cohort must contain exactly4096 examples: ' + name)
        if any(ref.get('dataset') != name for ref in fresh):
            raise ValueError('Fresh natural cohort contains an unexpected source dataset')
        result.extend(fresh)
    if len({reference_id(ref) for ref in result}) != len(result):
        raise ValueError('Natural training references overlap')
    return result


@dataclass(frozen=True)
class ScheduledBatch:
    batch_index: int
    references: tuple
    seeds: tuple
    primitive_mask: tuple
    natural_count: int
    primitive_count: int


class EpochSchedule:
    def __init__(self, natural_references, primitive_references, *, seed,
                 natural_batch_size=112, primitive_batch_size=16,
                 expected_natural_count=68192, expected_primitive_count=4096):
        deterministic_seed(seed, 'validation')
        if any(type(n) is not int or n < 1 for n in (natural_batch_size, primitive_batch_size)):
            raise ValueError('Both stream batch sizes must be positive integers')
        self.natural = tuple(deepcopy(list(natural_references)))
        self.primitives = tuple(deepcopy(list(primitive_references)))
        for refs, expected, name in ((self.natural, expected_natural_count, 'natural'),
                                     (self.primitives, expected_primitive_count, 'primitive')):
            if not refs or (expected is not None and len(refs) != expected):
                raise ValueError('Unexpected ' + name + ' source membership count')
            if len({reference_id(ref) for ref in refs}) != len(refs):
                raise ValueError('Duplicate ' + name + ' references')
        if {reference_id(r) for r in self.natural} & {reference_id(r) for r in self.primitives}:
            raise ValueError('Primitive examples must be separate from natural references')
        if expected_natural_count == 68192 and Counter(r['dataset'] for r in self.natural) != {
                'original': 60000, 'train_narrow': 4096, 'train_broad': 4096}:
            raise ValueError('Full natural membership must be SIM original60000+narrow4096+broad4096')
        if expected_primitive_count == 4096 and any(r['dataset'] != 'motion_train' for r in self.primitives):
            raise ValueError('The additional primitive stream must contain only safe SIM motion_train references')
        self.by_family = {family: [] for family in FAMILIES}
        for i, ref in enumerate(self.primitives):
            family = ref.get('native_family')
            if family not in self.by_family:
                raise ValueError('Every primitive requires one declared native family')
            self.by_family[family].append(i)
        if any(not indices for indices in self.by_family.values()):
            raise ValueError('All seven primitive families are required')
        self.seed, self.natural_batch_size, self.primitive_batch_size = seed, natural_batch_size, primitive_batch_size
        self.batch_count = (len(self.natural) + natural_batch_size - 1) // natural_batch_size
        self.identity = canonical_hash({
            'protocol': PROTOCOL, 'seed': seed, 'natural_references': self.natural,
            'primitive_references': self.primitives, 'natural_batch_size': natural_batch_size,
            'primitive_batch_size': primitive_batch_size, 'last_natural_partial': 'retained',
            'primitive_family_schedule': 'round-robin; independent shuffled source cycles',
        })

    @staticmethod
    def _epoch(epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('Epoch must be a nonnegative integer')

    def _indices(self, epoch):
        self._epoch(epoch)
        natural = list(range(len(self.natural)))
        random.Random(deterministic_seed(self.seed, epoch, 'natural-order')).shuffle(natural)
        primitive, pools, seen = [], {}, Counter()
        family_offset = deterministic_seed(self.seed, epoch, 'family-offset') % len(FAMILIES)
        for position in range(self.batch_count * self.primitive_batch_size):
            family = FAMILIES[(position + family_offset) % len(FAMILIES)]
            members = self.by_family[family]
            cycle, offset = divmod(seen[family], len(members))
            if offset == 0:
                pool = list(members)
                random.Random(deterministic_seed(self.seed, epoch, family, cycle)).shuffle(pool)
                pools[family] = pool
            primitive.append(pools[family][offset])
            seen[family] += 1
        return natural, primitive

    def iter_batches(self, epoch, *, start_batch=0):
        if type(start_batch) is not int or not 0 <= start_batch <= self.batch_count:
            raise ValueError('Resume batch is outside this epoch')
        natural, primitive = self._indices(epoch)
        for number in range(start_batch, self.batch_count):
            ni = natural[number*self.natural_batch_size:(number+1)*self.natural_batch_size]
            pi = primitive[number*self.primitive_batch_size:(number+1)*self.primitive_batch_size]
            refs = tuple(deepcopy([*(self.natural[i] for i in ni), *(self.primitives[i] for i in pi)]))
            seeds = tuple(deterministic_seed(self.seed, epoch, number, row, reference_id(ref))
                          for row, ref in enumerate(refs))
            yield ScheduledBatch(number, refs, seeds, (False,)*len(ni)+(True,)*len(pi), len(ni), len(pi))

    def resume_identity(self, epoch, next_batch):
        self._epoch(epoch)
        if type(next_batch) is not int or not 0 <= next_batch <= self.batch_count:
            raise ValueError('Resume batch is outside this epoch')
        return {'protocol': PROTOCOL, 'schedule_sha256': self.identity, 'epoch': epoch,
                'next_batch': next_batch,
                'natural_exposures_completed': min(len(self.natural), next_batch*self.natural_batch_size),
                'primitive_exposures_completed': next_batch*self.primitive_batch_size}

    def validate_resume(self, receipt):
        if not isinstance(receipt, dict) or receipt != self.resume_identity(receipt.get('epoch'), receipt.get('next_batch')):
            raise ValueError('Saved data schedule differs from the current frozen recipe')
        return receipt['epoch'], receipt['next_batch']

    def summary(self, epoch):
        _, primitive = self._indices(epoch)
        primitive_count = self.batch_count*self.primitive_batch_size
        exposures = len(self.natural)+primitive_count
        return {'protocol': PROTOCOL, 'schedule_sha256': self.identity, 'epoch': epoch,
                'natural_unique_examples': len(self.natural), 'primitive_unique_source_examples': len(self.primitives),
                'natural_exposures': len(self.natural), 'primitive_exposures': primitive_count,
                'total_exposures': exposures, 'optimizer_batches': self.batch_count,
                'primitive_exposure_fraction': primitive_count/exposures,
                'last_natural_batch_count': len(self.natural)-(self.batch_count-1)*self.natural_batch_size,
                'primitive_family_exposures': dict(Counter(self.primitives[i]['native_family'] for i in primitive)),
                'natural_source_counts': dict(Counter(ref['dataset'] for ref in self.natural)),
                'natural_membership_sha256': canonical_hash(self.natural),
                'primitive_membership_sha256': canonical_hash(self.primitives),
                'natural_references_replaced': 0, 'natural_references_dropped': 0}
