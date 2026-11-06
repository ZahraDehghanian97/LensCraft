"""Source preservation and durable-order contracts for the prospective mix."""
from collections import Counter
from copy import deepcopy
import random
import unittest

from training.robust_scratch.data_mix import EpochSchedule, FAMILIES, natural_references_from_manifest, primitive_references_from_pairs, reference_id


def sources(natural=23, per_family=3):
    return ([{'dataset': 'natural', 'index': i, 'source_scene': 'scene:' + str(i)} for i in range(natural)],
            [{'dataset': 'primitive', 'index': i, 'native_family': family}
             for i, family in enumerate(FAMILIES*per_family)])


class DataMixTests(unittest.TestCase):
    def schedule(self, natural=23, per_family=3, **kwargs):
        first, second = sources(natural, per_family)
        return EpochSchedule(first, second, seed=42, expected_natural_count=None,
                             expected_primitive_count=None, **kwargs)

    def test_full_recipe_counts_last_partial_and_every_natural_member(self):
        natural = [{'dataset': 'original', 'index': i} for i in range(60000)]
        natural += [{'dataset': name, 'index': i} for name in ('train_narrow', 'train_broad') for i in range(4096)]
        primitive = [{'dataset': 'motion_train', 'index': i, 'native_family': FAMILIES[i % 7]} for i in range(4096)]
        schedule = EpochSchedule(natural, primitive, seed=42)
        summary = schedule.summary(0)
        self.assertEqual((schedule.batch_count, summary['natural_exposures'], summary['primitive_exposures']),
                         (609, 68192, 9744))
        self.assertEqual(summary['total_exposures'], 77936)
        self.assertEqual(summary['last_natural_batch_count'], 96)
        self.assertEqual(summary['primitive_family_exposures'], {family: 1392 for family in FAMILIES})
        batches = list(schedule.iter_batches(0))
        seen = [reference_id(ref) for batch in batches for ref, primitive in zip(batch.references, batch.primitive_mask) if not primitive]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {reference_id(ref) for ref in natural})
        self.assertEqual(len(batches[-1].references), 112)
        self.assertTrue(all(batch.primitive_count == 16 for batch in batches))
        self.assertEqual(sum(batch.natural_count for batch in batches), 68192)

    def test_separate_stream_never_substitutes_a_natural_target(self):
        schedule = self.schedule(natural_batch_size=5, primitive_batch_size=3)
        batches = list(schedule.iter_batches(2))
        self.assertEqual(sum(batch.natural_count for batch in batches), 23)
        self.assertEqual(sum(batch.primitive_count for batch in batches), 15)
        for batch in batches:
            self.assertEqual(batch.primitive_mask, (False,)*batch.natural_count+(True,)*3)
            for ref in batch.references[:batch.natural_count]:
                self.assertEqual(ref['source_scene'], 'scene:' + str(ref['index']))

    def test_resume_replays_identical_refs_masks_and_per_example_seeds(self):
        schedule = self.schedule(natural_batch_size=5, primitive_batch_size=3)
        full = list(schedule.iter_batches(7))
        receipt = schedule.resume_identity(7, 3)
        self.assertEqual(schedule.validate_resume(receipt), (7, 3))
        self.assertEqual(full[3:], list(schedule.iter_batches(7, start_batch=3)))
        self.assertEqual(receipt['natural_exposures_completed'], 15)
        self.assertEqual(receipt['primitive_exposures_completed'], 9)
        self.assertEqual(list(schedule.iter_batches(7, start_batch=5)), [])

    def test_resume_rejects_recipe_membership_seed_and_count_changes(self):
        schedule = self.schedule(natural_batch_size=5, primitive_batch_size=3)
        receipt = schedule.resume_identity(1, 2)
        for key, value in [('schedule_sha256', 'different'), ('natural_exposures_completed', 9), ('extra', True)]:
            bad = dict(receipt, **{key: value})
            with self.assertRaises(ValueError):
                schedule.validate_resume(bad)
        natural, primitive = sources()
        changed = EpochSchedule(natural, primitive, seed=43, natural_batch_size=5, primitive_batch_size=3,
                                expected_natural_count=None, expected_primitive_count=None)
        with self.assertRaises(ValueError):
            changed.validate_resume(receipt)

    def test_family_exposure_balanced_and_each_source_cycle_without_replacement(self):
        schedule = self.schedule(natural=101, natural_batch_size=4, primitive_batch_size=7)
        batches = list(schedule.iter_batches(0))
        refs = [ref for batch in batches for ref, mask in zip(batch.references, batch.primitive_mask) if mask]
        counts = Counter(ref['native_family'] for ref in refs)
        self.assertEqual(max(counts.values())-min(counts.values()), 0)
        for family in FAMILIES:
            ids = [ref['index'] for ref in refs if ref['native_family'] == family]
            for start in range(0, len(ids)-2, 3):
                self.assertEqual(len(set(ids[start:start+3])), 3)

    def test_no_global_rng_change_and_reproducible_but_different_epochs(self):
        schedule = self.schedule(natural_batch_size=5, primitive_batch_size=3)
        random.seed(892)
        before = random.getstate()
        first = list(schedule.iter_batches(0))
        self.assertEqual(before, random.getstate())
        self.assertEqual(first, list(schedule.iter_batches(0)))
        self.assertNotEqual(first, list(schedule.iter_batches(1)))
        self.assertTrue(all(0 <= seed < 2**63 for batch in first for seed in batch.seeds))

    def test_constructor_and_yielded_refs_do_not_alias_callers(self):
        natural, primitive = sources()
        original = deepcopy((natural, primitive))
        schedule = EpochSchedule(natural, primitive, seed=42, expected_natural_count=None, expected_primitive_count=None)
        batch = next(schedule.iter_batches(0))
        batch.references[0]['source_scene'] = 'mutated'
        self.assertEqual((natural, primitive), original)
        self.assertNotEqual(next(schedule.iter_batches(0)).references[0]['source_scene'], 'mutated')

    def test_bad_membership_rejected_including_reused_broad_alias(self):
        natural, primitive = sources()
        for bad_natural, bad_primitive in ((natural+natural[:1], primitive),
                                           (natural, primitive+primitive[:1]),
                                           (natural+[{**primitive[0]}], primitive),
                                           (natural, [ref for ref in primitive if ref['native_family'] != 'orbit'])):
            with self.assertRaises(ValueError):
                EpochSchedule(bad_natural, bad_primitive, seed=42, expected_natural_count=None, expected_primitive_count=None)
        manifest = {'train_narrow': [{'dataset': 'train_narrow', 'index': i} for i in range(4096)]}
        manifest['train_broad'] = deepcopy(manifest['train_narrow'])
        with self.assertRaisesRegex(ValueError, 'unexpected source dataset'):
            natural_references_from_manifest(manifest, range(60000))

    def test_pair_conversion_preserves_scene_and_is_a_distinct_reference_stream(self):
        pairs = [{'base': {'dataset': 'natural', 'index': 9}, 'native_family': 'orbit',
                  'primitive': {'dataset': 'motion_train', 'index': 1, 'source_scene': 'original:9'}}]
        refs = primitive_references_from_pairs(pairs)
        self.assertEqual(refs, [{'dataset': 'motion_train', 'index': 1, 'source_scene': 'original:9', 'native_family': 'orbit'}])
        self.assertNotEqual(reference_id(refs[0]), reference_id(pairs[0]['base']))


if __name__ == '__main__':
    unittest.main()
