"""High-level text may supply facts without a low-level simulator program."""

import hashlib
import json

import pytest
import torch

from data.simulation.constants import NumericFeature
from data.simulation.utils import CLIP_PARAMETERS, cinematography_struct_parameters
from training.caption_missingness import CaptionMissingness, CaptionMissingnessConfig


@pytest.fixture
def vocabulary():
    result = {}
    for _, kind in CLIP_PARAMETERS:
        if isinstance(kind, NumericFeature):
            continue
        name = 'boolean' if kind is bool else kind.__name__
        result[name] = {
            'first': torch.tensor([1., 2., 3., 4.]),
            'second': torch.tensor([3., 4., 5., 6.]),
        }
    return result


def high_level(vocabulary, **config):
    return CaptionMissingness({'profile': 'high_level_only', **config}, vocabulary)


def test_high_level_facts_survive_absent_simulator_aliases_and_prior_missing_fill(vocabulary):
    sampler = high_level(vocabulary, min_keep_ratio=1., max_keep_ratio=1., endpoint_drop_probability=0.)
    fields, split = len(CLIP_PARAMETERS), len(cinematography_struct_parameters)
    caption = torch.arange(fields * 2 * 4, dtype=torch.float64).reshape(fields, 2, 4)
    present = torch.ones(2, fields, dtype=torch.bool)
    names = sampler.field_names
    present[0, names.index('dynamic_type')] = False
    present[0, names.index('dynamic_easing')] = False
    present[1, names.index('movement_type')] = False
    present[1, names.index('movement_speed')] = False
    present[1, names.index('initial_cameraAngle')] = False
    before, original_presence = caption.clone(), present.clone()
    result, effective = sampler.apply(caption, present, sample_seeds=[2, 3], force=True)
    expected = present.clone()
    expected[:, split:] = False
    assert torch.equal(effective, expected)
    assert torch.equal(caption, before) and torch.equal(present, original_presence)
    assert torch.equal(result[:split], before[:split])
    removed = (present & ~expected).T[..., None].expand_as(caption)
    expected_fill = sampler.unknown_tokens.to(caption)[:, None, :].expand_as(caption)
    torch.testing.assert_close(result[removed], expected_fill[removed])
    assert torch.equal(result[~removed], before[~removed])
    # Reapplying cannot revive a field hidden by a domain view or earlier mask.
    again, second_presence = sampler.apply(result, effective, sample_seeds=[2, 3], force=True)
    assert torch.equal(again, result) and torch.equal(second_presence, effective)


def test_movement_only_caption_drops_both_endpoints_and_all_instructions(vocabulary):
    sampler = high_level(vocabulary, min_keep_ratio=1., max_keep_ratio=1., endpoint_drop_probability=1.)
    keep = sampler.sample_keep_mask(3, sample_seeds=[4, 5, 6], force=True)
    expected = torch.zeros_like(keep)
    expected[:, [sampler.field_names.index(name) for name in ('movement_type', 'movement_speed')]] = True
    assert torch.equal(keep, expected)


def test_high_level_endpoints_can_be_observed_independently(vocabulary):
    sampler = high_level(vocabulary, min_keep_ratio=.5, max_keep_ratio=.5, endpoint_drop_probability=0.)
    keep = sampler.sample_keep_mask(64, sample_seeds=list(range(64)), force=True)
    initial = keep[:, sampler.field_names.index('initial_cameraAngle')]
    final = keep[:, sampler.field_names.index('final_cameraAngle')]
    assert (initial & ~final).any() and (~initial & final).any()
    assert not keep[:, len(cinematography_struct_parameters):].any()


def test_probability_zero_keeps_dense_rows_but_force_applies_profile(vocabulary):
    sampler = high_level(vocabulary, probability=0.)
    assert sampler.sample_keep_mask(3, sample_seeds=[4, 5, 6]).all()
    forced = sampler.sample_keep_mask(3, sample_seeds=[4, 5, 6], force=True)
    assert not forced[:, len(cinematography_struct_parameters):].any()


def test_profile_mask_bank_is_an_upper_bound_without_high_low_alias_closure(vocabulary, tmp_path):
    names = [name for name, _ in CLIP_PARAMETERS]
    row = [False] * len(names)
    for name in ('movement_type', 'initial_cameraAngle', 'dynamic_type', 'constraints_staticDistance'):
        row[names.index(name)] = True
    bank = tmp_path / 'sim_training_masks.json'
    bank.write_text(json.dumps({'field_names': names, 'presence_masks': [row]}))
    sampler = high_level(vocabulary, mask_bank_path=str(bank))
    keep = sampler.sample_keep_mask(2, sample_seeds=[7, 8], force=True)
    expected = torch.tensor([row, row])
    expected[:, len(cinematography_struct_parameters):] = False
    assert torch.equal(keep, expected)
    assert keep[:, names.index('movement_type')].all()
    assert not keep[:, names.index('movement_speed')].any()
    # The default bank policy still closes incomplete semantic alias groups.
    legacy = CaptionMissingness({'mask_bank_path': str(bank)}, vocabulary)
    assert not legacy.sample_keep_mask(1, sample_seeds=[7], force=True)[0, names.index('movement_type')]


@pytest.mark.parametrize('profile', ['alias_grouped', 'high_level_only'])
def test_seeded_masks_are_partition_invariant_and_leave_training_rng_unchanged(vocabulary, profile):
    sampler = CaptionMissingness({'profile': profile}, vocabulary)
    seeds = list(range(20))
    state = torch.get_rng_state().clone()
    full = sampler.sample_keep_mask(len(seeds), sample_seeds=seeds)
    parts = torch.cat([sampler.sample_keep_mask(7, sample_seeds=seeds[:7]),
                       sampler.sample_keep_mask(13, sample_seeds=seeds[7:])])
    assert torch.equal(full, parts)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize('force', [False, True])
def test_default_profile_retains_legacy_rng_draw_budget(vocabulary, force):
    # Four rows: keep-ratio draw, one draw per field, endpoint selection, and
    # optional augmentation selection. No profile-selector draw may be added.
    sampler = CaptionMissingness({}, vocabulary)
    generator = torch.Generator().manual_seed(90210)
    fields, size = len(CLIP_PARAMETERS), 4
    torch.rand(size, 1, generator=generator)
    torch.rand(size, fields, generator=generator)
    torch.rand(size, 1, generator=generator)
    if not force:
        torch.rand(size, 1, generator=generator)
    before = torch.get_rng_state().clone()
    try:
        torch.manual_seed(90210)
        implicit = sampler.sample_keep_mask(size, force=force)
        # Captured independently from the frozen pre-profile implementation,
        # CPU seed 90210, default configuration, 4 rows and 46 caption fields.
        legacy_masks = {
            False: '3db035f6ba995e4c4ddd4eb066b9c2bd6c28aa9d8d9602f0aeba31c896cf735c',
            True: 'f2e5f604428386e7b55f0e53debe288ec89b9519a7de272932233874ea42967c',
        }
        assert hashlib.sha256(implicit.numpy().tobytes()).hexdigest() == legacy_masks[force]
        assert torch.equal(torch.get_rng_state(), generator.get_state())
        torch.manual_seed(90210)
        explicit = CaptionMissingness({'profile': 'alias_grouped'}, vocabulary).sample_keep_mask(size, force=force)
        assert torch.equal(implicit, explicit)
        assert torch.equal(torch.get_rng_state(), generator.get_state())
    finally:
        torch.set_rng_state(before)


def test_invalid_profile_rejected_before_training():
    with pytest.raises(ValueError, match='profile'):
        CaptionMissingnessConfig(profile='native_et_test')
