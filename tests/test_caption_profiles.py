"""Ten-field categorical missingness stays deterministic and never restores facts."""
import json

import pytest
import torch

from models.categorical_latent import FIELD_NAMES, from_ids
from training.caption_missingness import CaptionMissingness, CaptionMissingnessConfig


def high_level(**config):
    return CaptionMissingness({'profile': 'high_level_only', **config})


def test_preserves_observed_facts_without_restoring_missing_fields_or_mutating_inputs():
    sampler = high_level(min_keep_ratio=1., max_keep_ratio=1., endpoint_drop_probability=0.)
    ids = torch.zeros(2, 10, dtype=torch.long)
    ids[1, [0, 4, 5]] = -1
    caption, present = from_ids(ids)
    before, original_presence = caption.clone(), present.clone()
    result, effective = sampler.apply(caption, present, sample_seeds=[2, 3], force=True)
    assert torch.equal(effective, present)
    assert torch.equal(result, caption)
    assert torch.equal(caption, before) and torch.equal(present, original_presence)
    again, second = sampler.apply(result, effective, sample_seeds=[2, 3], force=True)
    assert torch.equal(again, result) and torch.equal(second, effective)


def test_movement_only_caption_drops_both_endpoints():
    sampler = high_level(min_keep_ratio=1., max_keep_ratio=1., endpoint_drop_probability=1.)
    keep = sampler.sample_keep_mask(3, sample_seeds=[4, 5, 6], force=True)
    expected = torch.zeros_like(keep)
    expected[:, [4, 5]] = True
    assert torch.equal(keep, expected)


def test_endpoint_fields_can_be_observed_independently():
    keep = high_level(min_keep_ratio=.5, max_keep_ratio=.5,
                      endpoint_drop_probability=0.).sample_keep_mask(
                          64, sample_seeds=list(range(64)), force=True)
    assert (keep[:, 0] & ~keep[:, 6]).any() and (~keep[:, 0] & keep[:, 6]).any()


def test_probability_zero_keeps_dense_rows_but_force_applies_missingness():
    sampler = high_level(probability=0., min_keep_ratio=0., max_keep_ratio=0.)
    assert sampler.sample_keep_mask(3, sample_seeds=[4, 5, 6]).all()
    assert not sampler.sample_keep_mask(3, sample_seeds=[4, 5, 6], force=True)[:, :4].any()


@pytest.mark.parametrize('probability', [0., .75])
@pytest.mark.parametrize('force', [False, True])
def test_speed_dropout_is_independent_and_partition_invariant(probability, force):
    sampler = high_level(probability=probability, speed_drop_probability=.5)
    seeds = list(range(2048))
    state = torch.get_rng_state().clone()
    keep = sampler.sample_keep_mask(len(seeds), sample_seeds=seeds, force=force)
    parts = torch.cat([sampler.sample_keep_mask(700, sample_seeds=seeds[:700], force=force),
                       sampler.sample_keep_mask(1348, sample_seeds=seeds[700:], force=force)])
    assert torch.equal(keep, parts) and torch.equal(torch.get_rng_state(), state)
    assert .45 < (~keep[:, 5]).float().mean().item() < .55
    assert keep[:, 4].all()
    if probability == 0 and not force:
        assert keep[:, [0, 1, 2, 3, 4, 6, 7, 8, 9]].all()


def test_dropped_and_already_missing_tokens_are_zero_without_changing_clean_targets():
    sampler = high_level(probability=0., speed_drop_probability=1.)
    ids = torch.zeros(2, 10, dtype=torch.long)
    ids[1, 0] = -1
    caption, presence = from_ids(ids)
    before = caption.clone()
    result, effective = sampler.apply(caption, presence, sample_seeds=[2, 3])
    assert not effective[:, 5].any()
    assert torch.count_nonzero(result.masked_select(~effective.T[..., None])) == 0
    assert torch.equal(before, caption)


def test_mask_bank_is_only_an_upper_bound(tmp_path):
    row = [False] * 10
    row[0] = row[4] = True
    bank = tmp_path / 'training_masks.json'
    bank.write_text(json.dumps({'field_names': list(FIELD_NAMES), 'presence_masks': [row]}))
    sampler = high_level(mask_bank_path=str(bank))
    keep = sampler.sample_keep_mask(2, sample_seeds=[7, 8], force=True)
    assert torch.equal(keep, torch.tensor([row, row]))
    assert not keep[:, 5].any()


@pytest.mark.parametrize('profile', ['independent', 'high_level_only'])
def test_seeded_masks_leave_training_rng_unchanged(profile):
    sampler = CaptionMissingness({'profile': profile})
    seeds = list(range(20))
    state = torch.get_rng_state().clone()
    full = sampler.sample_keep_mask(20, sample_seeds=seeds)
    parts = torch.cat([sampler.sample_keep_mask(7, sample_seeds=seeds[:7]),
                       sampler.sample_keep_mask(13, sample_seeds=seeds[7:])])
    assert torch.equal(full, parts) and torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize('probability', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_speed_dropout_rejected(probability):
    with pytest.raises(ValueError, match='speed_drop_probability'):
        CaptionMissingnessConfig(speed_drop_probability=probability)


def test_mean_fill_is_no_longer_a_conditioning_option():
    with pytest.raises(ValueError, match='zero'):
        CaptionMissingnessConfig(fill='mean')
