"""Edit evaluation distinguishes real changes, controls and missing evidence."""

import numpy as np
import pytest
import torch

from data.simulation.constants import NumericFeature
from data.simulation.utils import cinematography_struct_parameters
from testing.trajectory_editing import (
    EDITABLE_FIELDS, FIELD_NAMES, build_edit_condition, edit_alias_fields,
    edit_geometry_totals, semantic_edit_totals,
)


@pytest.fixture
def vocabulary():
    result = {}
    for _, kind in cinematography_struct_parameters:
        if isinstance(kind, NumericFeature):
            continue
        key = 'boolean' if kind is bool else kind.__name__
        labels = (False, True) if kind is bool else [item.value for item in kind]
        result[key] = {label: torch.eye(21)[index] for index, label in enumerate(labels)}
    return result


@pytest.fixture
def caption(vocabulary):
    result = torch.zeros(len(FIELD_NAMES), 3, 21)
    for index, (_, kind) in enumerate(cinematography_struct_parameters):
        if isinstance(kind, NumericFeature):
            continue
        key = 'boolean' if kind is bool else kind.__name__
        result[index] = next(iter(vocabulary[key].values()))
    result[FIELD_NAMES.index('initial_shotSize'), 0] = vocabulary['ShotSize']['longShot']
    result[FIELD_NAMES.index('initial_shotSize'), 1] = vocabulary['ShotSize']['closeUp']
    return result


def test_literal_changed_edit_leaves_every_other_slot_and_input_untouched(caption, vocabulary):
    presence = torch.ones(3, len(FIELD_NAMES), dtype=torch.bool)
    index = FIELD_NAMES.index('initial_shotSize')
    presence[2, index] = False
    original, original_presence = caption.clone(), presence.clone()
    result = build_edit_condition(
        caption, presence, 'initial_shotSize', 'closeUp',
        source_values=['longShot', 'closeUp', None],
    )
    assert result['eligible'].tolist() == [True, False, False]
    assert result['unchanged'].tolist() == [False, True, False]
    assert result['missing'].tolist() == [False, False, True]
    expected_mask = torch.zeros_like(presence)
    expected_mask[:, index] = True
    assert torch.equal(result['caption_presence_mask'], expected_mask)
    assert torch.equal(result['caption_edit_mask'], expected_mask)
    untouched = [slot for slot in range(len(FIELD_NAMES)) if slot != index]
    assert torch.equal(result['caption_embedding'][untouched], original[untouched])
    torch.testing.assert_close(result['caption_embedding'][index], vocabulary['ShotSize']['closeUp'].expand(3, -1))
    assert torch.equal(caption, original) and torch.equal(presence, original_presence)
    assert result['excluded_preservation_fields'] == ('initial_shotSize',)
    assert result['diagnostic_alias_fields'] == ('setup_config_shotSize', 'dynamic_complementSetup_shotSize')


def test_noop_uses_actual_source_slot_and_cannot_restore_missing_labels(caption, vocabulary):
    presence = torch.ones(3, len(FIELD_NAMES), dtype=torch.bool)
    presence[2, FIELD_NAMES.index('initial_shotSize')] = False
    result = build_edit_condition(
        caption.double(), presence, 'initial_shotSize', None,
        source_values=['longShot', 'closeUp', 'longShot'],
    )
    assert torch.equal(result['caption_embedding'], caption.double())
    assert result['eligible'].tolist() == [True, True, False]
    assert result['caption_edit_mask'][:, result['field_index']].tolist() == [True, True, False]
    assert result['caption_embedding'].dtype == torch.float64


@pytest.mark.parametrize('field,value,error', [
    ('setup_config_shotSize', 'closeUp', 'high-level'),
    ('initial_shotSize', 'wide', 'Unknown'),
])
def test_invalid_edit_requests_fail(caption, vocabulary, field, value, error):
    with pytest.raises(ValueError, match=error):
        build_edit_condition(caption, torch.ones(3, len(FIELD_NAMES), dtype=torch.bool),
                             field, value, source_values=['longShot'] * 3)


def test_source_annotations_are_required_for_changed_cohort(caption, vocabulary):
    presence = torch.ones(3, len(FIELD_NAMES), dtype=torch.bool)
    with pytest.raises(ValueError, match='source_values'):
        build_edit_condition(caption, presence, 'initial_shotSize', 'closeUp', source_values=None)
    # An unknown/missing label is excluded even when a placeholder is marked present.
    result = build_edit_condition(caption, presence, 'initial_shotSize', 'closeUp',
                                  source_values=[None, 'unannotated', 'longShot'])
    assert result['missing'].tolist() == [True, True, False]
    assert result['eligible'].tolist() == [False, False, True]


@pytest.mark.parametrize('value,index', [('nativePan', 19), ('nativeOrbit', 20)])
def test_native_movement_edits_use_explicit_categories_without_metric_vectors(caption, value, index):
    result = build_edit_condition(
        caption, torch.ones((3, 10), dtype=torch.bool), 'movement_type', value,
        source_values=['static', 'static', None],
    )
    target = result['caption_embedding'][4]
    assert target[:, index].tolist() == [1., 1., 1.]
    assert target.sum(dim=-1).tolist() == [1., 1., 1.]
    assert result['eligible'].tolist() == [True, True, False]


def _semantic(caption, vocabulary, *, source=None, reconstruction=None, noop=None, edited=None,
              presence=None, eligible=None):
    return semantic_edit_totals(
        caption if source is None else source,
        caption if reconstruction is None else reconstruction,
        caption if noop is None else noop,
        caption if edited is None else edited,
        caption,
        torch.ones(3, len(FIELD_NAMES), dtype=torch.bool) if presence is None else presence,
        'initial_shotSize', 'closeUp', vocabulary,
        torch.tensor([True, False, False]) if eligible is None else eligible,
    )


def test_semantics_measure_true_target_gain_and_preserve_other_high_level_endpoint(caption, vocabulary):
    edited = caption.clone()
    edited[FIELD_NAMES.index('initial_shotSize'), 0] = vocabulary['ShotSize']['closeUp']
    totals = _semantic(caption, vocabulary, edited=edited)
    assert totals['target_source_top1'] == (0., 1)
    assert totals['target_edited_top1'] == (1., 1)
    assert totals['target_cosine_gain_vs_reconstruction'] == pytest.approx((1., 1))
    count = len(EDITABLE_FIELDS) - 1
    assert totals['unedited_edited_top1'] == (float(count), count)
    # Final shot size remains a measured preservation constraint.
    edited[FIELD_NAMES.index('final_shotSize'), 0] = vocabulary['ShotSize']['closeUp']
    changed = _semantic(caption, vocabulary, edited=edited)
    assert changed['unedited_edited_top1'] == (float(count - 1), count)


def test_prediction_agreement_is_separate_from_source_label_accuracy(caption, vocabulary):
    source = caption.clone()
    source[FIELD_NAMES.index('final_shotSize'), 0] = vocabulary['ShotSize']['closeUp']
    totals = _semantic(caption, vocabulary, source=source, edited=source)
    count = len(EDITABLE_FIELDS) - 1
    assert totals['unedited_edited_top1'] == (float(count - 1), count)
    assert totals['unedited_agreement_vs_source'] == (float(count), count)
    assert totals['unedited_agreement_vs_reconstruction'] == (float(count - 1), count)


def test_missing_and_placeholder_source_labels_do_not_become_preservation_targets(caption, vocabulary):
    presence = torch.ones(3, len(FIELD_NAMES), dtype=torch.bool)
    presence[0, FIELD_NAMES.index('final_shotSize')] = False
    caption[FIELD_NAMES.index('movement_type'), 0] = 0.
    totals = _semantic(caption, vocabulary, presence=presence)
    count = len(EDITABLE_FIELDS) - 3
    assert totals['unedited_edited_top1'] == (float(count), count)


def test_source_label_matching_permits_half_precision_storage(caption, vocabulary):
    vectors = torch.linspace(.013, .997, 21)
    vocabulary['CameraVerticalAngle']['low'] = vectors
    for field in ('initial_cameraAngle', 'final_cameraAngle'):
        caption[FIELD_NAMES.index(field)] = vectors
    totals = _semantic(caption.half(), vocabulary)
    count = len(EDITABLE_FIELDS) - 1
    assert totals['unedited_source_top1'] == (float(count), count)


def test_invalid_features_are_counted_and_cannot_win_top1_or_poison_totals(caption, vocabulary):
    edited = caption.clone()
    edited[FIELD_NAMES.index('initial_shotSize'), 0] = float('nan')
    edited[FIELD_NAMES.index('final_shotSize'), 0] = 0.
    totals = _semantic(caption, vocabulary, edited=edited)
    assert totals['target_edited_top1'] == (0., 0)
    assert totals['target_edited_cosine'] == (0., 0)
    assert totals['target_top1_gain_vs_source'] == (0., 0)
    assert totals['target_edited_invalid_feature_rate'] == (1., 1)
    assert totals['unedited_edited_invalid_feature_rate'] == (1., len(EDITABLE_FIELDS) - 1)
    assert all(np.isfinite(total) for total, _ in totals.values())


def test_no_eligible_edits_supply_no_evidence_instead_of_perfect_scores(caption, vocabulary):
    totals = _semantic(caption, vocabulary, eligible=torch.zeros(3, dtype=torch.bool))
    assert all(pair == (0., 0) for pair in totals.values())


@pytest.mark.parametrize('value', ['nativePan', 'nativeOrbit'])
def test_native_target_absent_from_frozen_vocabulary_is_unmeasured(caption, vocabulary, value):
    totals = semantic_edit_totals(
        caption, caption, caption, caption, caption,
        torch.ones((3, 10), dtype=torch.bool), 'movement_type', value, vocabulary,
        torch.tensor([True, False, False]),
    )
    assert all(pair == (0., 0) for name, pair in totals.items() if name.startswith('target_'))
    assert totals['unedited_edited_top1'] == (9., 9)


def test_invalid_semantic_target_does_not_masquerade_as_unavailable_native_class(caption, vocabulary):
    with pytest.raises(ValueError, match='Unknown'):
        semantic_edit_totals(
            caption, caption, caption, caption, caption,
            torch.ones((3, 10), dtype=torch.bool), 'movement_type', 'inventedMovement', vocabulary,
            torch.tensor([True, False, False]),
        )


def test_geometry_distinguishes_constant_camera_offset_from_motion_drift_and_excludes_padding():
    source = np.zeros((1, 4, 6))
    edited = source.copy()
    edited[0, :3, 2] = 2.
    edited[0, 3] = np.nan
    totals = edit_geometry_totals(edited, source, source, padding_mask=np.array([[False, False, False, True]]))
    assert totals['drift_vs_source_position_error'] == (6., 3)
    assert totals['reconstruction_step_drift'] == (0., 2)
    assert totals['reconstruction_relative_rotation_drift_deg'] == (0., 2)
    assert totals['drift_vs_source_invalid_generated_pose_rate'] == (0., 3)
    assert totals['projection_initial_bbox_height_delta_vs_reconstruction'] == (0., 0)


def test_geometry_measures_step_and_relative_rotation_drift_in_world_units():
    reference = np.zeros((1, 3, 6))
    edited = reference.copy()
    edited[0, :, 0] = [0., 1., 2.]
    edited[0, :, 4] = np.deg2rad([0., 10., 20.])
    totals = edit_geometry_totals(edited, reference, reference, padding_mask=np.zeros((1, 3), bool))
    assert totals['reconstruction_step_drift'] == (2., 2)
    assert totals['reconstruction_rotation_step_drift_deg'] == pytest.approx((20., 2))
    assert totals['reconstruction_relative_rotation_drift_deg'] == pytest.approx((30., 2))


def test_invalid_valid_poses_are_reported_but_excluded_from_drift_and_motion():
    reference = np.zeros((1, 3, 6))
    edited = reference.copy()
    edited[0, 1, 0] = np.nan
    totals = edit_geometry_totals(edited, reference, reference, padding_mask=np.zeros((1, 3), bool))
    assert totals['drift_vs_source_invalid_generated_pose_rate'] == (1., 3)
    assert totals['drift_vs_source_position_error'] == (0., 2)
    assert totals['reconstruction_step_drift'] == (0., 0)
    assert all(np.isfinite(total) for total, _ in totals.values())


def test_initial_projection_is_signed_paired_and_uses_first_real_frame():
    reconstruction = np.zeros((1, 3, 6))
    reconstruction[..., 2] = 5.
    edited = reconstruction.copy()
    edited[..., 2] = 3.
    # The padded frame would otherwise reverse the signed diagnostic.
    edited[0, 0, 2] = 10.
    kwargs = dict(padding_mask=np.array([[True, False, False]]),
                  subject_trajectory=np.zeros((1, 6)), subject_dimensions=np.array([[2., 2., 2.]]),
                  intrinsics=np.array([[35., 1.]]))
    totals = edit_geometry_totals(edited, reconstruction, reconstruction, **kwargs)
    assert totals['projection_initial_edited_bbox_height'] == pytest.approx((1., 1))
    assert totals['projection_initial_reconstruction_bbox_height'] == pytest.approx((.5, 1))
    assert totals['projection_initial_bbox_height_delta_vs_reconstruction'] == pytest.approx((.5, 1))
    edited[..., 2] = -3.
    behind = edit_geometry_totals(edited, reconstruction, reconstruction, **kwargs)
    assert behind['projection_initial_bbox_height_delta_vs_reconstruction'] == (0., 0)


def test_entirely_empty_geometry_has_zero_counts():
    empty = np.empty((2, 0, 6))
    totals = edit_geometry_totals(empty, empty, empty, padding_mask=np.empty((2, 0), bool))
    assert all(pair == (0., 0) for pair in totals.values())


def test_aliases_are_diagnostics_and_never_suggest_other_high_level_overrides():
    assert edit_alias_fields('movement_speed') == ('dynamic_easing',)
    assert edit_alias_fields('movement_type') == ('dynamic_type', 'dynamic_direction', 'dynamic_movementMode')
    assert edit_alias_fields('initial_subjectFraming') == (
        'setup_config_subjectFraming_position', 'dynamic_complementSetup_subjectFraming_position',
    )
