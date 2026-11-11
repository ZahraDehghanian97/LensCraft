"""Literal one-slot trajectory-edit conditions and additive evaluation metrics.

These helpers never construct an edited ground-truth trajectory. Learned
semantic metrics must receive features from a separate, frozen evaluator;
world-space differences describe drift, not edit correctness. In particular,
projected bbox height is a directional diagnostic, not a shot-size classifier.
"""

from enum import Enum

import numpy as np
import torch
import torch.nn.functional as functional

from data.simulation.utils import cinematography_struct_parameters
from models.categorical_latent import (
    CATEGORY_VALUES, FIELD_NAMES, TOKEN_WIDTH, validate_semantic,
)
from testing.metrics.modules.geometry import (
    _array, _frame_array, _xyz_matrix, geometry_metric_totals,
)


EDITABLE_FIELDS = tuple(name for name, _ in cinematography_struct_parameters)
_FIELD_TYPES = dict(cinematography_struct_parameters)


def _field(field):
    if field not in EDITABLE_FIELDS:
        raise ValueError(f"Edit field must be a categorical high-level field: {EDITABLE_FIELDS}")
    return FIELD_NAMES.index(field), _FIELD_TYPES[field]


def _tokens(value, name, shape=None):
    if (not torch.is_tensor(value) or value.ndim != 3
            or value.shape[0] < len(FIELD_NAMES) or not value.is_floating_point()):
        raise ValueError(f"{name} must contain floating-point metric features for ten high-level fields")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    return value


def _presence(value, batch, fields, device, name):
    if (not torch.is_tensor(value) or value.dtype != torch.bool
            or value.shape != (batch, fields)):
        raise ValueError(f"{name} must be boolean [batch, fields]")
    return value.to(device)


def _vocabulary(kind, clip_embeddings, width, device, dtype):
    key = 'boolean' if kind is bool else kind.__name__
    vocabulary = clip_embeddings.get(key, {})
    if not vocabulary:
        raise ValueError(f"Missing CLIP vocabulary: {key}")
    labels = list(vocabulary)
    vectors = torch.stack([
        torch.as_tensor(vocabulary[label], device=device, dtype=dtype).detach().flatten()
        for label in labels
    ])
    if vectors.shape != (len(labels), width):
        raise ValueError(f"CLIP vocabulary {key} does not match embedding width {width}")
    if not torch.isfinite(vectors).all() or (vectors.norm(dim=-1) == 0).any():
        raise ValueError(f"CLIP vocabulary {key} contains invalid or zero vectors")
    return labels, vectors


def _value(value):
    return value.value if isinstance(value, Enum) else value


def edit_alias_fields(field):
    """Return potentially conflicting low-level slots; never overwrite them.

    setup.kind controls which low-level endpoint describes the initial frame.
    Both endpoints are therefore reported without using source labels to infer
    a simulator program. Other high-level endpoints remain preservation tests.
    """
    _field(field)
    if field.startswith(('initial_', 'final_')):
        suffix = field.split('_', 1)[1]
        suffix = 'subjectFraming_position' if suffix == 'subjectFraming' else suffix
        return (f'setup_config_{suffix}', f'dynamic_complementSetup_{suffix}')
    if field == 'movement_type':
        return ('dynamic_type', 'dynamic_direction', 'dynamic_movementMode')
    return ('dynamic_easing',)


def build_edit_condition(
    caption, presence, field, value, *, source_values,
):
    """Build a sparse prompt without altering any other trajectory slot.

    ``source_values`` is one explicit source annotation per example (or None).
    It selects a genuinely changed cohort only; those labels are never added
    to the decoder conditions. ``value=None`` copies the source caption token
    for the no-op control. Missing and already-target rows remain distinguishable
    from changed rows and must not contribute to changed-edit success rates.
    The returned full caption is a clone; its unselected slots are not prompt
    conditions because both decoder masks contain only the requested field.
    """
    if (not torch.is_tensor(caption) or caption.ndim != 3
            or caption.shape[0] != len(FIELD_NAMES) or caption.shape[2] != TOKEN_WIDTH
            or not caption.is_floating_point()):
        raise ValueError('caption must be categorical [10,batch,21] semantic tokens')
    index, _ = _field(field)
    fields, batch, width = caption.shape
    presence = _presence(presence, batch, fields, caption.device, 'presence')
    safe_caption = caption.masked_fill(~presence.T[..., None], 0)
    validate_semantic(safe_caption, presence)
    if source_values is None or len(source_values) != batch:
        raise ValueError('source_values must contain one source value (or None) per example')
    labels = CATEGORY_VALUES[index]
    normalized = [_value(item) for item in source_values]
    known = torch.tensor([item in labels for item in normalized], dtype=torch.bool, device=caption.device)
    known &= presence[:, index]
    if value is not None:
        value = _value(value)
        if value not in labels:
            raise ValueError(f"Unknown {field} edit value {value!r}; expected one of {labels}")
        unchanged = known & torch.tensor([item == value for item in normalized], dtype=torch.bool, device=caption.device)
        eligible = known & ~unchanged
    else:
        unchanged, eligible = known.clone(), known.clone()
    edited = caption.clone()
    sparse = torch.zeros(batch, fields, dtype=torch.bool, device=caption.device)
    # A requested target is available even for an excluded/missing source row.
    # No-op controls can supply a source token only when its label was known.
    sparse[:, index] = True if value is not None else known
    if value is not None:
        edited[index] = 0
        edited[index, :, labels.index(value)] = 1
    return {
        'caption_embedding': edited,
        'caption_presence_mask': sparse,
        'caption_edit_mask': sparse.clone(),
        'eligible': eligible,
        'unchanged': unchanged,
        'missing': ~known,
        'field_index': index,
        'diagnostic_alias_fields': edit_alias_fields(field),
        'excluded_preservation_fields': (field,),
    }


def _prediction(tokens, vocabulary):
    valid = torch.isfinite(tokens).all(dim=-1) & (tokens.norm(dim=-1) > 0)
    safe = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
    cosine = functional.normalize(safe, dim=-1) @ functional.normalize(vocabulary, dim=-1).T
    return cosine.argmax(dim=-1), cosine, valid


def semantic_edit_totals(
    source_features, reconstruction_features, noop_features, edited_features,
    source_caption, source_presence, field, value, clip_embeddings, eligible,
):
    """Sum semantic edit/control outcomes using independent evaluator features.

    All token arrays use (F,B,D); presence uses (B,F). Only eligible changed
    examples are measured. Preservation uses observed high-level source labels
    with an exact vocabulary embedding, excluding only the requested field.
    Prediction agreement is reported separately from source-label accuracy.
    Invalid predictions receive their own rate and are excluded from accuracy
    denominators. Every metric returns (sum, count); no evidence means count 0.
    """
    source_caption = _tokens(source_caption, 'source_caption')
    fields, batch, width = source_caption.shape
    device = source_caption.device
    presence = _presence(source_presence, batch, fields, device, 'source_presence')
    if (not torch.is_tensor(eligible) or eligible.dtype != torch.bool
            or eligible.shape != (batch,)):
        raise ValueError('eligible must be boolean [batch]')
    eligible = eligible.to(device)
    index, kind = _field(field)
    labels, vocabulary = _vocabulary(kind, clip_embeddings, width, device, torch.float64)
    value = _value(value)
    target_available = value in labels
    native_unavailable = (field == 'movement_type' and value in ('nativePan', 'nativeOrbit')
                          and value in CATEGORY_VALUES[index] and not target_available)
    if not target_available and not native_unavailable:
        raise ValueError(f"Unknown {field} edit value {value!r}")
    target = labels.index(value) if target_available else None
    stages = {
        name: _tokens(tokens, f'{name}_features', source_caption.shape).detach().to(device=device, dtype=torch.float64)
        for name, tokens in (
            ('source', source_features), ('reconstruction', reconstruction_features),
            ('noop', noop_features), ('edited', edited_features),
        )
    }
    totals = {}

    def add(name, values, mask):
        selected = values.masked_select(mask)
        previous_sum, previous_count = totals.get(name, (0.0, 0))
        totals[name] = (previous_sum + float(selected.double().sum().item()), previous_count + selected.numel())

    target_outcomes = {}
    for name, tokens in stages.items():
        prediction, cosine, valid = _prediction(tokens[index], vocabulary)
        if target_available:
            correct, similarity = prediction == target, cosine[:, target]
            target_valid = valid
        else:
            # These two generator categories postdate the fixed metric
            # vocabulary. They have no semantic accuracy/cosine measurement;
            # do not fabricate a CLIP vector or classify them as bad features.
            correct = torch.zeros_like(valid)
            similarity = cosine.new_zeros(batch)
            target_valid = torch.zeros_like(valid)
        target_outcomes[name] = (correct, similarity, target_valid)
        add(f'target_{name}_top1', correct, eligible & target_valid)
        add(f'target_{name}_cosine', similarity, eligible & target_valid)
        add(f'target_{name}_invalid_feature_rate', ~valid,
            eligible if target_available else torch.zeros_like(eligible))
    edited_correct, edited_similarity, edited_valid = target_outcomes['edited']
    for name in ('source', 'reconstruction', 'noop'):
        correct, similarity, valid = target_outcomes[name]
        pair = eligible & edited_valid & valid
        add(f'target_top1_gain_vs_{name}', edited_correct.double() - correct.double(), pair)
        add(f'target_cosine_gain_vs_{name}', edited_similarity - similarity, pair)

    # Pre-create the keys so entirely empty/missing cohorts remain explicit.
    for name in stages:
        totals[f'unedited_{name}_top1'] = (0.0, 0)
        totals[f'unedited_{name}_invalid_feature_rate'] = (0.0, 0)
    for name in ('source', 'reconstruction', 'noop'):
        totals[f'unedited_agreement_vs_{name}'] = (0.0, 0)
    for preserve_field in EDITABLE_FIELDS:
        if preserve_field == field:
            continue
        slot = FIELD_NAMES.index(preserve_field)
        _, vocabulary = _vocabulary(_FIELD_TYPES[preserve_field], clip_embeddings, width, device, torch.float64)
        caption = source_caption[slot].detach().to(dtype=torch.float64)
        source_label, _, caption_valid = _prediction(caption, vocabulary)
        # Presence alone cannot turn an arbitrary/missing fill into a label.
        # Permit floating-point storage conversion (including float16), while
        # still requiring a vocabulary token rather than its nearest label.
        exact = torch.isclose(caption, vocabulary[source_label], rtol=1e-3, atol=1e-6).all(dim=-1)
        observed = eligible & presence[:, slot] & caption_valid & exact
        predictions = {}
        for name, tokens in stages.items():
            prediction, _, valid = _prediction(tokens[slot], vocabulary)
            predictions[name] = (prediction, valid)
            add(f'unedited_{name}_top1', prediction == source_label, observed & valid)
            add(f'unedited_{name}_invalid_feature_rate', ~valid, observed)
        edited_prediction, edited_valid = predictions['edited']
        for name in ('source', 'reconstruction', 'noop'):
            prediction, valid = predictions[name]
            add(f'unedited_agreement_vs_{name}', edited_prediction == prediction, observed & edited_valid & valid)
    return totals


def _rotation_distance_deg(left, right):
    relative = np.swapaxes(left, -1, -2) @ right
    sine = np.linalg.norm(np.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
    ), axis=-1), axis=-1) / 2
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.rad2deg(np.arctan2(sine, np.clip(cosine, -1, 1)))


def edit_geometry_totals(
    edited, reconstruction, source, *, padding_mask, subject_trajectory=None,
    subject_dimensions=None, intrinsics=None,
):
    """World-space camera/framing drift and motion preservation diagnostics.

    Inputs are denormalized simulator poses (B,T,6), camera-to-world Euler XYZ
    radians. True padding is excluded, including from motion differences. The
    initial-frame bbox delta is signed (edited minus reconstruction), using the
    first real frame and requiring valid shared calibration and both boxes in
    front of the near plane. It supplies no categorical close-up success label.
    """
    edited, reconstruction, source = map(_array, (edited, reconstruction, source))
    if (edited.ndim != 3 or edited.shape[-1] != 6
            or reconstruction.shape != edited.shape or source.shape != edited.shape):
        raise ValueError('edited, reconstruction and source must have matching (B,T,6) shapes')
    padding = _array(padding_mask, bool)
    if padding.shape != edited.shape[:2]:
        raise ValueError('padding_mask must have shape (B,T)')
    valid = ~padding
    geometry = dict(
        valid_mask=valid, subject_trajectory=subject_trajectory,
        subject_dimensions=subject_dimensions, intrinsics=intrinsics,
    )
    totals = {}
    for name, reference in (('source', source), ('reconstruction', reconstruction)):
        for metric, total in geometry_metric_totals(edited, reference, **geometry).items():
            totals[f'drift_vs_{name}_{metric}'] = total

    finite = np.isfinite(edited).all(axis=-1) & np.isfinite(reconstruction).all(axis=-1)
    pair_valid = valid & finite
    clean_edited = np.where(np.isfinite(edited), edited, 0.0)
    clean_reconstruction = np.where(np.isfinite(reconstruction), reconstruction, 0.0)
    edited_rotation = _xyz_matrix(clean_edited[..., 3:])
    reconstructed_rotation = _xyz_matrix(clean_reconstruction[..., 3:])

    def add(name, values, mask):
        selected = np.asarray(values)[mask]
        totals[name] = (float(selected.sum()), int(selected.size))

    adjacent = pair_valid[:, 1:] & pair_valid[:, :-1]
    residual = clean_edited[..., :3] - clean_reconstruction[..., :3]
    step_drift = np.linalg.norm(residual[:, 1:] - residual[:, :-1], axis=-1)
    add('reconstruction_step_drift', step_drift, adjacent)
    edited_step = np.swapaxes(edited_rotation[:, :-1], -1, -2) @ edited_rotation[:, 1:]
    reconstructed_step = np.swapaxes(reconstructed_rotation[:, :-1], -1, -2) @ reconstructed_rotation[:, 1:]
    add('reconstruction_rotation_step_drift_deg', _rotation_distance_deg(reconstructed_step, edited_step), adjacent)

    relative_valid = np.zeros_like(valid)
    relative_error = np.zeros(valid.shape, dtype=np.float64)
    first = np.zeros(len(edited), dtype=int)
    if edited.shape[1]:
        first = valid.argmax(axis=1)
        for row, initial in enumerate(first):
            if not pair_valid[row, initial]:
                continue
            edited_relative = edited_rotation[row, initial].T @ edited_rotation[row]
            reconstructed_relative = reconstructed_rotation[row, initial].T @ reconstructed_rotation[row]
            relative_error[row] = _rotation_distance_deg(reconstructed_relative, edited_relative)
            relative_valid[row] = pair_valid[row]
            relative_valid[row, initial] = False
    add('reconstruction_relative_rotation_drift_deg', relative_error, relative_valid)

    for name in ('projection_initial_edited_bbox_height', 'projection_initial_reconstruction_bbox_height',
                 'projection_initial_bbox_height_delta_vs_reconstruction'):
        totals[name] = (0.0, 0)
    if subject_trajectory is None or subject_dimensions is None or intrinsics is None:
        return totals
    subject = _frame_array(subject_trajectory, edited.shape[:2], 6, 'subject_trajectory')
    dimensions = _frame_array(subject_dimensions, edited.shape[:2], 3, 'subject_dimensions')
    calibration = _frame_array(intrinsics, edited.shape[:2], 2, 'intrinsics')
    for row, initial in enumerate(first):
        if not edited.shape[1] or not pair_valid[row, initial]:
            continue
        frame = (slice(row, row + 1), slice(initial, initial + 1))
        height = []
        for trajectory in (edited, reconstruction):
            measurement = geometry_metric_totals(
                trajectory[frame], trajectory[frame], subject_trajectory=subject[frame],
                subject_dimensions=dimensions[frame], intrinsics=calibration[frame],
            )['subject_bbox_height']
            height.append(measurement)
        if not all(count == 1 for _, count in height):
            continue
        for name, value in (
            ('projection_initial_edited_bbox_height', height[0][0]),
            ('projection_initial_reconstruction_bbox_height', height[1][0]),
            ('projection_initial_bbox_height_delta_vs_reconstruction', height[0][0] - height[1][0]),
        ):
            previous_sum, count = totals[name]
            totals[name] = (previous_sum + value, count + 1)
    return totals
