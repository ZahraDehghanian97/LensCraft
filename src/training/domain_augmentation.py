"""Training views with the subject information available in ET and CCDM.

The input and output use the model's normalized simulation representation.
Geometry is changed in physical, standard coordinates before renormalization.
CCDM changes the target camera together with its subject frame; it does not
place target camera information in the subject condition. Native evaluation
adapters and their captions are deliberately outside this module.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import torch

from data.ccdm.convertor import CCDMConvertor
from data.simulation.convertor import SIMConvertor
from data.simulation.dataset import SimulationDataset
from data.simulation.utils import (
    STRUCTURED_PARAMETERS, cinematography_struct_parameters,
)
from data.categorical_conditioning import cinematography_ids_from_batch


@dataclass
class DomainAugmentationConfig:
    enabled: bool = False
    et_probability: float = 0.25
    ccdm_probability: float = 0.25
    fill: str = 'zero'
    et_coordinate_frame: str = 'world'
    ccdm_static_motion: bool = False
    ccdm_native_manifold: bool = False

    def __post_init__(self):
        for name in ('et_probability', 'ccdm_probability'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        if self.et_probability + self.ccdm_probability >= 1:
            raise ValueError('Domain probabilities must sum to less than 1 to retain original examples')
        if self.fill != 'zero':
            raise ValueError('Missing categorical fields use zero fill')
        if self.et_coordinate_frame not in ('world', 'first_camera'):
            raise ValueError('ET coordinate frame must be world or first_camera')
        for name in ('ccdm_static_motion', 'ccdm_native_manifold'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')


class DomainAugmentation:
    """Mix original, ET-information, and CCDM-subject-relative training views.

    ``domain_augmentation_view`` in the returned batch records 0 (original),
    1 (ET), or 2 (CCDM). ``force_domain`` is for deterministic validation and
    regression tests. Explicit sample seeds never advance the training RNG.
    """

    VIEW_IDS = {'original': 0, 'et': 1, 'ccdm': 2}
    DEFAULT_SUBJECT_VOLUME = (0.5, 1.7, 0.3)
    # Full camera-frame canonicalization does not preserve the simulator's
    # gravity-based pedestal, crane or arc axes. Keep camera-local motions;
    # follow/track are omitted conservatively when subject heading is hidden.
    CAMERA_LOCAL_MOVEMENTS = frozenset({
        'static', 'dollyIn', 'dollyOut', 'truckLeft', 'truckRight',
        'panLeft', 'panRight', 'tiltUp', 'tiltDown', 'dutchLeft', 'dutchRight',
    })
    # With a stationary upright subject the frame change is one yaw/translation,
    # so gravity-based pedestal, arc and crane motions also keep their meaning.
    # Follow/track describe inherited subject motion; Dutch is not stored in 5-D.
    CCDM_STATIC_MOVEMENTS = (CAMERA_LOCAL_MOVEMENTS - {'dutchLeft', 'dutchRight'}) | {
        'pedestalUp', 'pedestalDown', 'arcLeft', 'arcRight', 'craneUp', 'craneDown',
    }
    # These describe the camera's position path about a fixed subject/gravity
    # axis. Roll removal leaves that complete path and its speed unchanged.
    CCDM_POSITION_MOVEMENTS = frozenset({
        'dollyIn', 'dollyOut', 'pedestalUp', 'pedestalDown',
        'arcLeft', 'arcRight', 'craneUp', 'craneDown',
    })
    CCDM_POSITION_MOTION_FIELDS = frozenset({
        'movement_type', 'movement_speed', 'dynamic_easing',
    })
    CCDM_STATIC_MOTION_FIELDS = frozenset({
        'movement_type', 'movement_speed', 'dynamic_type', 'dynamic_easing',
        'dynamic_scale', 'dynamic_direction', 'dynamic_movementMode',
        'constraints_staticDistance', 'constraints_staticCameraSubjectRotation',
        'constraints_maxSpeed', 'constraints_maxAccelerate',
    })

    def __init__(self, config):
        self.config = DomainAugmentationConfig(**dict(config))
        self.field_names = [name for name, _ in STRUCTURED_PARAMETERS]
        self.field_keep = self._field_keep()

    def _field_keep(self):
        keep = torch.ones(3, len(self.field_names), dtype=torch.bool)
        for index, name in enumerate(self.field_names):
            # The simulator rotates the spherical camera-angle/view offset by
            # the subject orientation (setup/angles.ts). Its framing uses the
            # projected bounding box, including dimension-dependent outer
            # positions (setup/framing.ts). Default orientation/size therefore
            # no longer support these endpoint or visibility claims.
            if (any(field in name for field in ('cameraAngle', 'subjectView', 'shotSize', 'subjectFraming')) or name in {
                'constraints_allFramesVisibility',
                'constraints_staticCameraSubjectRotation',
                'dynamic_subjectAwareInterpolation',
            }):
                keep[1, index] = False

            # A different rigid frame at every time step changes all world
            # motion, speed, easing, rotation-lock and acceleration claims.
            # Subject-local angular endpoints survive the coherent frame
            # change, together with the setup endpoint selector. Framing and
            # shot size depend on the replaced bounding-box dimensions; the
            # projected subject origin alone does not preserve those labels.
            keep[2, index] = (
                name == 'setup_kind' or name.endswith(('_subjectView', '_cameraAngle'))
            )
        return keep

    def sample_views(self, batch_size, *, sample_seeds=None, force_domain=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError('Domain augmentation batch size must be a positive integer')
        if force_domain is not None and force_domain not in self.VIEW_IDS:
            raise ValueError('force_domain must be original, et, or ccdm')
        if sample_seeds is not None:
            if len(sample_seeds) != batch_size or any(
                isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63
                for seed in sample_seeds
            ):
                raise ValueError('One integer domain seed in [0, 2**63) is required per sample')
        if not self.config.enabled:
            return torch.zeros(batch_size, dtype=torch.long)
        if force_domain is not None:
            return torch.full((batch_size,), self.VIEW_IDS[force_domain], dtype=torch.long)
        draws = (
            torch.tensor([torch.rand((), generator=torch.Generator().manual_seed(int(seed))).item()
                          for seed in sample_seeds])
            if sample_seeds is not None else torch.rand(batch_size)
        )
        views = torch.zeros(batch_size, dtype=torch.long)
        views[draws < self.config.et_probability] = 1
        views[(draws >= self.config.et_probability)
              & (draws < self.config.et_probability + self.config.ccdm_probability)] = 2
        return views

    def apply(self, batch, *, sample_seeds=None, force_domain=None):
        camera = batch['camera_trajectory']
        if camera.ndim != 3 or camera.shape[-1] != 6:
            raise ValueError('Domain augmentation needs simulation [batch, frames, 6] poses')
        views = self.sample_views(len(camera), sample_seeds=sample_seeds, force_domain=force_domain)
        result = dict(batch)
        result['domain_augmentation_view'] = views.to(camera.device)
        if not views.any():
            return result
        if not batch.get('simulation_normalized', True):
            raise ValueError('Domain augmentation requires normalized simulation features')
        subject, volume = batch.get('subject_trajectory'), batch.get('subject_volume')
        if subject is None or subject.shape != camera.shape:
            raise ValueError('Domain augmentation needs matching simulation subject poses')
        if volume is None or volume.shape not in ((len(camera), 1, 3), (len(camera), 3)):
            raise ValueError('Domain augmentation needs [batch, 1, 3] or [batch, 3] subject volume')
        padding = batch.get('padding_mask')
        if padding is None:
            padding = torch.zeros(camera.shape[:2], dtype=torch.bool, device=camera.device)
        elif padding.dtype != torch.bool or padding.shape != camera.shape[:2]:
            raise ValueError('Domain augmentation padding must be boolean [batch, frames]')
        padding = padding.to(camera.device)

        # Sanitizing padded frames before trigonometry handles even NaN padding
        # without touching their original values in the returned batch.
        safe_camera = torch.where(padding[..., None], 0., camera)
        safe_subject = torch.where(padding[..., None], 0., subject)
        with torch.autocast(device_type=camera.device.type, enabled=False):
            dtype = torch.float64 if camera.dtype == torch.float64 else torch.float32
            raw_camera, raw_subject, _ = SimulationDataset.normalize_item(
                safe_camera.to(dtype), safe_subject.to(dtype), volume.to(dtype), False,
            )
            convertor = SIMConvertor()
            camera_transform, subject_transform, _ = convertor.to_standard(raw_camera, raw_subject)
            relative_camera = CCDMConvertor.to_native_subject_frame(camera_transform, subject_transform)
            native_camera = None
            if self.config.ccdm_native_manifold:
                # CCDM stores relative position and the projected subject origin,
                # not roll. Use exactly the native 5-D decoder's target manifold
                # while leaving the implicit subject independent of this target.
                native_convertor = CCDMConvertor()
                native_features = native_convertor.transform_to_ccdm(relative_camera)
                native_camera = native_convertor.convert_ccdm_to_transform(native_features)
            et = (views == 1).to(camera.device)
            ccdm = (views == 2).to(camera.device)
            selected = et | ccdm

            augmented_camera = torch.where(
                ccdm[:, None, None, None],
                native_camera if native_camera is not None else relative_camera,
                camera_transform,
            )
            augmented_subject = subject_transform.clone()
            if self.config.et_coordinate_frame == 'first_camera':
                # Training-only scene re-expression. Native inference never
                # reads a reference camera to construct its subject condition.
                # R_std = R_GL @ F, so Q = inverse(R_GL0) = F @ R_std0.T.
                # The first GL camera becomes identity; the first subject
                # position is the common origin. Every camera/subject point
                # receives the SAME rigid transform, preserving optical rays.
                first = (~padding).to(torch.int64).argmax(dim=1)
                rows = torch.arange(len(camera), device=camera.device)
                flip = camera_transform.new_tensor(SIMConvertor._GL2CV)
                rotation = flip[:, None] * camera_transform[rows, first, :3, :3].transpose(-1, -2)
                origin = subject_transform[rows, first, :3, 3]
                for transformed, original in (
                    (augmented_camera, camera_transform),
                    (augmented_subject, subject_transform),
                ):
                    local_rotation = rotation[:, None] @ original[..., :3, :3]
                    local_position = (rotation[:, None] @ (
                        original[..., :3, 3] - origin[:, None]
                    )[..., None]).squeeze(-1)
                    transformed[..., :3, :3] = torch.where(
                        et[:, None, None, None], local_rotation, transformed[..., :3, :3],
                    )
                    transformed[..., :3, 3] = torch.where(
                        et[:, None, None], local_position, transformed[..., :3, 3],
                    )
            identity = torch.eye(3, dtype=dtype, device=camera.device)
            augmented_subject[..., :3, :3] = torch.where(
                et[:, None, None, None], identity, augmented_subject[..., :3, :3],
            )
            augmented_subject[..., :3, :3] = torch.where(
                ccdm[:, None, None, None], CCDMConvertor.native_subject_rotation(camera_transform),
                augmented_subject[..., :3, :3],
            )
            augmented_subject[..., :3, 3] = torch.where(
                ccdm[:, None, None], 0., augmented_subject[..., :3, 3],
            )
            default_volume = camera_transform.new_tensor(self.DEFAULT_SUBJECT_VOLUME).expand(len(camera), 3)
            raw_camera, raw_subject, raw_volume = convertor.from_standard(
                augmented_camera, augmented_subject, default_volume,
            )
            aug_camera, aug_subject, aug_volume = SimulationDataset.normalize_item(
                raw_camera, raw_subject, raw_volume, True,
            )
            motion_rows = None
            if self.config.ccdm_static_motion:
                motion_rows = self._ccdm_static_motion_rows(
                    subject_transform, relative_camera, native_camera, padding,
                )
        camera_changed = ccdm | (et & (self.config.et_coordinate_frame == 'first_camera'))
        result['camera_trajectory'] = torch.where(
            (camera_changed[:, None] & ~padding)[..., None], aug_camera.to(camera.dtype), camera,
        )
        result['subject_trajectory'] = torch.where(
            (selected[:, None] & ~padding)[..., None], aug_subject.to(subject.dtype), subject,
        )
        if self.config.et_coordinate_frame == 'world':
            # Legacy ET views do not change subject positions. Avoid an
            # unnecessary normalization round trip in the default recipe.
            result['subject_trajectory'][..., :3] = torch.where(
                et[:, None, None], subject[..., :3], result['subject_trajectory'][..., :3],
            )
        if volume.ndim == 2:
            aug_volume = aug_volume.squeeze(1)
        result['subject_volume'] = torch.where(
            selected.reshape(len(camera), *([1] * (volume.ndim - 1))), aug_volume.to(volume.dtype), volume,
        )
        self._apply_caption_mask(result, batch, views, ccdm_motion_rows=motion_rows)
        return result

    @staticmethod
    def _ccdm_static_motion_rows(subject, relative_camera, native_camera, padding):
        """Check invariants on SIM targets only, ignoring even NaN padding.

        A time-varying subject frame can turn a moving camera into a static
        native trajectory. A fixed tilted frame changes gravity-based labels.
        Native roll removal can change camera-local directions and rotation
        easing. Return its compatibility separately so position-path captions
        can survive without asserting unchanged rotation motions.
        """
        valid = ~padding
        first = valid.to(torch.int64).argmax(dim=1)
        rows = torch.arange(len(subject), device=subject.device)
        anchor = subject[rows, first]
        constant_pose = ((subject - anchor[:, None]).abs() <= 1e-5).flatten(2).all(-1)
        constant_pose = (constant_pose | padding).all(-1) & (valid.sum(-1) >= 2)
        up = subject.new_tensor([0., 1., 0.])
        upright = ((anchor[..., :3, :3] @ up - up).abs() <= 1e-5).all(-1)
        if native_camera is None:
            unchanged_rotation = torch.ones_like(constant_pose)
        else:
            same = ((native_camera[..., :3, :3] - relative_camera[..., :3, :3]).abs()
                    <= 1e-4).flatten(2).all(-1)
            unchanged_rotation = (same | padding).all(-1)
        return constant_pose & upright, unchanged_rotation

    def _ccdm_static_field_keep(self, batch, views, keep, motion_rows):
        keep = keep.clone()
        static_frame, unchanged_rotation = motion_rows
        metadata = batch.get('cinematography_prompt_parameters')
        if (views == 2).any() and metadata is None:
            raise ValueError('CCDM static motion views require cinematography_prompt_parameters; '
                             'set data.compact_batches=false')
        movement_index = self.field_names.index('movement_type')
        indices = [i for i, name in enumerate(self.field_names)
                   if name in self.CCDM_STATIC_MOTION_FIELDS]
        positional_indices = [i for i, name in enumerate(self.field_names)
                              if name in self.CCDM_POSITION_MOTION_FIELDS]
        for row in ((views == 2).to(static_frame.device) & static_frame).nonzero(
                as_tuple=False).flatten().tolist():
            movement = None
            if len(metadata) == len(views) and len(metadata[row]) > movement_index:
                entry = metadata[row][movement_index]
                if len(entry) >= 2 and entry[0] == 'movement_type':
                    movement = getattr(entry[1], 'value', entry[1])
            if movement in self.CCDM_STATIC_MOVEMENTS and unchanged_rotation[row]:
                keep[row, indices] = True
            elif movement in self.CCDM_POSITION_MOVEMENTS:
                # Keep high-level position-path motion and its easing, while
                # omitting simulator direction/mode and rotation constraints.
                keep[row, positional_indices] = True
        return keep

    def _first_camera_field_keep(self, batch, views, keep):
        """Invalidate gravity/Euler-axis claims using each training annotation.

        Raw training annotation values are needed to distinguish camera-local
        motions from gravity-based ones; compact batches discard those values.
        No native captions or test-derived masks are needed.
        """
        keep = keep.clone()
        metadata = batch.get('cinematography_prompt_parameters')
        if (views == 1).any() and metadata is None:
            raise ValueError('First-camera ET views require cinematography_prompt_parameters; '
                             'set data.compact_batches=false')
        movement_index = self.field_names.index('movement_type')
        aliases = [self.field_names.index(name) for name in (
            'movement_type', 'dynamic_direction', 'dynamic_movementMode',
        )]
        locks = [i for i, name in enumerate(self.field_names)
                 if name.startswith(('constraints_lockedMovement_', 'constraints_lockedRotation_'))]
        for row in (views == 1).nonzero(as_tuple=False).flatten().tolist():
            movement = None
            if metadata is not None and len(metadata) == len(views) and len(metadata[row]) > movement_index:
                entry = metadata[row][movement_index]
                if len(entry) >= 2 and entry[0] == 'movement_type':
                    movement = getattr(entry[1], 'value', entry[1])
            if movement not in self.CAMERA_LOCAL_MOVEMENTS:
                keep[row, aliases] = False
            keep[row, locks] = False
        return keep

    def _apply_caption_mask(self, result, batch, views, *, ccdm_motion_rows=None):
        ids = cinematography_ids_from_batch(batch).to(result['camera_trajectory'].device)
        split = len(cinematography_struct_parameters)
        supplied = batch.get('prompt_none_mask')
        if supplied is not None:
            if supplied.dtype != torch.bool or supplied.shape != ids.shape:
                raise ValueError('Caption presence must be boolean [batch,10]')
            ids = ids.masked_fill(~supplied.to(ids.device), -1)
        present = torch.zeros(len(views), len(self.field_names), dtype=torch.bool, device=ids.device)
        for key, start in (('cinematography_prompt_parameters', 0),
                           ('simulation_instruction_parameters', split)):
            metadata = batch.get(key)
            if metadata is not None:
                if len(metadata) != len(views):
                    raise ValueError('Domain annotation metadata must match the batch')
                for row, entries in enumerate(metadata):
                    for offset, entry in enumerate(entries):
                        present[row, start + offset] = entry[2] >= 0
        present[:, :split] = ids >= 0
        keep = self.field_keep[views].to(ids.device)
        if self.config.et_coordinate_frame == 'first_camera':
            keep = self._first_camera_field_keep(batch, views, keep)
        if self.config.ccdm_static_motion:
            keep = self._ccdm_static_field_keep(batch, views, keep, ccdm_motion_rows)
        retained = present & keep
        result['cinematography_ids'] = ids.masked_fill(~retained[:, :split], -1)
        result['prompt_none_mask'] = retained[:, :split]
        result['structured_presence_mask'] = retained
        result.pop('cinematography_prompt', None)
        result.pop('simulation_instruction', None)
        # Raw simulator metadata remains available for geometry certification;
        # only the first ten retained categorical IDs condition the model.
        dropped = (present & ~keep).cpu().tolist()
        for key, start in (('cinematography_prompt_parameters', 0),
                           ('simulation_instruction_parameters', split)):
            if key in batch:
                result[key] = [[(entry[0], None, -1, None) if dropped[row][start + offset]
                                else entry for offset, entry in enumerate(entries)]
                               for row, entries in enumerate(batch[key])]
