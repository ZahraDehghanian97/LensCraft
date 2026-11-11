"""LensCraft camera trajectories with one ten-field categorical bottleneck."""
from typing import Optional, Dict

import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .categorical_latent import (
    FIELD_COUNT, TOKEN_WIDTH, SEMANTIC_DIM, categorical_presence, validate_semantic,
)
from utils.pytorch3d_transform import (
    euler_angles_to_matrix, matrix_to_euler_angles, symmetric_orthogonalization,
)


class LensCraft(nn.Module):
    """Encode and decode ten independent categorical probability distributions.

    Model states have 256 channels by default; the sole camera-conditioning
    interface has ten 21-channel tokens with 93 valid categorical entries.
    Missing fields remain zero and impose no decoder constraint.
    """

    def __init__(
        self,
        input_dim: int = 6,
        subject_dim: int = 6,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout_rate: float = 0.1,
        seq_length: int = 30,
        latent_dim: int = 256,
        keyframe_pose_conditioning: bool = False,
    ):
        super().__init__()
        self.pos_dim = 3
        self.decoder_output_dim = 12
        self.semantic_dim = SEMANTIC_DIM
        self.semantic_token_width = TOKEN_WIDTH
        self.num_query_tokens = FIELD_COUNT
        self.memory_tokens_count = FIELD_COUNT
        self.keyframe_pose_conditioning = keyframe_pose_conditioning
        self.subject_trajectory_projection = nn.Linear(subject_dim, latent_dim)
        self.subject_volume_projection = nn.Linear(3, latent_dim)
        self.encoder = Encoder(
            input_dim, latent_dim, nhead, num_encoder_layers,
            dim_feedforward, dropout_rate, FIELD_COUNT,
        )
        self.decoder = Decoder(
            self.decoder_output_dim, latent_dim, nhead, num_decoder_layers,
            dim_feedforward, dropout_rate, seq_length,
        )

    def _euler6_to_mat12(self, traj6: torch.Tensor) -> torch.Tensor:
        pos = traj6[..., :3]
        rot = euler_angles_to_matrix(traj6[..., 3:6], "XYZ")
        return torch.cat([pos, rot.reshape(*traj6.shape[:-1], 9)], dim=-1)

    def _project_svd9d(self, raw12: torch.Tensor) -> torch.Tensor:
        pos = raw12[..., :3]
        mat = raw12[..., 3:].reshape(*raw12.shape[:-1], 3, 3)
        rot = symmetric_orthogonalization(mat)
        return torch.cat([pos, rot.reshape(*raw12.shape[:-1], 9)], dim=-1)

    def _mat12_to_euler6(self, mat12: torch.Tensor) -> torch.Tensor:
        pos = mat12[..., :3]
        rot = mat12[..., 3:].reshape(*mat12.shape[:-1], 3, 3)
        return torch.cat([pos, matrix_to_euler_angles(rot, "XYZ")], dim=-1)

    def _encode_target_for_decoder(self, target):
        if target is None:
            return None
        return self._euler6_to_mat12(target)

    def _known_camera_pose_inputs(self, src, camera_source_mask, caption_embedding, memory_ratio):
        """Expose only supplied visible camera poses to the optional query path."""
        if not self.keyframe_pose_conditioning or (
            caption_embedding is not None and memory_ratio >= 1.0
        ):
            return {}
        known_mask = torch.ones(src.shape[:2], device=src.device, dtype=torch.bool)
        if camera_source_mask is not None:
            if camera_source_mask.shape != src.shape[:2]:
                raise ValueError("Camera source mask must match the source batch and frame dimensions")
            known_mask = ~camera_source_mask.to(device=src.device, dtype=torch.bool)
        # Sanitize BEFORE rotation conversion as hidden poses may contain NaN.
        supplied = src.masked_fill(~known_mask.unsqueeze(-1), 0)
        poses = self._euler6_to_mat12(supplied).masked_fill(~known_mask.unsqueeze(-1), 0)
        return {"known_camera_poses": poses, "known_pose_mask": known_mask}

    def _validate_pose_decode_mode(self, decode_mode):
        if self.keyframe_pose_conditioning and decode_mode != "single_step":
            raise ValueError(
                "keyframe_pose_conditioning currently supports only decode_mode='single_step'; "
                "disable the experiment for autoregressive decoding"
            )

    def _prepare_subject_inputs(
        self, reference, subject_trajectory, subject_volume
    ):
        batch_size = reference.shape[0]
        if subject_trajectory is None:
            subject_trajectory = reference.new_zeros(
                batch_size,
                reference.shape[1],
                self.subject_trajectory_projection.in_features,
            )
        if subject_volume is None:
            subject_volume = reference.new_zeros(batch_size, 1, 3)
        return subject_trajectory, subject_volume

    def _finalize_trajectory(self, raw):
        recon_matrix = self._project_svd9d(raw)        # valid SO(3), flattened
        reconstructed = self._mat12_to_euler6(recon_matrix)
        return reconstructed, recon_matrix

    def _camera_masks(self, src, source_mask, padding_mask, known_camera_mask):
        """Separate hidden source context from explicitly fixed camera poses.

        A supplied known-camera mask is authoritative: only those poses may
        enter the encoder, and temporal padding can never become a constraint.
        Without it, source visibility retains its usual reconstruction meaning.
        """
        def normalize(mask, name):
            if mask is None:
                return None
            if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                    or mask.shape != src.shape[:2]):
                raise ValueError(f"{name} must be boolean [batch, frames]")
            return mask.to(device=src.device)

        source_mask = normalize(source_mask, 'src_key_mask')
        padding = normalize(padding_mask, 'padding_mask')
        known = normalize(known_camera_mask, 'known_camera_mask')
        if padding is None:
            padding = torch.zeros(src.shape[:2], device=src.device, dtype=torch.bool)
        if known is None:
            known = torch.zeros_like(padding)
        else:
            known = known & ~padding
            if source_mask is not None and ((source_mask != ~known) & ~padding).any():
                raise ValueError("src_key_mask must agree with known_camera_mask on valid frames")
            source_mask = ~known
            supplied = src.masked_fill(~known.unsqueeze(-1), 0)
            if not torch.isfinite(supplied).all():
                raise ValueError("Known camera poses must contain only finite values")

        source_mask = padding if source_mask is None else source_mask | padding
        return source_mask, known

    def _constrain_trajectory(self, raw, reconstructed, rotation_matrix, src, known):
        """Pin each emitted representation, preserving the original Euler values.

        Merge after projection/conversion so equivalent Euler representations
        (including angles outside [-pi, pi]) stay exactly as supplied. Geometry
        losses see the same fixed poses as inference and cycle reconstruction.
        """
        supplied = src.masked_fill(~known.unsqueeze(-1), 0)
        # Autocast must not round a supplied rotation while forming its matrix.
        with torch.autocast(device_type=src.device.type, enabled=False):
            supplied_matrix = self._euler6_to_mat12(supplied)
        mask = known.unsqueeze(-1)
        return (
            torch.where(mask, supplied_matrix, raw),
            torch.where(mask, supplied, reconstructed),
            torch.where(mask, supplied_matrix, rotation_matrix),
        )

    def prepare_embedding_memory_for_decoder(
        self,
        camera_embedding: Optional[torch.Tensor] = None,
        caption_embedding: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        mask_memory_prob: float = 0.0,
        caption_presence_mask: Optional[torch.Tensor] = None,
        caption_edit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compose semantic probabilities without adding attention field codes.

        Unknown caption fields provide no constraint. A partial blend retains
        available camera information there; caption-only generation excludes
        them entirely. Field identity is added only inside the decoder.
        """
        if not 0 <= teacher_forcing_ratio <= 1 or not 0 <= mask_memory_prob <= 1:
            raise ValueError("Categorical mixing and masking ratios must be in [0,1]")
        reference = camera_embedding if camera_embedding is not None else caption_embedding
        if reference is None:
            raise ValueError("camera_embedding and caption_embedding cannot both be None")
        if (not torch.is_tensor(reference) or reference.ndim != 3
                or reference.shape[0] != FIELD_COUNT or reference.shape[-1] != TOKEN_WIDTH):
            raise ValueError("Categorical memory must have shape [10,batch,21]")
        batch_size = reference.shape[1]

        def field_mask(value, name):
            if (not torch.is_tensor(value) or value.dtype != torch.bool
                    or value.shape != (batch_size, FIELD_COUNT)):
                raise ValueError(f"{name} must be boolean [batch,10]")
            return value.to(device=reference.device)

        def semantic(value, presence=None):
            if (not torch.is_tensor(value) or value.shape != reference.shape
                    or value.device != reference.device):
                raise ValueError("Categorical memories must have matching shape and device")
            if presence is not None:
                # Unspecified fillers may contain NaNs; remove them before
                # validation or arithmetic, without modifying caller inputs.
                value = value.masked_fill(~presence.T[..., None], 0)
            validate_semantic(value, atol=(5e-3 if value.dtype in (torch.float16, torch.bfloat16) else 1e-5))
            return value

        caption_presence = None
        if caption_presence_mask is not None:
            caption_presence = field_mask(caption_presence_mask, "caption_presence_mask")

        if caption_edit_mask is not None:
            if camera_embedding is None or caption_embedding is None:
                raise ValueError("caption_edit_mask requires camera and caption memories")
            edits = field_mask(caption_edit_mask, "caption_edit_mask")
            if caption_presence is not None and (edits & ~caption_presence).any():
                raise ValueError("caption_edit_mask must be a subset of caption_presence_mask")
            camera = semantic(camera_embedding)
            caption = semantic(caption_embedding, edits)
            if (edits & ~categorical_presence(caption)).any():
                raise ValueError("Edited fields must contain a supplied categorical distribution")
            memory = torch.where(edits.T[..., None], caption, camera)
        elif camera_embedding is None or (caption_embedding is not None and teacher_forcing_ratio == 1):
            memory = semantic(caption_embedding, caption_presence)
        elif caption_embedding is None or teacher_forcing_ratio == 0:
            memory = semantic(camera_embedding)
        else:
            camera = semantic(camera_embedding)
            caption = semantic(caption_embedding, caption_presence)
            # Normalize by available-field mass so missing zero captions do
            # not turn a valid distribution into a field summing below one.
            camera_weight = (1 - teacher_forcing_ratio) * categorical_presence(camera).T[..., None]
            caption_weight = teacher_forcing_ratio * categorical_presence(caption).T[..., None]
            total_weight = camera_weight + caption_weight
            denominator = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
            memory = (camera_weight * camera + caption_weight * caption) / denominator
        if mask_memory_prob > 0:
            keep = torch.rand(FIELD_COUNT, device=memory.device) > mask_memory_prob
            memory = memory.masked_fill(~keep[:, None, None], 0)
        return memory

    def forward(
        self,
        src: torch.Tensor,
        subject_trajectory: torch.Tensor,
        subject_volume: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        src_key_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        caption_embedding: Optional[torch.Tensor] = None,
        memory_teacher_forcing_ratio: float = 0.5,
        trajectory_teacher_forcing_ratio: float = 0.0,
        mask_memory_prob: float = 0.0,
        decode_mode: str = 'single_step',
        caption_presence_mask: Optional[torch.Tensor] = None,
        known_camera_mask: Optional[torch.Tensor] = None,
        caption_edit_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        editing = caption_edit_mask is not None
        if editing:
            if src is None or caption_embedding is None:
                raise ValueError("caption_edit_mask requires source camera trajectory and caption_embedding")
            if known_camera_mask is not None:
                raise ValueError("caption_edit_mask cannot be combined with known_camera_mask")
        self._validate_pose_decode_mode(decode_mode)
        constrain_camera = known_camera_mask is not None
        camera_source_mask, known_camera_mask = self._camera_masks(
            src, src_key_mask, tgt_key_padding_mask, known_camera_mask
        )
        subject_trajectory, subject_volume = self._prepare_subject_inputs(
            src, subject_trajectory, subject_volume
        )
        if tgt_key_padding_mask is not None:
            subject_trajectory = subject_trajectory.masked_fill(
                tgt_key_padding_mask.to(device=subject_trajectory.device, dtype=torch.bool).unsqueeze(-1), 0
            )
        subject_trajectory_embedding = self.subject_trajectory_projection(
            subject_trajectory
        )
        subject_volume_embedding = self.subject_volume_projection(subject_volume)
        subject_embedding = torch.cat([subject_trajectory_embedding, subject_volume_embedding], 1)

        pose_inputs = {} if (
            editing or not constrain_camera
        ) else self._known_camera_pose_inputs(
            src, camera_source_mask, caption_embedding, memory_teacher_forcing_ratio
        )
        # Sanitize before projection/attention: even masked NaNs otherwise
        # contaminate the encoder, including when direct pose queries are off.
        encoder_src = src.masked_fill(camera_source_mask.unsqueeze(-1), 0)
        if constrain_camera and decode_mode == 'autoregressive':
            with torch.autocast(device_type=src.device.type, enabled=False):
                fixed_poses = self._euler6_to_mat12(encoder_src)
            pose_inputs = {
                'known_camera_poses': fixed_poses,
                'known_pose_mask': known_camera_mask,
            }
        camera_embedding, embedding_logits = self.encoder.encode_with_logits(
            encoder_src, subject_embedding, camera_source_mask,
            subject_key_padding_mask=tgt_key_padding_mask,
        )

        memory = self.prepare_embedding_memory_for_decoder(
            camera_embedding=camera_embedding.clone(),
            caption_embedding=caption_embedding,
            teacher_forcing_ratio=memory_teacher_forcing_ratio,
            mask_memory_prob=mask_memory_prob,
            caption_presence_mask=caption_presence_mask,
            caption_edit_mask=caption_edit_mask,
        )

        reconstructed_raw = self.decoder(
            memory=memory,
            subject_embedding=subject_embedding,
            decode_mode=decode_mode,
            target=(None if self.keyframe_pose_conditioning else self._encode_target_for_decoder(target)),
            teacher_forcing_ratio=trajectory_teacher_forcing_ratio,
            tgt_key_padding_mask=tgt_key_padding_mask,
            feedback_transform=self._project_svd9d,
            memory_presence_mask=categorical_presence(memory),
            **pose_inputs,
        )
        reconstructed, recon_matrix = self._finalize_trajectory(reconstructed_raw)
        if constrain_camera:
            reconstructed_raw, reconstructed, recon_matrix = self._constrain_trajectory(
                reconstructed_raw, reconstructed, recon_matrix, src, known_camera_mask
            )

        output = {
            'subject_embedding': subject_embedding,
            'embeddings': camera_embedding,
            'reconstructed': reconstructed,                 # euler from valid SO(3): cycle / inference
            'reconstructed_raw_matrix': reconstructed_raw,  # decoder matrices with exact keyframe constraints
            'reconstructed_rot_matrix': recon_matrix,       # projected SO(3) geometry for trajectory losses
            'known_camera_mask': known_camera_mask,
            'embedding_logits': embedding_logits,
        }

        if self.keyframe_pose_conditioning:
            output['known_pose_mask'] = pose_inputs.get(
                'known_pose_mask', torch.zeros(src.shape[:2], device=src.device, dtype=torch.bool)
            )

        return output

    def generate_camera_trajectory(
        self,
        caption_embedding: Optional[torch.Tensor] = None,
        camera_trajectory: Optional[torch.Tensor] = None,
        subject_trajectory: Optional[torch.Tensor] = None,
        subject_volume: Optional[torch.Tensor] = None,
        memory_teacher_forcing_ratio: float = 0.0,
        trajectory_teacher_forcing_ratio: float = 0.0,
        src_key_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        decode_mode: str = 'single_step',
        caption_presence_mask: Optional[torch.Tensor] = None,
        known_camera_mask: Optional[torch.Tensor] = None,
        caption_edit_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate poses, optionally fixing arbitrary full camera keyframes.

        ``known_camera_mask`` is boolean [batch, frames], with True at supplied
        poses in ``camera_trajectory``. Unknown poses may contain NaN fillers.
        This mask also controls source visibility; if ``src_key_mask`` is given
        it must agree (True means hidden there). Padding is never preserved.
        Omitting the known mask retains predictive reconstruction/text modes.

        ``caption_edit_mask`` selects caption fields per example, replacing
        only those slots of encoded ``camera_trajectory``. It overrides the
        scalar memory ratio.
        Unselected caption fillers are ignored, and direct pose conditioning
        is bypassed so source poses can change. Fixed keyframes are a separate
        mode and cannot be combined with an edit mask.
        """
        if caption_edit_mask is not None:
            if camera_trajectory is None or caption_embedding is None:
                raise ValueError("caption_edit_mask requires camera_trajectory and caption_embedding")
            if known_camera_mask is not None:
                raise ValueError("caption_edit_mask cannot be combined with known_camera_mask")
        self._validate_pose_decode_mode(decode_mode)
        if known_camera_mask is not None and camera_trajectory is None:
            raise ValueError("known_camera_mask requires camera_trajectory")
        with torch.no_grad():
            device = next(self.parameters()).device

            if subject_trajectory is None or subject_volume is None:
                if camera_trajectory is not None:
                    reference = camera_trajectory.to(device)
                elif caption_embedding is not None:
                    batch_size = caption_embedding.shape[1]
                    reference = caption_embedding.new_zeros(
                        batch_size, self.decoder.seq_length, self.pos_dim + 3
                    ).to(device)
                else:
                    raise ValueError(
                        "camera_trajectory or caption_embedding is required "
                        "when subject inputs are absent"
                    )
                subject_trajectory, subject_volume = self._prepare_subject_inputs(
                    reference, subject_trajectory, subject_volume
                )

            subject_trajectory = subject_trajectory.to(device)
            subject_volume = subject_volume.to(device)

            if caption_embedding is not None:
                caption_embedding = caption_embedding.to(device)
            elif camera_trajectory is None:
                raise ValueError(
                    "Both camera_trajectory and caption_embedding cannot be None"
                )

            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            # If camera trajectory is provided, use the full model
            if camera_trajectory is not None:
                camera_trajectory = camera_trajectory.to(device)
                if src_key_mask is not None:
                    src_key_mask = src_key_mask.to(device)

                return self.forward(
                    src=camera_trajectory,
                    subject_trajectory=subject_trajectory,
                    subject_volume=subject_volume,
                    caption_embedding=caption_embedding,
                    memory_teacher_forcing_ratio=memory_teacher_forcing_ratio,
                    trajectory_teacher_forcing_ratio=trajectory_teacher_forcing_ratio,
                    src_key_mask=src_key_mask,
                    tgt_key_padding_mask=padding_mask,
                    decode_mode=decode_mode,
                    caption_presence_mask=caption_presence_mask,
                    known_camera_mask=known_camera_mask,
                    caption_edit_mask=caption_edit_mask,
                )

            # If there is no camera trajectory, use only the decoder
            else:
                if padding_mask is not None:
                    subject_trajectory = subject_trajectory.masked_fill(
                        padding_mask.to(dtype=torch.bool).unsqueeze(-1), 0
                    )
                subject_trajectory_embedding = self.subject_trajectory_projection(
                    subject_trajectory
                )
                subject_volume_embedding = self.subject_volume_projection(
                    subject_volume
                )
                subject_embedding = torch.cat(
                    [subject_trajectory_embedding, subject_volume_embedding], 1
                )

                memory = self.prepare_embedding_memory_for_decoder(
                    caption_embedding=caption_embedding,
                    teacher_forcing_ratio=0.0,
                    caption_presence_mask=caption_presence_mask,
                )

                reconstructed_raw = self.decoder(
                    memory=memory,
                    subject_embedding=subject_embedding,
                    decode_mode=decode_mode,
                    tgt_key_padding_mask=padding_mask,
                    teacher_forcing_ratio=0.0,
                    feedback_transform=self._project_svd9d,
                    memory_presence_mask=categorical_presence(memory),
                )
                reconstructed, recon_matrix = self._finalize_trajectory(reconstructed_raw)

                output = {
                    'reconstructed': reconstructed,
                    'reconstructed_raw_matrix': reconstructed_raw,
                    'reconstructed_rot_matrix': recon_matrix,
                    'known_camera_mask': torch.zeros(
                        reconstructed.shape[:2], device=reconstructed.device, dtype=torch.bool
                    ),
                }
                if self.keyframe_pose_conditioning:
                    output['known_pose_mask'] = torch.zeros(
                        reconstructed.shape[:2], device=reconstructed.device, dtype=torch.bool
                    )
                return output

    def embed_trajectory(
        self,
        camera_trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor,
        subject_volume: torch.Tensor,
        src_key_mask: Optional[torch.Tensor] = None,
        subject_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            device = next(self.parameters()).device
            subject_trajectory, subject_volume = self._prepare_subject_inputs(
                camera_trajectory, subject_trajectory, subject_volume
            )
            effective_subject_mask = subject_key_padding_mask
            subject_trajectory = subject_trajectory.to(device)
            if effective_subject_mask is not None:
                subject_trajectory = subject_trajectory.masked_fill(
                    effective_subject_mask.to(device=device, dtype=torch.bool)[..., None], 0,
                )
            subject_embedding = torch.cat([
                self.subject_trajectory_projection(subject_trajectory),
                self.subject_volume_projection(subject_volume.to(device)),
            ], 1)
            camera_embedding = self.encoder(
                camera_trajectory.to(device),
                subject_embedding,
                src_key_mask,
                subject_key_padding_mask=effective_subject_mask,
            )
            return camera_embedding[:self.memory_tokens_count]
