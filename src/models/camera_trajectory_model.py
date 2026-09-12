from typing import Optional, Dict

import pickle
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from data.simulation.constants import NumericFeature
from data.simulation.utils import (
    CLIP_PARAMETERS,
    cinematography_struct_size,
    simulation_struct_size,
)
from utils.pytorch3d_transform import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    symmetric_orthogonalization,
)


class LensCraft(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        subject_dim: int = 6,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout_rate: float = 0.1,
        seq_length: int = 30,
        latent_dim: int = 512,
        use_merged_memory: bool = False,
        denormalize_memory: bool = False,
        camera_memory_norms=None,
        keyframe_pose_conditioning: bool = False,
    ):
        super(LensCraft, self).__init__()

        self.pos_dim = 3
        self.decoder_output_dim = self.pos_dim + 9   # 3 pos + 9D (svd9d) rotation matrix

        self.num_query_tokens = cinematography_struct_size + simulation_struct_size
        self.memory_tokens_count = self.num_query_tokens

        slot_position = torch.arange(self.num_query_tokens, dtype=torch.float32)
        slot_frequency = torch.exp(
            torch.arange(0, latent_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / latent_dim)
        )
        slot_phase = slot_position[:, None] * slot_frequency[None, :]
        slot_encoding = torch.zeros(
            self.num_query_tokens, 1, latent_dim, dtype=torch.float32
        )
        slot_encoding[:, 0, 0::2] = torch.sin(slot_phase)
        slot_encoding[:, 0, 1::2] = torch.cos(
            slot_phase[:, : slot_encoding[:, 0, 1::2].shape[-1]]
        )
        self.register_buffer(
            "structured_slot_encoding", slot_encoding, persistent=False
        )

        self.subject_trajectory_projection = nn.Linear(subject_dim, latent_dim)
        self.subject_volume_projection = nn.Linear(3, latent_dim)

        self.encoder = Encoder(
            input_dim,
            latent_dim,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout_rate,
            self.num_query_tokens
        )

        self.decoder = Decoder(
            self.decoder_output_dim,
            latent_dim,
            nhead,
            num_decoder_layers,
            dim_feedforward,
            dropout_rate,
            seq_length
        )

        self.embedding_merger = nn.Sequential(
            nn.Linear(self.num_query_tokens * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.use_merged_memory = use_merged_memory
        self.denormalize_memory = denormalize_memory
        self.keyframe_pose_conditioning = keyframe_pose_conditioning
        self.register_buffer("camera_memory_norms", None, persistent=False)
        self.set_camera_memory_norms(camera_memory_norms)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        if self.denormalize_memory:
            self.embedding_means, self.embedding_stds = self.load_means_and_stds()
        self.memory_feature_types = []
        for _, value_type in CLIP_PARAMETERS:
            if isinstance(value_type, NumericFeature):
                self.memory_feature_types.append(None)
            elif value_type is bool:
                self.memory_feature_types.append("boolean")
            else:
                self.memory_feature_types.append(value_type.__name__)

    def set_camera_memory_norms(self, norms):
        """Configure camera-to-decoder calibration; None restores raw memory."""
        if norms is None:
            self.camera_memory_norms = None
            return
        if self.use_merged_memory or self.denormalize_memory:
            raise ValueError(
                "Camera memory calibration requires unmerged, non-denormalized memory; "
                "use camera_memory_norms=null and inference.camera_memory_normalization=none "
                "for legacy merged/denormalized checkpoints"
            )
        norms = torch.as_tensor(norms, dtype=torch.float32,
                                device=self.structured_slot_encoding.device).detach().clone()
        if (norms.shape != (self.num_query_tokens,)
                or not torch.isfinite(norms).all() or (norms <= 0).any()):
            raise ValueError(
                f"camera_memory_norms must contain {self.num_query_tokens} finite positive values"
            )
        self.camera_memory_norms = norms

    def _calibrate_camera_memory(self, memory):
        if self.camera_memory_norms is None:
            return memory
        if memory.ndim != 3 or memory.shape[0] != self.num_query_tokens:
            raise ValueError("Camera memory calibration requires every structured token")
        values = memory.float()
        lengths = values.norm(dim=-1, keepdim=True)
        calibrated = values / lengths.clamp_min(1e-6)
        calibrated = calibrated * self.camera_memory_norms.float()[:, None, None]
        calibrated = torch.where(lengths > 1e-6, calibrated, torch.zeros_like(calibrated))
        return calibrated.to(dtype=memory.dtype)

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

    def prepare_embedding_memory_for_decoder(
        self,
        camera_embedding: Optional[torch.Tensor] = None,
        caption_embedding: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        mask_memory_prob: float = 0.0
    ) -> torch.Tensor:
        if caption_embedding is not None:
            caption_embedding = self._prepare_caption_tokens(caption_embedding)

        if camera_embedding is None:
            if caption_embedding is None:
                raise ValueError("camera_embedding and caption_embedding cannot both be None")
            memory = caption_embedding
            if self.use_merged_memory:
                memory = self._merge_token_memory(memory)
            if self.denormalize_memory:
                memory = self._denormalize_features(memory)
            if not self.use_merged_memory:
                # Slot identity is model-space information, so add it after
                # any feature-space denormalization rather than scaling it by
                # the statistics of the serialized value.
                memory = self._add_structured_slot_identity(memory)
            return self._apply_memory_mask(memory, mask_memory_prob)

        if caption_embedding is None or teacher_forcing_ratio < 1.0:
            camera_embedding = self._calibrate_camera_memory(camera_embedding)

        if self.use_merged_memory:
            memory = self._merge_token_memory(camera_embedding)
            if caption_embedding is not None:
                caption_embedding = self._merge_token_memory(caption_embedding)
        else:
            memory = camera_embedding[:self.memory_tokens_count]

        if teacher_forcing_ratio > 0 and caption_embedding is not None:
            memory = self._blend_caption(memory, caption_embedding, teacher_forcing_ratio)

        if not self.use_merged_memory:
            # Adding one shared slot code after blending preserves the exact
            # field identity for camera-only, caption-only, and mixed memory.
            memory = self._add_structured_slot_identity(memory)

        return self._apply_memory_mask(memory, mask_memory_prob)

    def _add_structured_slot_identity(self, memory):
        if memory.shape[0] == 1:
            # A single pooled free-text token has no structured field slot.
            return memory
        if memory.shape[0] != self.num_query_tokens:
            raise ValueError(
                f"Cannot add slot identity to {memory.shape[0]} tokens; "
                f"expected 1 or {self.num_query_tokens}"
            )
        return memory + self.structured_slot_encoding.to(
            device=memory.device, dtype=memory.dtype
        )

    def _prepare_caption_tokens(self, caption_embedding):
        token_count = caption_embedding.shape[0]
        if token_count == cinematography_struct_size:
            neutral = caption_embedding.new_zeros(
                self.num_query_tokens - token_count,
                *caption_embedding.shape[1:],
            )
            return torch.cat([caption_embedding, neutral], dim=0)
        if token_count not in (1, self.num_query_tokens):
            raise ValueError(
                f"Expected 1, {cinematography_struct_size}, or "
                f"{self.num_query_tokens} caption tokens; got {token_count}"
            )
        return caption_embedding

    def _merge_token_memory(self, memory):
        if memory.shape[0] == 1:
            return memory
        if memory.shape[0] != self.num_query_tokens:
            raise ValueError(
                f"Cannot merge {memory.shape[0]} tokens; expected "
                f"{self.num_query_tokens}"
            )
        batch_size = memory.shape[1]
        flat = memory.transpose(0, 1).reshape(batch_size, -1)
        return self.embedding_merger(flat).unsqueeze(0)

    def _denormalize_features(self, memory):
        memory = memory.clone()
        for i in range(memory.shape[0]):
            if i >= len(self.memory_feature_types):
                continue
            feature = self.memory_feature_types[i]
            if (
                feature is None
                or feature not in self.embedding_means
                or feature not in self.embedding_stds
            ):
                continue
            mean, std = self.get_mean_and_std(feature)
            memory[i] = memory[i] * std + mean
        return memory

    def _blend_caption(self, memory, caption_embedding, ratio):
        if self.denormalize_memory:
            memory = self._denormalize_features(memory)
            caption_embedding = self._denormalize_features(caption_embedding)

        if ratio >= 1.0:
            # Besides documenting the caption-only evaluation contract, this
            # avoids retaining a numerical/data dependency on ground-truth
            # camera memory through ``0 * memory``.
            return caption_embedding
        if ratio <= 0.0:
            return memory

        # A true convex blend makes the scheduled ratio deterministic and
        # gives validation the same conditioning objective as training.
        return (1.0 - ratio) * memory + ratio * caption_embedding

    def _apply_memory_mask(self, memory, mask_memory_prob):
        if mask_memory_prob <= 0.0:
            return memory
        keep = torch.rand(memory.shape[0], device=memory.device) > mask_memory_prob
        return memory * keep.float().unsqueeze(1).unsqueeze(2)

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
    ) -> Dict[str, torch.Tensor]:
        self._validate_pose_decode_mode(decode_mode)
        subject_trajectory, subject_volume = self._prepare_subject_inputs(
            src, subject_trajectory, subject_volume
        )
        if self.keyframe_pose_conditioning and tgt_key_padding_mask is not None:
            subject_trajectory = subject_trajectory.masked_fill(
                tgt_key_padding_mask.to(device=subject_trajectory.device, dtype=torch.bool).unsqueeze(-1), 0
            )
        subject_trajectory_embedding = self.subject_trajectory_projection(
            subject_trajectory
        )
        subject_volume_embedding = self.subject_volume_projection(subject_volume)
        subject_embedding = torch.cat([subject_trajectory_embedding, subject_volume_embedding], 1)

        # Temporal padding is never valid source-camera context.  Treat a
        # caller-provided source mask as an additional mask (for example,
        # keyframing), while keeping the subject stream masked only by real
        # temporal padding.
        camera_source_mask = src_key_mask
        if tgt_key_padding_mask is not None:
            padding_mask = tgt_key_padding_mask.to(
                device=src.device, dtype=torch.bool
            )
            camera_source_mask = (
                padding_mask
                if camera_source_mask is None
                else camera_source_mask.to(
                    device=src.device, dtype=torch.bool
                ) | padding_mask
            )

        pose_inputs = self._known_camera_pose_inputs(
            src, camera_source_mask, caption_embedding, memory_teacher_forcing_ratio
        )
        encoder_src = src
        if self.keyframe_pose_conditioning and camera_source_mask is not None:
            encoder_src = src.masked_fill(
                camera_source_mask.to(device=src.device, dtype=torch.bool).unsqueeze(-1), 0
            )
        camera_embedding = self.encoder(
            encoder_src,
            subject_embedding,
            camera_source_mask,
            subject_key_padding_mask=tgt_key_padding_mask,
        )

        memory = self.prepare_embedding_memory_for_decoder(
            camera_embedding=camera_embedding.clone(),
            caption_embedding=caption_embedding,
            teacher_forcing_ratio=memory_teacher_forcing_ratio,
            mask_memory_prob=mask_memory_prob
        )

        reconstructed_raw = self.decoder(
            memory=memory,
            subject_embedding=subject_embedding,
            decode_mode=decode_mode,
            target=(None if self.keyframe_pose_conditioning else self._encode_target_for_decoder(target)),
            teacher_forcing_ratio=trajectory_teacher_forcing_ratio,
            tgt_key_padding_mask=tgt_key_padding_mask,
            feedback_transform=self._project_svd9d,
            **pose_inputs,
        )
        reconstructed, recon_matrix = self._finalize_trajectory(reconstructed_raw)

        output = {
            'subject_embedding': subject_embedding,
            'embeddings': camera_embedding,
            'reconstructed': reconstructed,                 # euler from valid SO(3): cycle / inference
            'reconstructed_raw_matrix': reconstructed_raw,  # raw matrices for the target auxiliary loss
            'reconstructed_rot_matrix': recon_matrix,       # projected SO(3) geometry for trajectory losses
        }

        if self.use_merged_memory:
            output['cls_embedding'] = memory[0]
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
        decode_mode: str = 'single_step'
    ) -> Dict[str, torch.Tensor]:
        self._validate_pose_decode_mode(decode_mode)
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
                    decode_mode=decode_mode
                )

            # If there is no camera trajectory, use only the decoder
            else:
                if self.keyframe_pose_conditioning and padding_mask is not None:
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
                    teacher_forcing_ratio=0.0
                )

                reconstructed_raw = self.decoder(
                    memory=memory,
                    subject_embedding=subject_embedding,
                    decode_mode=decode_mode,
                    tgt_key_padding_mask=padding_mask,
                    teacher_forcing_ratio=0.0,
                    feedback_transform=self._project_svd9d,
                )
                reconstructed, recon_matrix = self._finalize_trajectory(reconstructed_raw)

                output = {
                    'reconstructed': reconstructed,
                    'reconstructed_raw_matrix': reconstructed_raw,
                    'reconstructed_rot_matrix': recon_matrix,
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
            subject_embedding = torch.cat([
                self.subject_trajectory_projection(subject_trajectory.to(device)),
                self.subject_volume_projection(subject_volume.to(device)),
            ], 1)
            camera_embedding = self.encoder(
                camera_trajectory.to(device),
                subject_embedding,
                src_key_mask,
                subject_key_padding_mask=(
                    src_key_mask
                    if subject_key_padding_mask is None
                    else subject_key_padding_mask
                ),
            )
            return camera_embedding[:self.memory_tokens_count]

    def load_means_and_stds(self):
        with open("embedding_means.pkl", 'rb') as f:
            embedding_means_raw = pickle.load(f)
        with open("embedding_stds.pkl", "rb") as f:
            embedding_stds_raw = pickle.load(f)

        embedding_means = {}
        embedding_stds = {}
        for feature, value in embedding_means_raw.items():
            embedding_means[feature] = torch.tensor(value, device=self.device)

        for feature, value in embedding_stds_raw.items():
            embedding_stds[feature] = torch.tensor(value, device=self.device)

        return embedding_means, embedding_stds


    def get_mean_and_std(self, feature):
        return self.embedding_means[feature], self.embedding_stds[feature]
