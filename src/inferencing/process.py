import torch
from data.ccdm.alignment import align_ccdm_to_subject

from data.simulation.dataset import SimulationDataset
from data.categorical_conditioning import conditioning_from_batch, conditioning_presence_from_batch
from data.sim_format import (
    SIM_SEQ_LENGTH,
    NUM_VISIBLE_KEYFRAMES,
    MEMORY_TEACHER_FORCING_BY_MODE,
    build_keyframing_mask,
    to_simulation_format,
    geometry_context_from_batch,
)
from generation import generate_baseline_trajectory, generate_lenscraft_trajectory
from utils.device import move_batch_to_device

_KEYFRAMING_MODES = {"key_framing", "key_framing+prompt"}


def inference_batch(model, batch, device, dataset_type="simulation",
                    model_type="lens_craft", seq_length=SIM_SEQ_LENGTH,
                    num_keyframes=NUM_VISIBLE_KEYFRAMES,
                    keyframe_sample_seeds=None):
    batch = move_batch_to_device(batch, device)
    batch_size = len(batch["text_prompts"])

    view = to_simulation_format(
        batch, dataset_type, target_len=seq_length
    )
    sim_camera, sim_subject, sim_volume, sim_padding = view

    if model_type in ("ccdm", "et", "gendop"):
        sim_generated, _ = generate_baseline_trajectory(
            model, batch, dataset_type, model_type, seq_length,
            align_et_scene=(model_type == "et" and dataset_type in ("simulation", "lens_craft")),
        )
        if sim_camera.shape[1] != sim_generated.shape[1]:
            sim_camera, sim_subject, sim_volume, sim_padding = to_simulation_format(
                batch, dataset_type, target_len=sim_generated.shape[1]
            )
        if model_type == "ccdm" and sim_subject is not None:
            _, subject_denorm, _ = SimulationDataset.normalize_item(
                sim_camera, sim_subject, None, False
            )
            sim_generated = align_ccdm_to_subject(sim_generated, subject_denorm)
        return ({"prompt_generation": sim_generated},
                sim_camera, sim_subject, sim_volume, sim_padding, None)

    caption_embedding = conditioning_from_batch(batch, model)

    keyframing_mask = build_keyframing_mask(
        batch_size, device, sim_camera.shape[1],
        num_keyframes=num_keyframes,
        padding_mask=sim_padding,
        sample_seeds=keyframe_sample_seeds,
    )

    geometry_context = geometry_context_from_batch(
        batch, dataset_type, target_len=sim_camera.shape[1],
    )
    results = {}
    for mode in ("prompt_generation", "reconstruction", "key_framing+prompt",
                 "key_framing", "source_trajectory"):
        # Keyframing hides source-camera frames from the encoder.  It is not a
        # decoder padding operation: sending this mask only as ``padding_mask``
        # leaked every supposedly hidden frame into the encoder and instead
        # suppressed output timesteps. Real temporal padding must still be
        # hidden from both paths, including in keyframing modes.
        source_mask = (
            keyframing_mask
            if mode in _KEYFRAMING_MODES
            else sim_padding
        )
        mode_caption = caption_embedding
        mode_presence = conditioning_presence_from_batch(batch, model)

        if mode == "source_trajectory":
            # Pair each trajectory with a *different* sample's prompt.
            if caption_embedding is None:
                continue
            shuffled = [caption_embedding[:, idx, :]
                        for idx in batch["random_prompt_index"]]
            mode_caption = torch.stack(shuffled, dim=1).to(device)
            if mode_presence is not None:
                indices = torch.as_tensor(batch['random_prompt_index'], device=device, dtype=torch.long)
                mode_presence = mode_presence.index_select(0, indices)

        results[mode] = generate_lenscraft_trajectory(
            model, view, source_mask, mode_caption, MEMORY_TEACHER_FORCING_BY_MODE[mode],
            caption_presence_mask=mode_presence,
            known_camera_mask=(~source_mask if mode in _KEYFRAMING_MODES else None),
            geometry_context=geometry_context,
        )

    return results, sim_camera, sim_subject, sim_volume, sim_padding, keyframing_mask
