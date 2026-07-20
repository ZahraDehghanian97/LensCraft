import torch

from data.convertor.alignment import recenter_rescale_sim, undo_recenter_rescale
from data.convertor.convertor import convert_to_target
from data.simulation.dataset import SimulationDataset
from data.sim_format import (
    SIM_SEQ_LENGTH,
    MEMORY_TEACHER_FORCING_BY_MODE,
    build_keyframing_mask,
    to_simulation_format,
)
from utils.device import move_batch_to_device

_KEYFRAMING_MODES = {"key_framing", "key_framing+prompt"}


def inference_batch(model, batch, device, dataset_type="simulation",
                    model_type="lens_craft", seq_length=SIM_SEQ_LENGTH):
    batch = move_batch_to_device(batch, device)
    batch_size = len(batch["text_prompts"])

    caption_embedding = (
        batch.get("cinematography_prompt")
        if dataset_type in ("simulation", "et") else None
    )

    sim_camera, sim_subject, sim_volume, sim_padding = to_simulation_format(
        batch, dataset_type
    )

    et_scene_origin = et_scene_scale = None
    if model_type == "et" and dataset_type in ("simulation", "lens_craft"):
        aligned_camera, aligned_subject, aligned_volume, et_scene_origin, et_scene_scale = (
            recenter_rescale_sim(
                batch["camera_trajectory"], batch["subject_trajectory"],
                batch["subject_volume"],
            )
        )
        trajectory, subject_trajectory, subject_volume, padding_mask = convert_to_target(
            "simulation", "et",
            aligned_camera, aligned_subject, aligned_volume,
            batch["padding_mask"], seq_length,
            need_denormal=False,
        )
    else:
        trajectory, subject_trajectory, subject_volume, padding_mask = convert_to_target(
            dataset_type, model_type,
            batch["camera_trajectory"], batch["subject_trajectory"],
            batch["subject_volume"], batch["padding_mask"], seq_length,
            torch.full((batch_size,), SIM_SEQ_LENGTH, device=device),  # TODO: other datasets
        )

    if model_type in ("ccdm", "et", "gendop"):
        generated = model.generate_using_text(
            batch["text_prompts"], subject_trajectory, trajectory, padding_mask,
        )
        gen_padding_mask = None if model_type == "ccdm" else padding_mask
        sim_generated, *_ = convert_to_target(
            model_type, "simulation", generated, None, None,
            gen_padding_mask, SIM_SEQ_LENGTH,
            need_denormal=False, need_normal=False,
        )
        if model_type == "ccdm" and sim_subject is not None:
            _, subject_denorm, _ = SimulationDataset.normalize_item(
                sim_camera, sim_subject, None, False
            )
            sim_generated[..., :3] += subject_denorm[..., :3]
        if et_scene_origin is not None:
            sim_generated = undo_recenter_rescale(
                sim_generated, et_scene_origin, et_scene_scale
            )
        return ({"prompt_generation": sim_generated},
                sim_camera, sim_subject, sim_volume, sim_padding, None)

    keyframing_mask = build_keyframing_mask(batch_size, device)

    results = {}
    for mode in ("prompt_generation", "reconstruction", "key_framing+prompt",
                 "key_framing", "source_trajectory"):
        mask = keyframing_mask if mode in _KEYFRAMING_MODES else sim_padding
        mode_caption = caption_embedding

        if mode == "source_trajectory":
            # Pair each trajectory with a *different* sample's prompt.
            if "cinematography_prompt" not in batch:
                continue
            shuffled = [batch["cinematography_prompt"][:, idx, :]
                        for idx in batch["random_prompt_index"]]
            mode_caption = torch.stack(shuffled, dim=1).to(device)

        results[mode] = model.generate_camera_trajectory(
            subject_trajectory=sim_subject,
            subject_volume=sim_volume,
            camera_trajectory=sim_camera,
            padding_mask=mask,
            memory_teacher_forcing_ratio=MEMORY_TEACHER_FORCING_BY_MODE[mode],
            caption_embedding=mode_caption,
        )["reconstructed"]

    return results, sim_camera, sim_subject, sim_volume, sim_padding, keyframing_mask
