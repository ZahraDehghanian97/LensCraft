"""Generation operations shared by inference and evaluation."""

from data.convertor.alignment import recenter_rescale_sim, undo_recenter_rescale
from data.convertor.convertor import convert_to_target


def generate_baseline_trajectory(
    model, batch, dataset_type, model_type, seq_length,
    *, align_et_scene=False, **generation_kwargs,
):
    """Generate in the baseline's native format, then return simulation poses.

    The returned poses are not normalized or translated to the subject. Those
    output conventions belong to the caller's inference or evaluation mode.
    """
    camera = batch["camera_trajectory"]
    subject = batch["subject_trajectory"]
    volume = batch["subject_volume"]
    conversion_kwargs = {}
    if align_et_scene:
        camera, subject, volume, origin, scale = recenter_rescale_sim(
            camera, subject, volume,
        )
        dataset_type = "simulation"
        conversion_kwargs["need_denormal"] = False

    trajectory, subject, _, padding = convert_to_target(
        dataset_type, model_type, camera, subject, volume,
        batch["padding_mask"], seq_length, **conversion_kwargs,
    )
    generated = model.generate_using_text(
        batch["text_prompts"], subject, trajectory, padding, **generation_kwargs,
    )
    generated_padding = None if model_type in ("ccdm", "gendop") else padding
    sim_generated, *_ = convert_to_target(
        model_type, "simulation", generated, None, None,
        generated_padding, generated.shape[1],
        valid_target_len=(
            (~generated_padding).sum(dim=1)
            if generated_padding is not None else None
        ),
        need_denormal=False, need_normal=False,
    )
    if align_et_scene:
        sim_generated = undo_recenter_rescale(sim_generated, origin, scale)
    return sim_generated, generated_padding


def generate_lenscraft_trajectory(model, view, source_mask, caption_embedding, memory_ratio):
    """Apply a conditioning mode while keeping source visibility and padding separate."""
    camera, subject, volume, padding = view
    return model.generate_camera_trajectory(
        subject_trajectory=subject,
        subject_volume=volume,
        camera_trajectory=camera,
        src_key_mask=source_mask,
        padding_mask=padding,
        memory_teacher_forcing_ratio=memory_ratio,
        caption_embedding=caption_embedding,
    )["reconstructed"]
