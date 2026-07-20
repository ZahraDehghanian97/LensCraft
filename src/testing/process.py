from typing import Any, Dict, List, Optional

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
from testing.baseline_modes import (
    NO_NORM_ITEM,
    NORM_ITEM,
    NORM_LENSCRAFT_INIT_ITEM,
    place_trajectory_at_first_position,
)
from utils.device import move_batch_to_device

BASELINE_MODELS = ("ccdm", "et", "gendop")


def _to_sim_space(model_type: str, generated: torch.Tensor,
                  gen_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    sim_generated, *_ = convert_to_target(
        model_type,
        "simulation",
        generated,
        None,
        None,
        gen_padding_mask,
        SIM_SEQ_LENGTH,
        need_denormal=False,
        need_normal=False,
    )
    return sim_generated


def _lenscraft_first_position(
    ref_model,
    batch: Dict[str, Any],
    sim_subject_trajectory: torch.Tensor,
    sim_subject_volume: torch.Tensor,
    sim_padding_mask: torch.Tensor,
) -> torch.Tensor:
    caption_embedding = batch.get("cinematography_prompt")
    if caption_embedding is None:
        raise ValueError(
            "The normalized + LensCraft-init baseline mode requires a "
            "cinematography prompt."
        )

    lenscraft_generation = ref_model.generate_camera_trajectory(
        caption_embedding=caption_embedding,
        camera_trajectory=None,
        subject_trajectory=sim_subject_trajectory,
        subject_volume=sim_subject_volume,
        padding_mask=sim_padding_mask,
    )["reconstructed"]
    return lenscraft_generation[..., :1, :3]


def _generate_baseline_variants(
    ref_model,
    model,
    batch: Dict[str, Any],
    metric_items: List[str],
    dataset_type: str,
    model_type: str,
    seq_length: int,
    sim_camera_trajectory: torch.Tensor,
    sim_subject_trajectory: Optional[torch.Tensor],
    sim_subject_volume: Optional[torch.Tensor],
    sim_padding_mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
):
    want_norm = NORM_ITEM in metric_items
    want_no_norm = NO_NORM_ITEM in metric_items
    want_lenscraft_init = NORM_LENSCRAFT_INIT_ITEM in metric_items
    need_norm = want_norm or want_lenscraft_init

    variant_trajectories: Dict[str, torch.Tensor] = {}
    normalized_trajectory: Optional[torch.Tensor] = None

    et_on_sim = model_type == "et" and dataset_type in ("simulation", "lens_craft")

    if et_on_sim:
        if need_norm:
            (
                aligned_camera,
                aligned_subject,
                aligned_volume,
                et_scene_origin,
                et_scene_scale,
            ) = recenter_rescale_sim(
                batch["camera_trajectory"],
                batch["subject_trajectory"],
                batch["subject_volume"],
            )
            traj_n, subj_n, _, pad_n = convert_to_target(
                "simulation",
                "et",
                aligned_camera,
                aligned_subject,
                aligned_volume,
                batch["padding_mask"],
                seq_length,
                need_denormal=False,
            )
            gen_norm = model.generate_using_text(
                batch["text_prompts"], subj_n, traj_n, pad_n
            )
            sim_gen = _to_sim_space("et", gen_norm, pad_n)
            sim_gen = undo_recenter_rescale(sim_gen, et_scene_origin, et_scene_scale)
            sim_gen, _, _ = SimulationDataset.normalize_item(sim_gen, None, None, True)
            normalized_trajectory = sim_gen
            if want_norm:
                variant_trajectories[NORM_ITEM] = normalized_trajectory

        if want_no_norm:
            traj_r, subj_r, _, pad_r = convert_to_target(
                "simulation",
                "et",
                batch["camera_trajectory"],
                batch["subject_trajectory"],
                batch["subject_volume"],
                batch["padding_mask"],
                seq_length,
            )
            gen_raw = model.generate_using_text(
                batch["text_prompts"], subj_r, traj_r, pad_r
            )
            sim_gen = _to_sim_space("et", gen_raw, pad_r)
            variant_trajectories[NO_NORM_ITEM] = sim_gen

        if want_lenscraft_init:
            first_position = _lenscraft_first_position(
                ref_model,
                batch,
                sim_subject_trajectory,
                sim_subject_volume,
                sim_padding_mask,
            )
            variant_trajectories[NORM_LENSCRAFT_INIT_ITEM] = (
                place_trajectory_at_first_position(
                    normalized_trajectory, first_position
                )
            )

        return variant_trajectories

    (
        trajectory,
        subject_trajectory,
        subject_volume,
        padding_mask,
    ) = convert_to_target(
        dataset_type,
        model_type,
        batch["camera_trajectory"],
        batch["subject_trajectory"],
        batch["subject_volume"],
        batch["padding_mask"],
        seq_length,
        torch.full((batch_size,), SIM_SEQ_LENGTH, device=device), # fix me for other datasets
    )
    generated = model.generate_using_text(
        batch["text_prompts"], subject_trajectory, trajectory, padding_mask
    )

    gen_padding_mask = None if model_type == "ccdm" else padding_mask
    sim_generated = _to_sim_space(model_type, generated, gen_padding_mask)

    if need_norm:
        aligned = sim_generated.clone()
        if model_type == "ccdm" and sim_subject_trajectory is not None:
            _, subject_denorm, _ = SimulationDataset.normalize_item(
                sim_camera_trajectory, sim_subject_trajectory, None, False
            )
            aligned[..., :3] += subject_denorm[..., :3]
        aligned, _, _ = SimulationDataset.normalize_item(aligned, None, None, True)
        normalized_trajectory = aligned
        if want_norm:
            variant_trajectories[NORM_ITEM] = normalized_trajectory

    if want_no_norm:
        variant_trajectories[NO_NORM_ITEM] = sim_generated.clone()

    if want_lenscraft_init:
        first_position = _lenscraft_first_position(
            ref_model,
            batch,
            sim_subject_trajectory,
            sim_subject_volume,
            sim_padding_mask,
        )
        variant_trajectories[NORM_LENSCRAFT_INIT_ITEM] = (
            place_trajectory_at_first_position(
                normalized_trajectory, first_position
            )
        )

    return variant_trajectories


def _update_generation_metrics(
    metric_callback,
    metric_item: str,
    sim_generated_trajectory: torch.Tensor,
    sim_subject_trajectory: Optional[torch.Tensor],
    sim_subject_volume: Optional[torch.Tensor],
    sim_padding_mask: torch.Tensor,
    ref_clatr: Optional[torch.Tensor],
    text_clatr: Optional[torch.Tensor],
    clatr_extractor,
    ref_model,
    batch: Dict[str, Any],
) -> None:
    if clatr_extractor is not None and ref_clatr is not None:
        gen_clatr = clatr_extractor.encode_trajectory(
            sim_generated_trajectory,
            sim_subject_trajectory,
            sim_subject_volume,
            sim_padding_mask,
        )
        metric_callback.update_clatr_metrics(
            metric_item,
            gen_features=gen_clatr,
            ref_features=ref_clatr,
            text_features=text_clatr,
        )

    if batch.get("cinematography_prompt") is not None:
        n_high = ref_model.memory_tokens_count
        gen_embedding = ref_model.embed_trajectory(
            sim_generated_trajectory,
            sim_subject_trajectory,
            sim_subject_volume,
        )
        prompt_embedding = batch["cinematography_prompt"][:n_high]
        prompt_none_mask = batch.get("prompt_none_mask")
        if prompt_none_mask is not None:
            prompt_none_mask = prompt_none_mask[:, :n_high]
        metric_callback.update_clip_score(
            metric_item,
            gen_embedding,
            prompt_embedding,
            prompt_none_mask,
        )


def test_batch(
    ref_model,
    model,
    batch: Dict[str, Any],
    metric_callback,
    device: torch.device,
    metric_items: List[str],
    dataset_type: str = "simulation",
    model_type: str = "lens_craft",
    seq_length: int = 30,
    clatr_extractor=None,
    cached_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    batch = move_batch_to_device(batch, device)
    batch_size = len(batch["text_prompts"])

    (
        sim_camera_trajectory,
        sim_subject_trajectory,
        sim_subject_volume,
        sim_padding_mask,
    ) = to_simulation_format(batch, dataset_type)

    ref_clatr: Optional[torch.Tensor] = None
    text_clatr: Optional[torch.Tensor] = None
    if clatr_extractor is not None:
        ref_clatr = clatr_extractor.encode_trajectory(
            sim_camera_trajectory,
            sim_subject_trajectory,
            sim_subject_volume,
            sim_padding_mask,
        )
        if batch.get("text_prompts") is not None:
            text_clatr = clatr_extractor.encode_text(batch["text_prompts"])

    if model_type in BASELINE_MODELS:
        if cached_outputs is None:
            variant_trajectories = _generate_baseline_variants(
                ref_model,
                model,
                batch,
                metric_items,
                dataset_type,
                model_type,
                seq_length,
                sim_camera_trajectory,
                sim_subject_trajectory,
                sim_subject_volume,
                sim_padding_mask,
                device,
                batch_size,
            )
        else:
            variant_trajectories = {
                item: trajectory.to(device)
                for item, trajectory in cached_outputs["trajectories"].items()
            }
        for metric_item, sim_generated_trajectory in variant_trajectories.items():
            _update_generation_metrics(
                metric_callback,
                metric_item,
                sim_generated_trajectory,
                sim_subject_trajectory,
                sim_subject_volume,
                sim_padding_mask,
                ref_clatr,
                text_clatr,
                clatr_extractor,
                ref_model,
                batch,
            )
        return {
            "trajectories": {
                item: trajectory.detach().cpu()
                for item, trajectory in variant_trajectories.items()
            }
        }

    if model_type != "lens_craft":
        raise ValueError(f"Unsupported model_type: {model_type}")

    key_framing_padding_mask = build_keyframing_mask(batch_size, device)
    generated_outputs: Dict[str, Any] = {"items": {}}

    for metric_item in metric_items:
        caption_embedding = (
            batch.get("cinematography_prompt", None)
            if dataset_type in ("simulation", "et")
            else None
        )
        memory_teacher_forcing_ratio = MEMORY_TEACHER_FORCING_BY_MODE[metric_item]

        current_padding_mask = sim_padding_mask
        if metric_item in ("key_framing", "key_framing+prompt"):
            current_padding_mask = key_framing_padding_mask

        if cached_outputs is None:
            ref_output = ref_model.generate_camera_trajectory(
                subject_trajectory=sim_subject_trajectory,
                subject_volume=sim_subject_volume,
                camera_trajectory=sim_camera_trajectory,
                padding_mask=current_padding_mask,
                memory_teacher_forcing_ratio=memory_teacher_forcing_ratio,
                caption_embedding=caption_embedding,
            )
            sim_generated_trajectory = ref_output["reconstructed"]
        else:
            item_output = cached_outputs["items"][metric_item]
            sim_generated_trajectory = item_output["trajectory"].to(device)
            ref_output = None

        if cached_outputs is None:
            generated_outputs["items"][metric_item] = {
                "trajectory": sim_generated_trajectory.detach().cpu(),
                "encoder_features": None,
            }

        _update_generation_metrics(
            metric_callback,
            metric_item,
            sim_generated_trajectory,
            sim_subject_trajectory,
            sim_subject_volume,
            sim_padding_mask,
            ref_clatr,
            text_clatr,
            clatr_extractor,
            ref_model,
            batch,
        )

        if (
            metric_callback.clip_embeddings is not None
            and "cinematography_prompt_parameters" in batch
        ):
            if ref_output is not None:
                encoder_features = ref_output["embeddings"][
                    : ref_model.memory_tokens_count, ...
                ]
                encoder_features = encoder_features.permute(1, 0, 2).detach()
                generated_outputs["items"][metric_item]["encoder_features"] = (
                    encoder_features.cpu()
                )
            else:
                encoder_features = item_output["encoder_features"].to(device)
            metric_callback.update_caption_top1(
                metric_item,
                encoder_features,
                batch["cinematography_prompt_parameters"],
            )

    if cached_outputs is not None:
        return cached_outputs
    return generated_outputs
