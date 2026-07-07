from typing import Any, Dict, List, Optional

import torch

from data.convertor.convertor import convert_to_target
from data.sim_format import (
    SIM_SEQ_LENGTH,
    MEMORY_TEACHER_FORCING_BY_MODE,
    build_keyframing_mask,
    to_simulation_format,
)
from utils.device import move_batch_to_device


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
    pre_generated_trajectory: Optional[torch.Tensor] = None,
    clatr_extractor=None,
) -> Optional[torch.Tensor]:
    batch = move_batch_to_device(batch, device)
    batch_size = len(batch["text_prompts"])

    generated_trajectory_data: Optional[torch.Tensor] = None

    (
        sim_camera_trajectory,
        sim_subject_trajectory,
        sim_subject_volume,
        sim_padding_mask,
    ) = to_simulation_format(batch, dataset_type)

    (
        camera_trajectory,
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

    key_framing_padding_mask = build_keyframing_mask(batch_size, device)

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

        ref_output = ref_model.generate_camera_trajectory(
            subject_trajectory=sim_subject_trajectory,
            subject_volume=sim_subject_volume,
            camera_trajectory=sim_camera_trajectory,
            padding_mask=current_padding_mask,
            memory_teacher_forcing_ratio=memory_teacher_forcing_ratio,
            caption_embedding=caption_embedding,
        )

        if model_type in ("ccdm", "et", "gendop"):
            if pre_generated_trajectory is not None:
                generated_trajectory = pre_generated_trajectory
            else:
                generated_trajectory = model.generate_using_text(
                    batch["text_prompts"],
                    subject_trajectory,
                    camera_trajectory,
                    padding_mask,
                )
                generated_trajectory_data = generated_trajectory.detach().cpu()

            (
                sim_generated_trajectory,
                _,
                _,
                _,
            ) = convert_to_target(
                model_type,
                "simulation",
                generated_trajectory,
                subject_trajectory,
                batch["subject_volume"],
                padding_mask,
                SIM_SEQ_LENGTH,
                need_denormal=False,
            )
        elif model_type == "lens_craft":
            sim_generated_trajectory = ref_output["reconstructed"]
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

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

        if (
            model_type == "lens_craft"
            and metric_callback.clip_embeddings is not None
            and "cinematography_prompt_parameters" in batch
        ):
            encoder_features = ref_output["embeddings"][
                : ref_model.memory_tokens_count, ...
            ]
            encoder_features = encoder_features.permute(1, 0, 2).detach()
            metric_callback.update_caption_top1(
                metric_item,
                encoder_features,
                batch["cinematography_prompt_parameters"],
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

    return generated_trajectory_data
