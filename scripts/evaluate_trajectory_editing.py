#!/usr/bin/env python3
"""Evaluate literal structured-prompt edits on a held-out simulation cohort.

This is reference-assisted editing: the full source trajectory is an input.
It neither trains a model nor changes the existing generation evaluation suite.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("TEST_CHECKPOINT_PATH"))
    parser.add_argument("--train-config", type=Path, default=os.environ.get("TEST_CONFIG_PATH"))
    parser.add_argument("--data-config", type=Path, help="Full evaluation YAML; defaults to training config.")
    parser.add_argument("--semantic-evaluator-checkpoint", type=Path,
                        default=os.environ.get("SEMANTIC_EVALUATOR_CHECKPOINT_PATH"))
    parser.add_argument("--semantic-evaluator-config", type=Path,
                        default=os.environ.get("SEMANTIC_EVALUATOR_CONFIG_PATH"))
    parser.add_argument("--normalization-data-path", type=Path,
                        help="Simulation export used for the checkpoint's normalization.")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory.")
    parser.add_argument("--field", default="initial_shotSize")
    parser.add_argument("--value", default="closeUp")
    parser.add_argument("--limit", type=int, default=128, help="Holdout prefix before eligibility; 0 means all.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args(argv)
    for name in ("checkpoint", "train_config", "semantic_evaluator_checkpoint"):
        if getattr(args, name) in (None, "", "None"):
            parser.error("--" + name.replace("_", "-") + " is required (or set its environment variable)")
    if args.limit < 0 or args.batch_size < 1 or args.seed < 0:
        parser.error("limit/seed must be nonnegative and batch-size positive")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(totals):
    return {name: {"mean": total / count if count else None, "count": int(count)}
            for name, (total, count) in sorted(totals.items())}


def _absolute_paths(value):
    """Saved repository configurations conventionally resolve paths at repo root."""
    if isinstance(value, dict):
        return {key: (str((REPO_ROOT / Path(child).expanduser()).resolve())
                      if key in ("data_path", "cache_file") and isinstance(child, str)
                      else _absolute_paths(child)) for key, child in value.items()}
    if isinstance(value, list):
        return [_absolute_paths(child) for child in value]
    return value


@contextmanager
def artifact_workdir(path):
    """Embedding statistics written by existing loaders stay with this run."""
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def source_values(batch, field):
    return [next((entry[1] for entry in entries if entry[0] == field), None)
            for entries in batch["cinematography_prompt_parameters"]]


def attention_boxes(dataset, indices, volume_world):
    """Scoring-only physical metadata; never passed to the generator."""
    import torch
    from data.simulation.loader import parse_simulation_file_to_dict

    dimensions, positions = [], []
    for row, index in enumerate(indices):
        data = parse_simulation_file_to_dict(dataset.simulation_files[index],
                                             dataset.parameter_dictionary, dataset.fixed_point_scale)
        attention = data["subjectsInfo"][0]["subject"].get("attentionBox")
        if attention is None:
            body = volume_world[row].reshape(-1, 3)[0].detach().cpu()
            dimensions.append(body * .5)
            positions.append(torch.tensor([0., float(body[1]) * .25, 0.]))
        else:
            dimensions.append(torch.tensor([attention["dimensions"][key]
                                             for key in ("width", "height", "depth")]))
            positions.append(torch.tensor([attention["position"][key] for key in ("x", "y", "z")]))
    return torch.stack(dimensions), torch.stack(positions)


def world_geometry(batch, dataset):
    from data.simulation.dataset import SimulationDataset

    reference = batch.get("geometry_reference")
    if reference is not None:
        if reference.get("simulation_normalized", False):
            raise ValueError("geometry_reference must use world coordinates")
        return (reference["camera_trajectory"], reference["subject_trajectory"],
                reference["subject_volume"], reference.get("camera_intrinsics"),
                reference["padding_mask"])
    camera, subject, volume = (batch[key] for key in
                                ("camera_trajectory", "subject_trajectory", "subject_volume"))
    if dataset.normalize:
        camera, subject, volume = SimulationDataset._normalize_item_with_parameters(
            camera, subject, volume, dataset.normalization_parameters, normalize=False)
    return camera, subject, volume, batch.get("camera_intrinsics"), batch["padding_mask"]


def encode_valid(evaluator, camera, subject, volume, padding):
    """An invalid prediction is unscorable, rather than sanitized into success."""
    import torch

    finite = (torch.isfinite(camera).all(-1) | padding).all(-1) & (~padding).any(-1)
    width = evaluator.subject_trajectory_projection.out_features
    features = camera.new_full((evaluator.num_query_tokens, len(camera), width), float("nan"))
    if finite.any():
        selected = camera[finite].masked_fill(padding[finite, :, None], 0)
        features[:, finite] = evaluator.embed_trajectory(
            selected, subject[finite], volume[finite], src_key_mask=padding[finite],
            subject_key_padding_mask=padding[finite])
    return features


def evaluate_batches(model, evaluator, dataset, loader, indices, args, output, metric_vocabulary):
    """Run paired controls and edits; return sums/counts and cohort accounting."""
    import torch
    from data.simulation.dataset import SimulationDataset
    from data.simulation.utils import structured_conditioning_from_batch
    from testing.trajectory_editing import (
        build_edit_condition, edit_geometry_totals, semantic_edit_totals,
    )
    from testing.edit_shot_geometry import shot_size_geometry_totals
    from testing.metrics.semantic_evaluator import metric_conditioning_from_batch
    from utils.device import move_batch_to_device

    totals = defaultdict(lambda: [0., 0])
    counts = {"scanned": 0, "edited": 0, "skipped_missing": 0, "skipped_unchanged": 0}
    saved = []
    alias_fields = []
    with (output / "samples.jsonl").open("x") as records, torch.inference_mode():
        for raw in loader:
            batch = move_batch_to_device(raw, args.device)
            caption = structured_conditioning_from_batch(batch)
            presence = batch["prompt_none_mask"]
            values = source_values(raw, args.field)
            condition = build_edit_condition(caption, presence, args.field, args.value,
                                             source_values=values)
            noop = build_edit_condition(caption, presence, args.field, None,
                                       source_values=values)
            metric_caption, metric_presence = metric_conditioning_from_batch(
                batch, metric_vocabulary, device=batch["camera_trajectory"].device)
            alias_fields = condition["diagnostic_alias_fields"]
            selected = condition["eligible"].nonzero().flatten()
            offset = counts["scanned"]
            batch_indices = indices[offset:offset + len(values)]
            counts["scanned"] += len(values)
            counts["skipped_missing"] += int(condition["missing"].sum())
            counts["skipped_unchanged"] += int(condition["unchanged"].sum())
            for row, value in enumerate(values):
                if not condition["eligible"][row]:
                    records.write(json.dumps({
                        "sample_id": dataset.simulation_files[batch_indices[row]].stem,
                        "source_value": value, "target_value": args.value,
                        "status": "missing" if condition["missing"][row] else "unchanged",
                    }, allow_nan=False) + "\n")
            if not len(selected):
                continue
            cpu_rows = selected.cpu().tolist()
            camera, subject, volume, padding = [batch[key][selected] for key in
                ("camera_trajectory", "subject_trajectory", "subject_volume", "padding_mask")]
            inputs = dict(camera_trajectory=camera, subject_trajectory=subject,
                          subject_volume=volume, padding_mask=padding, src_key_mask=padding)
            # All three controls use the same encoder-only source path, even
            # for checkpoints with optional direct pose queries enabled.
            generated = {"reconstruction": model.generate_camera_trajectory(
                **inputs, caption_embedding=caption[:, selected],
                caption_presence_mask=presence[selected],
                caption_edit_mask=torch.zeros_like(presence[selected]),
            )["reconstructed"]}
            for name, prompt in (("noop", noop), ("edited", condition)):
                generated[name] = model.generate_camera_trajectory(
                    **inputs, caption_embedding=prompt["caption_embedding"][:, selected],
                    caption_presence_mask=prompt["caption_presence_mask"][selected],
                    caption_edit_mask=prompt["caption_edit_mask"][selected],
                )["reconstructed"]
            features = {"source": encode_valid(evaluator, camera, subject, volume, padding)}
            features.update({name: encode_valid(evaluator, trajectory, subject, volume, padding)
                             for name, trajectory in generated.items()})
            source_world, subject_world, volume_world, intrinsics, physical_padding = world_geometry(batch, dataset)
            source_world, subject_world, volume_world, physical_padding = (
                value[selected] for value in (source_world, subject_world, volume_world, physical_padding))
            intrinsics = None if intrinsics is None else intrinsics[selected]
            world = {name: (SimulationDataset._normalize_item_with_parameters(
                trajectory, None, None, dataset.normalization_parameters, normalize=False)[0]
                if dataset.normalize else trajectory) for name, trajectory in generated.items()}
            selected_indices = [batch_indices[row] for row in cpu_rows]
            attention_dims, attention_pos = (attention_boxes(dataset, selected_indices, volume_world)
                if args.field == "initial_shotSize" else (None, None))
            for local, row in enumerate(cpu_rows):
                sl = slice(local, local + 1)
                semantic = semantic_edit_totals(
                    features["source"][:, sl], features["reconstruction"][:, sl],
                    features["noop"][:, sl], features["edited"][:, sl],
                    metric_caption[:, selected[sl]], metric_presence[selected[sl]], args.field, args.value,
                    metric_vocabulary, torch.ones(1, dtype=torch.bool, device=camera.device))
                geometry_kwargs = dict(padding_mask=physical_padding[sl],
                    subject_trajectory=subject_world[sl], subject_dimensions=volume_world[sl],
                    intrinsics=None if intrinsics is None else intrinsics[sl])
                measured = dict(semantic)
                measured.update(edit_geometry_totals(world["edited"][sl], world["reconstruction"][sl],
                                                     source_world[sl], **geometry_kwargs))
                measured.update({"noop_" + name: metric for name, metric in edit_geometry_totals(
                    world["noop"][sl], world["reconstruction"][sl], source_world[sl],
                    **geometry_kwargs).items()})
                if args.field == "initial_shotSize":
                    measured.update(shot_size_geometry_totals(
                        world["edited"][sl], world["reconstruction"][sl],
                        subject_trajectory=subject_world[sl], subject_dimensions=volume_world[sl],
                        intrinsics=None if intrinsics is None else intrinsics[sl],
                        valid_mask=~physical_padding[sl], shot_size=args.value,
                        attention_dimensions=attention_dims[sl], attention_position=attention_pos[sl]))
                for name, (total, count) in measured.items():
                    totals[name][0] += float(total)
                    totals[name][1] += int(count)
                sample_id = dataset.simulation_files[batch_indices[row]].stem
                records.write(json.dumps({"sample_id": sample_id, "source_value": values[row],
                    "target_value": args.value, "status": "edited", "metrics": summarize(measured)},
                    allow_nan=False) + "\n")
                counts["edited"] += 1
                if args.save_trajectories:
                    saved.append({"sample_id": sample_id, "source": source_world[sl].cpu(),
                        "subject": subject_world[sl].cpu(), "volume": volume_world[sl].cpu(),
                        "padding_mask": physical_padding[sl].cpu(),
                        **{key: value[sl].cpu() for key, value in world.items()}})
            records.flush()
            print(f"Scanned {counts['scanned']}/{len(indices)}; edited {counts['edited']}", flush=True)
    if args.save_trajectories:
        torch.save({"coordinate_space": "simulation_world", "samples": saved}, output / "trajectories.pt")
    return summarize(totals), counts, alias_fields


def run(args):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset
    from data.simulation.dataset import collate_fn
    from data.simulation.utils import cinematography_struct_parameters
    from testing.edit_checkpoint import load_edit_generator, read_edit_checkpoint
    from testing.evaluator import checkpoint_identity, load_semantic_evaluator
    from testing.trajectory_editing import FIELD_NAMES, build_edit_condition

    train = OmegaConf.load(args.train_config)
    evaluation = OmegaConf.load(args.data_config) if args.data_config else train
    module = OmegaConf.create(OmegaConf.to_container(train.training.model.module, resolve=True))
    # CLIP belongs to the frozen metric judge, independently of generator config.
    ref = OmegaConf.load(REPO_ROOT / "config/ref_model/default.yaml")
    metric_clip = evaluation.get("clip", train.get("clip", ref.clip))
    clip = OmegaConf.create(OmegaConf.to_container(metric_clip, resolve=True))
    data = _absolute_paths(OmegaConf.to_container(evaluation.data, resolve=True))
    # Evaluation resolution must not change how training statistics are sampled.
    normalization_frames = OmegaConf.select(train, "data.dataset.config.normalization_frame_count")
    if normalization_frames is None:
        normalization_frames = (OmegaConf.select(train, "data.dataset.config.target_frame_count")
                                or module.get("seq_length", 30))
    data["dataset"]["config"]["normalization_frame_count"] = int(normalization_frames)
    if data["dataset"]["config"]["_target_"] != "data.simulation.dataset.SimulationDataset":
        raise ValueError("Editing evaluation currently requires SimulationDataset with structured source labels")
    if data["dataset"]["config"].get("allowed_movement_types"):
        raise ValueError("Use the full simulation inventory before selecting the held-out split")
    if not data["dataset"]["config"].get("normalize", True):
        raise ValueError("Editing evaluation requires the checkpoint's normalized simulation inputs")
    training_data_path = OmegaConf.select(train, "data.dataset.config.data_path")
    normalization = (args.normalization_data_path or os.environ.get("SIMULATION_NORMALIZATION_DATA_PATH")
                     or training_data_path)
    payload = read_edit_checkpoint(args.checkpoint)
    saved_normalization = payload.get("normalization")
    use_saved_normalization = saved_normalization is not None and args.normalization_data_path is None
    if not normalization and not use_saved_normalization:
        raise ValueError("Supply --normalization-data-path with the checkpoint's simulation normalization source")
    if normalization:
        os.environ["SIMULATION_NORMALIZATION_DATA_PATH"] = str((REPO_ROOT / Path(normalization).expanduser()).resolve())
    data["dataset"]["config"]["target_frame_count"] = int(module.get("seq_length", 30))
    # Migrate data-only keys from the saved categorical experiment; model
    # validation below rejects old generator checkpoints.
    for removed in ("conditioning_representation", "clip_embeddings", "embedding_dim", "fill_none_with_mean"):
        data["dataset"]["config"].pop(removed, None)
    args.output.mkdir(parents=True, exist_ok=False)
    with artifact_workdir(args.output):
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        dataset_definition = dict(data["dataset"]["config"])
        if use_saved_normalization:
            # Robust scratch checkpoints carry authoritative moments. Do not
            # recompute them from a new evaluation cohort or a JSON-only source.
            dataset_definition["normalize"] = False
        dataset = instantiate(OmegaConf.create(dataset_definition))
        if use_saved_normalization:
            dataset.normalization_parameters = {
                group: {name: torch.as_tensor(value, dtype=torch.float32)
                        for name, value in moments.items()}
                for group, moments in saved_normalization.items()}
            dataset.normalize = True
        total = len(dataset)
        train_count = int((1 - float(data["val_size"]) - float(data["test_size"])) * total)
        val_count = int(float(data["val_size"]) * total)
        if train_count < 0 or val_count < 0 or train_count + val_count >= total:
            raise ValueError("The configured split must contain at least one held-out sample")
        split_seed = int(data.get("split_seed", 42))
        order = torch.randperm(total, generator=torch.Generator().manual_seed(split_seed)).tolist()
        indices = order[train_count + val_count:]
        full_holdout_count = len(indices)
        if args.limit:
            indices = indices[:args.limit]
        # Validate the requested field/value before model construction or generation.
        from models.categorical_latent import TOKEN_WIDTH
        build_edit_condition(torch.zeros(len(FIELD_NAMES), 1, TOKEN_WIDTH),
                             torch.zeros(1, len(FIELD_NAMES), dtype=torch.bool),
                             args.field, args.value, source_values=[None])
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
        inference = OmegaConf.create({"checkpoint_path": str(args.checkpoint), "config": None})
        model, checkpoint_adapter = load_edit_generator(
            module, inference, device, normalization_parameters=dataset.normalization_parameters,
            checkpoint_payload=payload)
        del payload
        ref.inference.checkpoint_path = str(args.semantic_evaluator_checkpoint)
        ref.inference.config = str(args.semantic_evaluator_config) if args.semantic_evaluator_config else None
        evaluator = load_semantic_evaluator(ref, device, input_clip=clip)
        from testing.metrics.semantic_vocabulary import load_metric_clip_embeddings
        metric_vocabulary = load_metric_clip_embeddings(
            str(clip.model_name),
            str(Path(os.environ.get("CLIP_EMBEDDINGS_CACHE_DIR", ".")) /
                f"clip_embeddings_{clip.model_name}_cache.pkl"),
            embedding_dimension=int(clip.latent_dim))
        if evaluator is model:
            raise ValueError("The semantic evaluator must be a separate frozen model")
        generator_identity = checkpoint_identity(args.checkpoint)
        provenance = {
            "protocol": "literal-structured-trajectory-edit-v1", "status": "running",
            "field": args.field, "value": args.value, "seed": args.seed, "split_seed": split_seed,
            "full_holdout_count": full_holdout_count, "selected_count": len(indices), "indices": indices,
            "sample_ids_sha256": hashlib.sha256(json.dumps([
                dataset.simulation_files[index].stem for index in indices]).encode()).hexdigest(),
            "dataset_fingerprint": dataset.dataset_fingerprint,
            "generator": generator_identity,
            "generator_loading": checkpoint_adapter,
            "train_config": checkpoint_identity(args.train_config),
            "data_config": checkpoint_identity(args.data_config or args.train_config),
            "semantic_evaluator": evaluator.evaluation_provenance,
            "target_semantic_metric_available": args.value in metric_vocabulary.get(
                dict(cinematography_struct_parameters)[args.field].__name__, {}),
            "generator_matches_evaluator_checkpoint": (
                generator_identity["sha256"] == evaluator.evaluation_provenance["checkpoint"]["sha256"]),
            "normalization_source": "checkpoint" if use_saved_normalization else "training_export",
            "normalization_data_path": (None if use_saved_normalization else
                                         os.environ["SIMULATION_NORMALIZATION_DATA_PATH"]),
            "normalization_parameters": {group: {name: value.tolist() for name, value in fields.items()}
                                         for group, fields in dataset.normalization_parameters.items()},
            "batch_size": args.batch_size, "resolved_data_config": data,
            "conditioning": "Replace one categorical field with its one-hot target; retain all other encoder probability tokens.",
            "controls": ["source", "reconstruction", "noop_original_field", "changed_field"],
            "source_metadata_policy": "Source labels select and score the cohort; no source labels/attention boxes enter edited decoder memory.",
            "metric_policy": "Learned semantics use the fixed evaluator; source-coordinate errors are drift diagnostics, not edit correctness. No universal pass threshold.",
        }
        write_json(args.output / "manifest.json", provenance)
        metrics, counts, aliases = evaluate_batches(
            model, evaluator, dataset, loader, indices, args, args.output, metric_vocabulary)
        provenance.update(status="complete", counts=counts, diagnostic_alias_fields=aliases,
            alias_policy="Literal substitution: low-level aliases are deliberately retained and may conflict with the edited field.")
        write_json(args.output / "manifest.json", provenance)
        write_json(args.output / "metrics.json", {"protocol": provenance, "metrics": metrics,
            "quality_status": "measured" if counts["edited"] else "no_eligible_edits"})
        print(json.dumps({"output": str(args.output), **counts,
                          "quality_status": "measured" if counts["edited"] else "no_eligible_edits"}))


def main(argv=None):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
