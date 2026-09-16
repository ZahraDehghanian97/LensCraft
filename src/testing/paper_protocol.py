"""Additional paper protocols, usable with the frozen 96a9cd7 evaluator.

No generation, trained weights, or existing metric implementations are changed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch


RANDOM_KEYFRAME_MODES = ("key_framing_random_1_10", "key_framing+prompt_random_1_10")


class UnsupportedGenerationLength(ValueError):
    """A native generator stopped before the requested length; do not pad it."""


def random_keyframe_mask(padding_mask, sample_seeds):
    """Uniform K in [1,10], then a uniform subset of min(K, valid frames).

    Count and frame RNG streams are independent, deterministic on CPU, and
    invariant to batch partitioning and the global RNG. Both conditioning
    modes must reuse this returned mask.
    """
    if padding_mask.dtype != torch.bool or padding_mask.ndim != 2:
        raise ValueError("padding_mask must be a [batch, frames] boolean tensor")
    if len(sample_seeds) != len(padding_mask):
        raise ValueError("One seed is required for every sample")
    mask = torch.ones_like(padding_mask, device="cpu")
    requested = []
    for row, seed in enumerate(sample_seeds):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Sample seeds must be nonnegative integers")
        streams = [int.from_bytes(hashlib.sha256(f"paper-random-k-v1:{seed}:{key}".encode()).digest()[:8], "little")
                   for key in ("count", "frames")]
        count = int(torch.randint(1, 11, (1,), generator=torch.Generator().manual_seed(streams[0])))
        valid = (~padding_mask[row].cpu()).nonzero(as_tuple=True)[0]
        ordering = torch.randperm(len(valid), generator=torch.Generator().manual_seed(streams[1]))
        mask[row, valid[ordering[:count]]] = False
        requested.append(count)
    return mask.to(padding_mask.device), requested


def install_random_keyframe_protocol(evaluator, cache_dir=None):
    """Install a process-local test.py extension; source files remain untouched."""
    from data.sim_format import to_simulation_format
    from data.simulation.utils import structured_conditioning_from_batch
    from testing.process import _update_generation_metrics, _update_geometry_metrics
    from utils.device import move_batch_to_device

    requested_histogram, actual_histogram = Counter(), Counter()
    recorded_masks = []
    original_writer = evaluator._write_metrics_json
    batch_index = 0

    def evaluate_batch(ref_model, model, batch, metric_callback, device, metric_items,
                       dataset_type="simulation", model_type="lens_craft", seq_length=30,
                       clatr_extractor=None, cached_outputs=None, generation_seeds=None,
                       num_keyframes=4, keyframe_sample_seeds=None, geometry_settings=None):
        nonlocal batch_index
        if model_type != "lens_craft" or cached_outputs is not None:
            raise ValueError("Random-K protocol requires uncached LensCraft evaluation")
        if keyframe_sample_seeds is None:
            raise ValueError("Explicit per-sample keyframe seeds are required")
        batch = move_batch_to_device(batch, device)
        view = to_simulation_format(batch, dataset_type, target_len=model.decoder.seq_length)
        camera, subject, volume, padding = view
        source_mask, requested = random_keyframe_mask(padding, keyframe_sample_seeds)
        requested_histogram.update(requested)
        actual_histogram.update((~source_mask).sum(1).cpu().tolist())
        recorded_masks.extend(zip(keyframe_sample_seeds, requested, source_mask.cpu().tolist()))
        reference = to_simulation_format(batch, dataset_type, target_len=ref_model.decoder.seq_length)
        ref_clatr = clatr_extractor.encode_trajectory(*reference) if clatr_extractor else None
        text_clatr = clatr_extractor.encode_text(batch["text_prompts"]) if clatr_extractor else None
        caption = structured_conditioning_from_batch(batch)
        generated = {"items": {}, "sample_seeds": list(keyframe_sample_seeds), "requested_counts": requested}
        for item in metric_items:
            if item not in RANDOM_KEYFRAME_MODES:
                raise ValueError(f"Unexpected random-K metric mode: {item}")
            trajectory = model.generate_camera_trajectory(
                subject_trajectory=subject, subject_volume=volume,
                camera_trajectory=camera, src_key_mask=source_mask, padding_mask=padding,
                memory_teacher_forcing_ratio=(0.0 if item == RANDOM_KEYFRAME_MODES[0] else 0.5),
                caption_embedding=caption,
            )["reconstructed"]
            _update_generation_metrics(
                metric_callback, item, trajectory, reference[1], reference[2], reference[3],
                ref_clatr, text_clatr, clatr_extractor, ref_model, batch,
                generated_padding_mask=padding,
            )
            _update_geometry_metrics(
                metric_callback, item, trajectory, view, batch,
                generated_padding_mask=padding, known_mask=~source_mask,
                settings=geometry_settings,
            )
            generated["items"][item] = {
                "trajectory": trajectory.detach().cpu(), "source_mask": source_mask.cpu(),
                "padding_mask": padding.cpu(),
            }
        if cache_dir is not None:
            path = Path(cache_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(generated, path / f"batch_{batch_index:05d}.pt")
        batch_index += 1
        return generated

    def write_metrics(cfg, *args, **kwargs):
        original_writer(cfg, *args, **kwargs)
        tag = "lens_craft"
        if cfg.get("variant"):
            tag += "_" + str(cfg.variant)
        if cfg.get("eval_set"):
            tag += "_" + str(cfg.eval_set)
        output = Path(cfg.output_dir) / f"metrics_{tag}.json"
        payload = json.loads(output.read_text())
        payload["keyframe_protocol"] = {
            "version": "paper-random-k-v1", "seed": int(cfg.keyframes.seed),
            "sampling": "independent uniform integer K in [1,10] per sample; uniform valid-frame subset",
            "seed_indexing": "seed + sample offset within cohort, modulo 2**32",
            "count_policy": "min(sampled K, valid frame count)",
            "same_masks_across_modes": True,
            "aggregation": "pooled per-sample generated trajectories; no averaging fixed-K metrics",
            "requested_count_histogram": dict(sorted(requested_histogram.items())),
            "actual_count_histogram": dict(sorted(actual_histogram.items())),
            "sample_count": sum(requested_histogram.values()),
            "sampling_manifest_sha256": hashlib.sha256(json.dumps(recorded_masks).encode()).hexdigest(),
        }
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    evaluator._select_metric_items = lambda *args, **kwargs: list(RANDOM_KEYFRAME_MODES)
    evaluator.test_batch = evaluate_batch
    evaluator._write_metrics_json = write_metrics


def equal_length_support(model_type, et_type, frames, batch_size):
    """Reject changes that would silently change a checkpoint's learned shape."""
    if model_type == "et" and et_type in ("adaln", "incontext") and frames != 300:
        return False, "Released checkpoint projects a flattened 300-frame subject path (900 values); a 30-frame subject has 90 values. Resizing learned projections or retaining 300-frame conditioning is not equal-input evaluation."
    if model_type == "gendop" and batch_size != 1:
        return False, "Released upstream GenDoP generation requires batch size 1; its adapter serializes larger batches."
    if model_type == "lens_craft" and 2 * frames + 1 > 5000:
        return False, "Subject and camera tokens exceed LensCraft's 5000-position sinusoidal buffer."
    return True, None


def configure_native_length(model, model_type, frames):
    """Change only sequence metadata; never resize learned parameter tensors.

    GenDoP's adapter normally repeats the last pose after early EOS. This
    benchmark rejects such outputs before that adapter fallback can run.
    """
    observations = []
    metadata = {"requested_frames": frames, "weights_changed": False,
                "post_generation_padding_or_trimming": False}
    if model_type == "lens_craft":
        metadata["checkpoint_native_frames"] = int(model.decoder.seq_length)
        model.decoder.seq_length = frames
        metadata["metadata_changes"] = {"decoder.seq_length": frames}
    elif model_type == "gendop":
        metadata["checkpoint_native_frames"] = int(model.pose_length)
        model.pose_length = frames
        model.opt.pose_length = frames
        model.opt.test_max_seq_length = frames * 10 + 1
        metadata["metadata_changes"] = {"pose_length": frames, "opt.pose_length": frames,
                                        "opt.test_max_seq_length": frames * 10 + 1}
        # The native wrapper reserves one extra token for EOS, which can instead
        # become an extra coordinate that its adapter silently trims. A fixed
        # output-length benchmark requests exactly N coordinate tokens from the
        # autoregressive sampler itself, with standard HF min/max token bounds.
        # Sampling probabilities/top-k stay native; only the stopping rule changes.
        original_decode = model.model.mesh_decoder.generate
        def decode(*args, **kwargs):
            kwargs["min_new_tokens"] = frames * 10
            kwargs["max_new_tokens"] = frames * 10
            return original_decode(*args, **kwargs)
        model.model.mesh_decoder.generate = decode
        metadata["fixed_length_decoding"] = {
            "min_new_tokens": frames * 10, "max_new_tokens": frames * 10,
            "eos_policy": "Standard Hugging Face minimum-new-token constraint suppresses EOS until the coordinate budget is reached",
            "sampling": "Native stochastic top-k policy retained; stopping rule explicitly constrained for equal-length timing",
        }
        original = model.model.generate
        def generate(*args, **kwargs):
            tokens = original(*args, **kwargs)
            counts = [len(sequence) for sequence in tokens]
            observations.append(counts)
            if any(count != frames * 10 for count in counts):
                raise UnsupportedGenerationLength(
                    f"GenDoP emitted {counts} raw coordinate tokens for requested {frames} frames "
                    f"({frames * 10} tokens). Early EOS or malformed length cannot be padded/repeated "
                    "into a qualified efficiency measurement."
                )
            return tokens
        model.model.generate = generate
        metadata["expected_raw_coordinate_tokens"] = frames * 10
        metadata["raw_coordinate_token_counts"] = observations
        metadata["raw_count_scope"] = "After native EOS removal, before any adapter padding/trimming/fallback"
    elif model_type == "ccdm":
        metadata["checkpoint_native_frames"] = 300
        if model.seq_len != frames:
            raise ValueError("CCDM sequence length must be configured before loading")
        metadata["metadata_changes"] = {"seq_len": frames}
    elif model_type == "et":
        metadata["checkpoint_native_frames"] = int(model.diffuser.net.num_cams)
        for network in (model.diffuser.net, model.diffuser.ema.ema_model):
            network.num_cams = frames
            if hasattr(network, "model"):
                network.model.num_cams = frames
        metadata["metadata_changes"] = {"diffuser network num_cams": frames}
    metadata["extrapolates_checkpoint_native_length"] = metadata["checkpoint_native_frames"] != frames
    return metadata


def validate_native_shape(output, frames, batch_size):
    if not torch.is_tensor(output) or output.ndim < 3:
        raise ValueError("Generation must return a tensor with [batch, frames, ...] dimensions")
    if tuple(output.shape[:2]) != (batch_size, frames):
        raise ValueError(f"Native generated shape {tuple(output.shape[:2])} differs from requested {(batch_size, frames)}; post-generation trimming is prohibited")
    if not torch.isfinite(output).all():
        raise ValueError("Native generation contains nonfinite values")


def profile_generation_flops(generate, batch_size):
    """Count observed operations in the real generation graph, not model.forward."""
    from torch.utils.flop_counter import FlopCounterMode

    class CounterWithCoverage(FlopCounterMode):
        def __init__(self):
            super().__init__(display=False)
            self.uncounted = set()

        def _count_flops(self, func_packet, out, args, kwargs):
            if func_packet not in self.flop_registry:
                self.uncounted.add(str(func_packet))
            return super()._count_flops(func_packet, out, args, kwargs)

    with torch.no_grad(), CounterWithCoverage() as counter:
        output = generate()
        if torch.is_tensor(output) and output.device.type == "cuda":
            torch.cuda.synchronize(output.device)
    counted = {str(operation): int(value) for operation, value in counter.get_flop_counts().get("Global", {}).items()}
    total = counter.get_total_flops()
    return {
        "gflops_per_traj": total / 1e9 / batch_size if total else None,
        "flop_count_is_lower_bound": True,
        "flops_protocol": {
            "implementation": "torch.utils.flop_counter.FlopCounterMode; streaming dispatcher accounting",
            "scope": "one actual adapter generation call, including all diffusion steps/autoregressive tokens",
            "multiply_add_convention": "PyTorch FLOP counter: matrix multiply counts multiply and add separately",
            "counted_operations": counted, "operations_without_flop_estimate": sorted(counter.uncounted),
            "limitation": "Only registered PyTorch operation formulas are counted. Unregistered fused, normalization, elementwise, custom and other unestimated kernels are omitted; this is an observed lower bound, not total FLOPs. Instrumentation is outside timing measurements.",
        },
    }
