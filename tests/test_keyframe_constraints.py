"""Supplied camera keyframes are exact constraints on every output representation."""

import math

import pytest
import torch

from models.camera_trajectory_model import LensCraft
from models.categorical_latent import FIELD_NAMES, FIELD_WIDTHS, from_ids


OUTPUTS = (
    "reconstructed",
    "reconstructed_raw_matrix",
    "reconstructed_rot_matrix",
)


def tiny_model(**options):
    torch.manual_seed(7)
    return LensCraft(
        nhead=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout_rate=0., seq_length=5, latent_dim=16,
        **options,
    ).cpu().eval()


@pytest.fixture
def camera():
    values = torch.arange(60, dtype=torch.float32).reshape(2, 5, 6) / 100
    values[..., :3] += 10
    # Equivalent rotations must not canonicalize the user's supplied angles.
    values[0, 0, 3] += 2 * math.pi
    values[1, 4, 4] -= 2 * math.pi
    return values


@pytest.fixture
def known():
    return torch.tensor([
        [True, False, False, True, False],
        [False, True, False, False, True],
    ])


def run(model, api, camera, **kwargs):
    subject = torch.linspace(-.2, .2, camera.numel()).reshape_as(camera)
    volume = torch.ones(camera.shape[0], 1, 3)
    if api == "forward":
        if "padding_mask" in kwargs:
            kwargs["tgt_key_padding_mask"] = kwargs.pop("padding_mask")
        return model(
            src=camera, subject_trajectory=subject, subject_volume=volume,
            **kwargs,
        )
    return model.generate_camera_trajectory(
        camera_trajectory=camera, subject_trajectory=subject,
        subject_volume=volume, **kwargs,
    )


def assert_fixed(model, output, camera, known):
    assert torch.equal(output["known_camera_mask"], known)
    assert torch.equal(output["reconstructed"][known], camera[known])
    supplied_matrices = model._euler6_to_mat12(camera)
    for name in OUTPUTS[1:]:
        assert torch.equal(output[name][known], supplied_matrices[known])


@pytest.mark.parametrize("api", ["forward", "generate"])
@pytest.mark.parametrize("direct_pose", [False, True])
def test_noncontiguous_per_sample_keyframes_are_exact_without_changing_unknown_predictions(
    camera, known, api, direct_pose,
):
    model = tiny_model(keyframe_pose_conditioning=direct_pose)
    original_camera, original_mask = camera.clone(), known.clone()
    raw_predictions = []
    handle = model.decoder.register_forward_hook(
        lambda module, args, output: raw_predictions.append(output.detach().clone())
    )
    try:
        constrained = run(model, api, camera, known_camera_mask=known)
    finally:
        handle.remove()

    # Compare against the decoder prediction from this same explicit-keyframe
    # call. With direct pose queries enabled, omitting known_camera_mask would
    # also remove those query inputs and describe a different inference task.
    assert len(raw_predictions) == 1
    raw = raw_predictions[0]
    reconstructed, rotation_matrix = model._finalize_trajectory(raw)
    before_constraints = {
        "reconstructed": reconstructed,
        "reconstructed_raw_matrix": raw,
        "reconstructed_rot_matrix": rotation_matrix,
    }

    assert_fixed(model, constrained, camera, known)
    for name in OUTPUTS:
        torch.testing.assert_close(constrained[name][~known], before_constraints[name][~known])
        assert torch.isfinite(constrained[name]).all()
    assert torch.equal(camera, original_camera)
    assert torch.equal(known, original_mask)


@pytest.mark.parametrize("direct_pose", [False, True])
def test_hidden_nan_fillers_are_equivalent_to_zero_fillers(camera, known, direct_pose):
    model = tiny_model(keyframe_pose_conditioning=direct_pose)
    zeros = camera.masked_fill(~known[..., None], 0)
    missing = camera.masked_fill(~known[..., None], float("nan"))
    zero_output = run(model, "forward", zeros, known_camera_mask=known)
    nan_output = run(model, "forward", missing, known_camera_mask=known)

    assert_fixed(model, nan_output, camera, known)
    for name in (*OUTPUTS, "embeddings"):
        assert torch.isfinite(nan_output[name]).all()
        torch.testing.assert_close(nan_output[name], zero_output[name])


@pytest.mark.parametrize("api", ["forward", "generate"])
def test_padding_cannot_become_a_keyframe_and_mask_disagreements_on_padding_are_ignored(
    camera, known, api,
):
    model = tiny_model()
    padding = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, False, True],
    ])
    requested = known | padding
    supplied = camera.masked_fill(padding[..., None], float("nan"))
    output = run(
        model, api, supplied, known_camera_mask=requested,
        src_key_mask=(~requested) | padding, padding_mask=padding,
    )

    assert_fixed(model, output, camera, requested & ~padding)
    for name in OUTPUTS:
        assert torch.isfinite(output[name]).all()


@pytest.mark.parametrize("all_known", [False, True])
@pytest.mark.parametrize("direct_pose", [False, True])
def test_all_and_no_keyframes_are_supported(camera, all_known, direct_pose):
    model = tiny_model(keyframe_pose_conditioning=direct_pose)
    mask = torch.full(camera.shape[:2], all_known, dtype=torch.bool)
    output = run(model, "generate", camera, known_camera_mask=mask)

    assert_fixed(model, output, camera, mask)
    for name in OUTPUTS:
        assert torch.isfinite(output[name]).all()
    if not all_known:
        baseline = run(model, "generate", camera, src_key_mask=~mask)
        for name in OUTPUTS:
            torch.testing.assert_close(output[name], baseline[name])


@pytest.mark.parametrize("api", ["forward", "generate"])
def test_autoregressive_decoder_preserves_explicit_keyframes(camera, known, api):
    model = tiny_model()
    output = run(
        model, api, camera, known_camera_mask=known,
        decode_mode="autoregressive",
    )
    assert_fixed(model, output, camera, known)
    assert all(torch.isfinite(output[name]).all() for name in OUTPUTS)


def test_autoregressive_feedback_uses_supplied_keyframes_for_later_predictions(camera, known):
    model = tiny_model()
    decoder_inputs = []

    def capture_input(module, args):
        decoder_inputs.append(args[0].detach().clone())

    handle = model.decoder.embedding.register_forward_pre_hook(capture_input)
    try:
        run(model, "generate", camera, known_camera_mask=known, decode_mode="autoregressive")
    finally:
        handle.remove()

    assert len(decoder_inputs) == camera.shape[1]
    supplied_matrices = model._euler6_to_mat12(camera)
    for timestep in range(camera.shape[1] - 1):
        visible = known[:, timestep]
        if visible.any():
            assert torch.equal(
                decoder_inputs[timestep + 1][visible, timestep],
                supplied_matrices[visible, timestep],
            )


@pytest.mark.parametrize("output_name", OUTPUTS)
def test_decoder_output_gradients_flow_only_through_unknown_frames(camera, known, output_name):
    model = tiny_model()
    captured = []

    def capture_prediction(module, args, output):
        output.retain_grad()
        captured.append(output)

    handle = model.decoder.register_forward_hook(capture_prediction)
    try:
        output = run(model, "forward", camera, known_camera_mask=known)
        output[output_name].square().sum().backward()
    finally:
        handle.remove()

    gradient = captured[0].grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[known]).item() == 0
    assert gradient[~known].abs().sum().item() > 0


@pytest.mark.parametrize("direct_pose", [False, True])
def test_default_prompt_generation_does_not_copy_or_depend_on_ground_truth_camera(camera, direct_pose):
    model = tiny_model(keyframe_pose_conditioning=direct_pose)
    caption, _ = from_ids(torch.tensor([[width - 1 for width in FIELD_WIDTHS]] * camera.shape[0]))
    first = run(
        model, "generate", camera, caption_embedding=caption,
        memory_teacher_forcing_ratio=1.,
    )
    second = run(
        model, "generate", camera * -3, caption_embedding=caption,
        memory_teacher_forcing_ratio=1.,
    )
    caption_only = model.generate_camera_trajectory(caption_embedding=caption)

    assert not first["known_camera_mask"].any()
    assert not caption_only["known_camera_mask"].any()
    assert not torch.equal(first["reconstructed"], camera)
    for name in OUTPUTS:
        torch.testing.assert_close(first[name], second[name])


def test_default_reconstruction_retains_decoder_predictions(camera):
    model = tiny_model()
    output = run(model, "forward", camera)
    reconstructed, rotations = model._finalize_trajectory(output["reconstructed_raw_matrix"])

    assert not output["known_camera_mask"].any()
    assert not torch.equal(output["reconstructed"], camera)
    torch.testing.assert_close(output["reconstructed"], reconstructed)
    torch.testing.assert_close(output["reconstructed_rot_matrix"], rotations)


def test_explicit_keyframes_stay_fixed_when_caption_supplies_all_memory(camera, known):
    model = tiny_model()
    caption = torch.zeros(model.num_query_tokens, camera.shape[0], 21)
    output = run(
        model, "generate", camera, caption_embedding=caption,
        memory_teacher_forcing_ratio=1., known_camera_mask=known,
    )
    assert_fixed(model, output, camera, known)


def test_pose_query_flag_remains_opt_in_and_checkpoint_parameters_are_compatible(camera, known):
    previous = tiny_model()
    with_pose_queries = tiny_model(keyframe_pose_conditioning=True)
    assert previous.keyframe_pose_conditioning is False
    checkpoint = {key: value.clone() for key, value in previous.state_dict().items()}
    loaded = with_pose_queries.load_state_dict(checkpoint, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    run(previous, "forward", camera, known_camera_mask=known)
    assert set(previous.state_dict()) == set(checkpoint) == set(with_pose_queries.state_dict())
    for key, value in checkpoint.items():
        assert torch.equal(previous.state_dict()[key], value)
        assert torch.equal(with_pose_queries.state_dict()[key], value)


@pytest.mark.parametrize("api", ["forward", "generate"])
@pytest.mark.parametrize("problem", ["shape", "dtype", "not_tensor"])
def test_known_mask_requires_boolean_batch_frame_tensor(camera, api, problem):
    mask = torch.zeros(camera.shape[:2], dtype=torch.bool)
    if problem == "shape":
        mask = mask[:, :-1]
    elif problem == "dtype":
        mask = mask.float()
    else:
        mask = mask.tolist()
    with pytest.raises(ValueError, match="known_camera_mask"):
        run(tiny_model(), api, camera, known_camera_mask=mask)


@pytest.mark.parametrize("api", ["forward", "generate"])
def test_known_mask_and_camera_visibility_must_agree_on_valid_frames(camera, known, api):
    source_mask = ~known
    source_mask[0, 0] = True
    with pytest.raises(ValueError, match="mask"):
        run(tiny_model(), api, camera, known_camera_mask=known, src_key_mask=source_mask)


@pytest.mark.parametrize("api", ["forward", "generate"])
def test_visible_keyframe_values_must_be_finite(camera, known, api):
    camera[0, 0, 3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run(tiny_model(), api, camera, known_camera_mask=known)


def test_keyframe_mask_requires_camera_poses():
    model = tiny_model()
    caption = torch.zeros(model.num_query_tokens, 2, 21)
    with pytest.raises(ValueError, match="camera"):
        model.generate_camera_trajectory(
            caption_embedding=caption,
            known_camera_mask=torch.ones(2, 5, dtype=torch.bool),
        )


def test_trainer_pins_only_explicit_keyframe_modes_and_keeps_cycle_embeddings_finite(camera, monkeypatch):
    from training.base_trainer import (
        BaseTrainer, MaskConfig, NoiseConfig, TeacherForcingConfig,
    )
    from training.conditioning import TRAINING_MODES

    model = tiny_model()
    trainer = BaseTrainer(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.01),
        lr_scheduler=None, loss_module=torch.nn.Identity(),
        noise=NoiseConfig(0., 0.), mask=MaskConfig(0., 0.),
        teacher_forcing_schedule=TeacherForcingConfig(0., 0., 0., 0.),
        compile_enabled=False,
    )
    monkeypatch.setattr(trainer, "_calculate_schedule_parameters", lambda: {
        "noise_std": 0., "mask_ratio": 0., "memory_mask_ratio": 0.,
        "memory_teacher_forcing_ratio": 0., "trajectory_teacher_forcing_ratio": 0.,
    })
    subject = torch.zeros_like(camera)
    volume = torch.ones(camera.shape[0], 1, 3)
    caption = torch.zeros(model.num_query_tokens, camera.shape[0], 21)
    padding = torch.tensor([
        [False, False, False, False, True],
        [False, False, False, False, False],
    ])

    for mode in TRAINING_MODES:
        keyframing = mode.startswith("key_framing")
        output = trainer._forward_step(
            camera, subject, volume, caption, padding, is_training=True,
            compute_cycle_embeddings=True, conditioning_mode=mode,
            keyframe_count=2 if keyframing else None,
            keyframe_sample_seeds=[91, 92],
        )
        assert output["conditioning_mode"] == mode
        assert torch.isfinite(output["cycle_embeddings"]).all()
        assert all(torch.isfinite(output[name]).all() for name in OUTPUTS)
        if keyframing:
            effective = output["known_mask"] & ~padding
            assert effective.sum(1).tolist() == [2, 2]
            assert_fixed(model, output, camera, effective)
        else:
            assert not output["known_camera_mask"].any()
            assert not torch.equal(output["reconstructed"][~padding], camera[~padding])
            if mode == "prompt_generation":
                assert not output["known_mask"].any()


def test_inference_batch_preserves_sampled_keyframes_in_both_keyframe_modes(camera):
    from inferencing.process import inference_batch

    model = tiny_model()
    padding = torch.tensor([
        [False, False, False, False, True],
        [False, False, False, False, False],
    ])
    batch = {
        "camera_trajectory": camera,
        "subject_trajectory": torch.zeros_like(camera),
        "subject_volume": torch.ones(camera.shape[0], 1, 3),
        "padding_mask": padding,
        "text_prompts": ["move the camera", "keep the camera steady"],
        "random_prompt_index": [1, 0],
        "cinematography_prompt_parameters": [[(name, None, -1) for name in FIELD_NAMES] for _ in range(camera.shape[0])],
    }
    results, source, _, _, returned_padding, hidden = inference_batch(
        model, batch, "cpu", dataset_type="simulation", seq_length=5,
        num_keyframes=2, keyframe_sample_seeds=[91, 92],
    )

    assert torch.equal(source, camera)
    assert torch.equal(returned_padding, padding)
    assert (~hidden).sum(1).tolist() == [2, 2]
    assert not ((~hidden) & padding).any()
    for mode in ("key_framing", "key_framing+prompt"):
        assert torch.equal(results[mode][~hidden], source[~hidden])
    assert not torch.equal(results["reconstruction"][~padding], source[~padding])
    assert all(torch.isfinite(generated).all() for generated in results.values())


def test_keyframe_cycle_loss_trains_unknown_predictions_without_moving_supplied_poses(camera, monkeypatch):
    from training.base_trainer import (
        BaseTrainer, MaskConfig, NoiseConfig, TeacherForcingConfig,
    )
    from training.losses.categorical_loss import CategoricalCameraTrajectoryLoss

    model = tiny_model()
    trainer = BaseTrainer(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.01),
        lr_scheduler=None, loss_module=torch.nn.Identity(),
        noise=NoiseConfig(0., 0.), mask=MaskConfig(0., 0.),
        teacher_forcing_schedule=TeacherForcingConfig(0., 0., 0., 0.),
        compile_enabled=False,
    )
    monkeypatch.setattr(trainer, "_calculate_schedule_parameters", lambda: {
        "noise_std": 0., "mask_ratio": 0., "memory_mask_ratio": 0.,
        "memory_teacher_forcing_ratio": 0., "trajectory_teacher_forcing_ratio": 0.,
    })
    padding = torch.tensor([
        [False, False, False, False, True],
        [False, False, False, False, False],
    ])
    target_ids = torch.full((camera.shape[0], 10), -1, dtype=torch.long)
    target_ids[:, 0] = 1
    target_ids[:, 4] = torch.tensor([19, 20])
    target_ids[:, 7] = 3
    semantic_target, label_presence = from_ids(target_ids)
    captured = []

    def capture_prediction(module, args, output):
        output.retain_grad()
        captured.append(output)

    handle = model.decoder.register_forward_hook(capture_prediction)
    try:
        output = trainer._forward_step(
            camera, torch.zeros_like(camera), torch.ones(camera.shape[0], 1, 3),
            semantic_target, padding,
            is_training=True, compute_cycle_embeddings=True,
            conditioning_mode="key_framing", keyframe_count=2,
            keyframe_sample_seeds=[91, 92],
        )
        cycle_objective = CategoricalCameraTrajectoryLoss(
            losses_list={"cycle": 1., "categorical": 0.},
        )
        loss, _ = cycle_objective(
            output, camera, semantic_target, {"prompt_none_mask": label_presence}, padding,
            known_mask=output["known_mask"],
        )
        assert loss > 0
        loss.backward()
    finally:
        handle.remove()

    known = output["known_camera_mask"]
    gradient = captured[0].grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[known | padding]).item() == 0
    assert (gradient[~(known | padding)].norm(dim=-1) > 0).all()
    assert_fixed(model, output, camera, known)


def test_autocast_keeps_supplied_pose_matrices_exact(camera, known):
    model = tiny_model()
    decoder_dtypes = []
    handle = model.decoder.register_forward_hook(
        lambda module, args, output: decoder_dtypes.append(output.dtype)
    )
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = run(model, "forward", camera, known_camera_mask=known)
    finally:
        handle.remove()

    assert decoder_dtypes == [torch.bfloat16]
    assert_fixed(model, output, camera, known)
