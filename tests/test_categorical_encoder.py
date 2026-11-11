"""The camera encoder exposes categorical probabilities, never model states."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.encoder import Encoder
from models.categorical_latent import FIELD_WIDTHS, fixed_query_codes, valid_category_mask


def tiny_encoder(**options):
    torch.manual_seed(42)
    arguments = dict(input_dim=6, latent_dim=16, nhead=2,
                     num_encoder_layers=1, dim_feedforward=32,
                     dropout_rate=0.0, num_query_tokens=10)
    arguments.update(options)
    return Encoder(**arguments).cpu().eval()


def inputs(width=16):
    generator = torch.Generator().manual_seed(7)
    return (torch.randn(2, 5, 6, generator=generator),
            torch.randn(2, 6, width, generator=generator))


def test_standard_configuration_has_four_256d_layers_and_only_ten_fixed_queries():
    encoder = Encoder().cpu().eval()
    layers = encoder.transformer_encoder.layers
    assert len(layers) == 4
    assert all(layer.self_attn.embed_dim == 256 for layer in layers)
    assert all(layer.self_attn.num_heads == 8 for layer in layers)
    assert all(layer.linear1.out_features == 1024 for layer in layers)
    assert encoder.query_tokens.shape == (10, 1, 256)
    assert encoder.semantic_projection.in_features == 256
    assert encoder.semantic_projection.out_features == 21
    assert "query_tokens" not in dict(encoder.named_parameters())
    assert not any(isinstance(layer, nn.Embedding) for layer in encoder.modules())
    torch.testing.assert_close(encoder.query_tokens[:, 0], fixed_query_codes(width=256))
    camera, subject = inputs(width=256)
    with torch.no_grad():
        probabilities = encoder(camera, subject)
    assert probabilities.shape == (10, 2, 21)
    assert encoder.semantic_dim == 93


def test_probabilities_and_stable_logits_have_exactly_93_valid_coordinates():
    encoder = tiny_encoder()
    probabilities, logits = encoder.encode_with_logits(*inputs())
    valid = valid_category_mask()[:, None, :].expand_as(probabilities)
    assert valid_category_mask().sum().item() == 93
    assert (~valid_category_mask()).sum().item() == 117
    assert probabilities.shape == logits.shape == (10, 2, 21)
    assert torch.equal(probabilities[~valid], torch.zeros_like(probabilities[~valid]))
    assert torch.isneginf(logits[~valid]).all()
    assert torch.isfinite(logits[valid]).all()
    assert (probabilities[valid] > 0).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(10, 2))
    torch.testing.assert_close(probabilities, logits.softmax(-1))
    # Both native motion labels belong to field four's valid categorical range.
    assert torch.isfinite(logits[4, :, 19:21]).all()
    assert torch.equal(encoder(*inputs()), probabilities)


@pytest.mark.parametrize("filler", [float("nan"), float("inf"), -98765.0])
def test_hidden_camera_and_subject_fillers_do_not_change_semantic_outputs(filler):
    encoder = tiny_encoder()
    camera, subject = inputs()
    camera_mask = torch.tensor([[False, True, True, False, True],
                                [True, True, True, True, True]])
    subject_mask = torch.tensor([[False, False, True, True, True],
                                 [True, True, True, True, True]])
    subject_tokens = torch.cat([subject_mask, torch.zeros(2, 1, dtype=torch.bool)], 1)
    zero_camera = camera.masked_fill(camera_mask[..., None], 0)
    zero_subject = subject.masked_fill(subject_tokens[..., None], 0)
    hidden_camera = camera.masked_fill(camera_mask[..., None], filler)
    hidden_subject = subject.masked_fill(subject_tokens[..., None], filler)
    with torch.no_grad():
        expected = encoder(zero_camera, zero_subject, camera_mask, subject_mask)
        actual = encoder(hidden_camera, hidden_subject, camera_mask, subject_mask)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mask_order_keeps_all_ten_queries_and_the_volume_visible():
    encoder = tiny_encoder()
    camera, subject = inputs()
    camera_mask = torch.tensor([[False, True, False, True, True],
                                [True, False, False, False, True]])
    subject_mask = torch.tensor([[False, False, True, True, True],
                                 [False, True, False, True, True]])
    seen = []
    hook = encoder.transformer_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["src_key_padding_mask"].clone()),
        with_kwargs=True,
    )
    try:
        encoder(camera, subject, camera_mask, subject_mask)
    finally:
        hook.remove()
    expected = torch.cat([torch.zeros(2, 10, dtype=torch.bool), subject_mask,
                          torch.zeros(2, 1, dtype=torch.bool), camera_mask], dim=1)
    assert torch.equal(seen[0], expected)
    assert not seen[0][:, :10].any()
    assert not seen[0][:, 15].any()


def test_ce_gradients_reach_shared_head_and_visible_inputs_without_padding_gradients():
    encoder = tiny_encoder(num_encoder_layers=4).train()
    camera, subject = inputs()
    camera_mask = torch.tensor([[False, True, False, True, False],
                                [True, False, True, False, True]])
    subject_mask = torch.tensor([[False, False, False, True, True],
                                 [False, False, True, True, True]])
    subject_tokens = torch.cat([subject_mask, torch.zeros(2, 1, dtype=torch.bool)], 1)
    camera = camera.masked_fill(camera_mask[..., None], float("nan")).requires_grad_()
    subject = subject.masked_fill(subject_tokens[..., None], float("nan")).requires_grad_()
    probabilities, logits = encoder.encode_with_logits(camera, subject, camera_mask, subject_mask)
    targets = torch.tensor([[width - 1 for width in FIELD_WIDTHS], [0] * 10])
    loss = F.cross_entropy(logits.permute(1, 0, 2).reshape(-1, 21), targets.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (encoder.semantic_projection.weight, encoder.input_projection.weight,
                      encoder.transformer_encoder.layers[0].self_attn.in_proj_weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.isfinite(camera.grad).all()
    assert torch.isfinite(subject.grad).all()
    assert torch.equal(camera.grad[camera_mask], torch.zeros_like(camera.grad[camera_mask]))
    assert torch.equal(subject.grad[subject_tokens], torch.zeros_like(subject.grad[subject_tokens]))
    assert not encoder.query_tokens.requires_grad
    assert probabilities.shape[-1] == 21
    assert not any(name.startswith("last_") for name in vars(encoder))


@pytest.mark.parametrize("query_count", [1, 21, 46])
def test_rejects_legacy_or_pooled_query_counts(query_count):
    with pytest.raises(ValueError, match="exactly ten"):
        tiny_encoder(num_query_tokens=query_count)


@pytest.mark.parametrize("case", ["camera", "subject_width", "volume", "camera_mask", "subject_mask", "mask_type"])
def test_rejects_shapes_that_would_change_field_or_mask_meaning(case):
    encoder = tiny_encoder()
    camera, subject = inputs()
    kwargs = {}
    if case == "camera":
        camera = camera[..., :3]
    elif case == "subject_width":
        subject = subject[..., :15]
    elif case == "volume":
        subject = subject[:, :-1]
    elif case == "camera_mask":
        kwargs["src_key_padding_mask"] = torch.zeros(2, 6, dtype=torch.bool)
    elif case == "subject_mask":
        kwargs["subject_key_padding_mask"] = torch.zeros(2, 6, dtype=torch.bool)
    else:
        kwargs["src_key_padding_mask"] = torch.zeros(2, 5)
    with pytest.raises(ValueError):
        encoder(camera, subject, **kwargs)
