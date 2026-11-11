"""End-to-end component checks for the sole 93-value semantic bottleneck."""
import pytest
import torch
from torch import nn

from models.decoder import Decoder
from models.encoder import Encoder
from models.categorical_latent import valid_category_mask, validate_semantic


@pytest.fixture
def components():
    """Keep model width small while preserving the actual 10-by-21 interface."""
    torch.manual_seed(42)
    encoder = Encoder(
        input_dim=6, latent_dim=16, nhead=2, num_encoder_layers=1,
        dim_feedforward=32, dropout_rate=0.0,
    ).cpu().eval()
    decoder = Decoder(
        output_dim=12, latent_dim=16, nhead=2, num_decoder_layers=2,
        dim_feedforward=32, dropout_rate=0.0, seq_length=5,
    ).cpu().eval()
    subject_projection = nn.Linear(6, 16)
    volume_projection = nn.Linear(3, 16)
    generator = torch.Generator().manual_seed(7)
    camera = torch.randn(2, 5, 6, generator=generator)
    subject = torch.randn(2, 5, 6, generator=generator)
    volume = torch.randn(2, 1, 3, generator=generator)
    subject_embedding = torch.cat([
        subject_projection(subject), volume_projection(volume),
    ], dim=1)
    return encoder, decoder, camera, subject_embedding


def test_end_to_end_decoder_attends_to_21d_fixed_coded_semantics_without_mutation(components):
    encoder, decoder, camera, subject_embedding = components
    semantic = encoder(camera, subject_embedding)
    original = semantic.detach().clone()
    validate_semantic(semantic)
    valid = valid_category_mask()[:, None, :].expand_as(semantic)
    assert valid_category_mask().sum().item() == 93
    assert (~valid_category_mask()).sum().item() == 117
    assert semantic.shape == (10, 2, 21)
    assert torch.equal(semantic[~valid], torch.zeros_like(semantic[~valid]))
    torch.testing.assert_close(semantic.sum(-1), torch.ones(10, 2))

    presence = torch.ones(2, 10, dtype=torch.bool)
    expected_memory, expected_padding = decoder.prepare_attention_memory(semantic, presence)
    expected_codes = decoder.field_codes.to(semantic)[:, None, :]
    torch.testing.assert_close(expected_memory, semantic + expected_codes)
    assert not expected_padding.any()
    assert not decoder.field_codes.requires_grad
    assert "field_codes" not in dict(decoder.named_parameters())
    calls = []

    def capture_attention(module, args, kwargs):
        query, key, value = args[:3]
        calls.append((module.kdim, module.vdim, query.detach().clone(),
                      key.detach().clone(), value.detach().clone()))

    hooks = [layer.multihead_attn.register_forward_pre_hook(capture_attention, with_kwargs=True)
             for layer in decoder.transformer_decoder.layers]
    try:
        output = decoder(memory=semantic, subject_embedding=subject_embedding,
                         memory_presence_mask=presence)
    finally:
        for hook in hooks:
            hook.remove()
    assert output.shape == (2, 5, 12)
    assert torch.isfinite(output).all()
    assert len(calls) == len(decoder.transformer_decoder.layers)
    for key_width, value_width, query, key, value in calls:
        assert key_width == value_width == 21
        assert query.shape[-1] == 16
        assert key.shape == value.shape == (10, 2, 21)
        torch.testing.assert_close(key, expected_memory, atol=0, rtol=0)
        torch.testing.assert_close(value, expected_memory, atol=0, rtol=0)
        assert torch.equal(key[~valid], torch.zeros_like(key[~valid]))
    assert torch.equal(semantic.detach(), original)


def test_pose_loss_gradients_cross_the_semantic_probabilities_into_the_camera_encoder(components):
    encoder, decoder, camera, subject_embedding = components
    camera.requires_grad_()
    semantic = encoder(camera, subject_embedding)
    semantic.retain_grad()
    output = decoder(memory=semantic, subject_embedding=subject_embedding)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert semantic.grad is not None
    assert torch.isfinite(semantic.grad).all()
    assert semantic.grad.abs().sum() > 0
    valid = valid_category_mask()[:, None, :].expand_as(semantic)
    assert torch.equal(semantic.grad[~valid], torch.zeros_like(semantic.grad[~valid]))
    for parameter in (encoder.semantic_projection.weight, encoder.input_projection.weight,
                      encoder.transformer_encoder.layers[0].self_attn.in_proj_weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert camera.grad is not None
    assert torch.isfinite(camera.grad).all()
    assert camera.grad.abs().sum() > 0


def test_detaching_semantics_blocks_every_camera_encoder_gradient(components):
    encoder, decoder, camera, subject_embedding = components
    camera.requires_grad_()
    semantic = encoder(camera, subject_embedding)
    assert semantic.grad_fn is not None
    output = decoder(memory=semantic.detach(), subject_embedding=subject_embedding)
    output.square().mean().backward()
    # The decoder did backpropagate; the semantic detach alone cuts the entire
    # camera path, including its shared categorical projection and Transformer.
    assert decoder.output_projection.weight.grad is not None
    assert decoder.output_projection.weight.grad.abs().sum() > 0
    assert camera.grad is None
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_replaying_the_same_semantics_after_other_camera_inputs_is_identical(components):
    encoder, decoder, camera, subject_embedding = components
    changed_camera = camera.flip(1) * 7 + torch.linspace(-10, 10, 6)
    captured = []
    hook = encoder.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
    try:
        with torch.no_grad():
            semantic = encoder(camera, subject_embedding)
            first = decoder(memory=semantic, subject_embedding=subject_embedding)
            changed_semantic = encoder(changed_camera, subject_embedding)
            changed = decoder(memory=changed_semantic, subject_embedding=subject_embedding)
            # A subsequent encoder call must not install hidden states that
            # bypass the explicitly replayed 21D semantic tokens.
            replayed = decoder(memory=semantic, subject_embedding=subject_embedding)
    finally:
        hook.remove()
    assert len(captured) == 2
    assert all(value.shape == (10, 2, 21) for value in captured)
    torch.testing.assert_close(captured[0], semantic, atol=0, rtol=0)
    torch.testing.assert_close(captured[1], changed_semantic, atol=0, rtol=0)
    assert not torch.equal(semantic, changed_semantic)
    assert not torch.equal(first, changed)
    torch.testing.assert_close(first, replayed, atol=0, rtol=0)
    assert not any(name.startswith("last_") for name in vars(encoder))
    assert not any(name.startswith("last_") for name in vars(decoder))
