"""Direct categorical attention, absence handling and source decoder contracts."""
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
from torch import nn

from models.decoder import Decoder
from models.categorical_latent import FIELD_WIDTHS, from_ids, valid_category_mask


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def decoder(frames=4):
    torch.manual_seed(17)
    return Decoder(
        output_dim=12, latent_dim=16, nhead=2, num_decoder_layers=1,
        dim_feedforward=32, dropout_rate=0., seq_length=frames,
    ).eval()


def inputs(model, batch=2):
    ids = torch.tensor([[index % width for index, width in enumerate(FIELD_WIDTHS)]] * batch)
    memory, presence = from_ids(ids)
    subject = torch.linspace(-.3, .3, batch * (model.seq_length + 1) * model.latent_dim).reshape(
        batch, model.seq_length + 1, model.latent_dim)
    return memory, presence, subject


def test_default_architecture_attends_to_21d_without_an_external_memory_projection():
    model = Decoder(dropout_rate=0.)
    assert len(model.transformer_decoder.layers) == 4
    for layer in model.transformer_decoder.layers:
        attention = layer.multihead_attn
        assert attention.embed_dim == 256 and attention.num_heads == 8
        assert attention.kdim == attention.vdim == 21
        assert attention.k_proj_weight.shape == attention.v_proj_weight.shape == (256, 21)
        assert layer.linear1.out_features == 1024
    assert not any(isinstance(module, nn.Linear)
                   and module.in_features == 21 and module.out_features == 256
                   for module in model.modules())
    assert model.output_projection.out_features == 12
    assert all('field_code' not in name and 'field_id' not in name
               for name, _ in model.named_parameters())


def test_field_codes_are_fixed_category_padded_and_do_not_mutate_semantics():
    model = decoder()
    memory, presence, subject = inputs(model)
    original, original_presence = memory.clone(), presence.clone()
    prepared, padding = model.prepare_attention_memory(memory, presence)
    valid = valid_category_mask()[:, None, :].expand_as(prepared)
    assert torch.count_nonzero(prepared.masked_select(~valid)) == 0
    assert torch.count_nonzero(model.field_codes.masked_select(~valid_category_mask())) == 0
    assert not model.field_codes.requires_grad
    assert torch.equal(memory, original) and torch.equal(presence, original_presence)
    assert not padding.any()
    assert not torch.equal(prepared, memory)
    assert model(memory, subject, memory_presence_mask=presence).shape == (2, 4, 12)
    assert torch.equal(memory, original)


def test_field_identity_makes_swapping_equal_width_fields_observable():
    model = decoder()
    memory, presence, subject = inputs(model, batch=1)
    memory[0, 0].zero_()
    memory[0, 0, 0] = 1
    memory[6, 0].zero_()
    memory[6, 0, 1] = 1
    permuted = memory.clone()
    permuted[[0, 6]] = permuted[[6, 0]]
    with torch.inference_mode():
        first = model(memory, subject, memory_presence_mask=presence)
        second = model(permuted, subject, memory_presence_mask=presence)
    assert not torch.allclose(first, second, rtol=1e-6, atol=1e-7)


def test_unknown_fields_are_masked_and_all_none_has_a_code_free_zero_fallback():
    model = decoder()
    memory, presence, subject = inputs(model)
    presence[0, 2] = False
    presence[1] = False
    changed = memory.clone()
    changed[2, 0] = float('nan')
    changed[:, 1] = float('nan')
    original_presence = presence.clone()
    prepared, padding = model.prepare_attention_memory(changed, presence)
    assert torch.equal(prepared[:, 1], torch.zeros_like(prepared[:, 1]))
    assert torch.equal(padding[1], torch.tensor([False] + [True] * 9))
    assert padding[0, 2]
    assert torch.equal(presence, original_presence)
    with torch.inference_mode():
        clean = model(memory, subject, memory_presence_mask=presence)
        hidden_nan = model(changed, subject, memory_presence_mask=presence)
    assert torch.isfinite(hidden_nan).all()
    torch.testing.assert_close(clean, hidden_nan)
    # Public callers may omit presence; zero fields still mean DONTCARE.
    implicit_memory, implicit_padding = model.prepare_attention_memory(torch.zeros_like(memory))
    assert torch.count_nonzero(implicit_memory) == 0
    assert torch.equal(implicit_padding, torch.tensor([[False] + [True] * 9] * 2))


def test_gradients_reach_valid_semantics_and_model_while_all_none_remains_finite():
    model = decoder()
    memory, presence, subject = inputs(model)
    presence[1] = False
    memory = memory.clone().requires_grad_()
    output = model(memory, subject, memory_presence_mask=presence)
    output.square().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(memory.grad).all()
    assert memory.grad[:, 0].abs().sum() > 0
    assert memory.grad[:, 1].abs().sum() == 0
    valid = valid_category_mask()[:, None, :].expand_as(memory)
    assert torch.count_nonzero(memory.grad.masked_select(~valid)) == 0
    assert model.transformer_decoder.layers[0].multihead_attn.k_proj_weight.grad.abs().sum() > 0
    assert model.output_projection.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


@pytest.mark.parametrize('defect', ['negative', 'nonunit', 'padding', 'nonfinite'])
def test_nonzero_memory_fields_must_be_bounded_categorical_probabilities(defect):
    model = decoder()
    memory, presence, subject = inputs(model)
    if defect == 'negative':
        memory[0, 0, 0] = -1
    elif defect == 'nonunit':
        memory[0, 0] *= .5
    elif defect == 'padding':
        memory[0, 0, FIELD_WIDTHS[0]] = .1
    else:
        memory[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        model(memory, subject, memory_presence_mask=presence)


def test_zero_dropped_fields_remain_valid_before_attention_policy_is_applied():
    model = decoder()
    memory, presence, subject = inputs(model)
    memory[0] = 0
    prepared, padding = model.prepare_attention_memory(memory, presence)
    assert padding[:, 0].all() and torch.count_nonzero(prepared[0]) == 0
    result = model(memory, subject, memory_presence_mask=presence)
    assert torch.isfinite(result).all()


def test_padding_and_hidden_camera_pose_values_cannot_change_visible_predictions():
    model = decoder()
    memory, presence, subject = inputs(model, batch=1)
    padding = torch.tensor([[False, False, True, True]])
    known = torch.tensor([[True, False, True, False]])
    poses = torch.linspace(-.1, .1, 48).reshape(1, 4, 12)
    hidden_nan = poses.masked_fill(~(known & ~padding)[..., None], float('nan'))
    changed_subject = subject.clone()
    changed_subject[:, 2:4] = 100000
    with torch.inference_mode():
        reference = model(memory, subject, tgt_key_padding_mask=padding,
            known_camera_poses=poses, known_pose_mask=known, memory_presence_mask=presence)
        changed = model(memory, changed_subject, tgt_key_padding_mask=padding,
            known_camera_poses=hidden_nan, known_pose_mask=known, memory_presence_mask=presence)
    torch.testing.assert_close(reference[~padding], changed[~padding])
    assert torch.isfinite(changed).all()


@pytest.mark.parametrize('mode', ['single_step', 'autoregressive'])
def test_padded_nan_inf_subject_context_is_sanitized_before_attention_and_backward(mode):
    model = decoder()
    memory, presence, subject = inputs(model, batch=1)
    padding = torch.tensor([[False, False, True, True]])
    subject[:, 2] = float('nan')
    subject[:, 3] = float('inf')
    subject.requires_grad_()
    memory.requires_grad_()
    original = subject.detach().clone()
    result = model(memory, subject, decode_mode=mode,
        tgt_key_padding_mask=padding, memory_presence_mask=presence)
    result[~padding].square().mean().backward()
    assert torch.isfinite(result).all()
    assert torch.isfinite(subject.grad).all() and torch.isfinite(memory.grad).all()
    assert torch.count_nonzero(subject.grad[:, 2:4]) == 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
    torch.testing.assert_close(subject.detach(), original, equal_nan=True)


def test_autoregressive_future_targets_are_causal_and_feedback_honors_known_poses():
    model = decoder()
    memory, presence, subject = inputs(model, batch=1)
    target = torch.randn(1, 4, 12)
    later_target = target.clone()
    later_target[:, 2] += 100
    with torch.inference_mode():
        first = model(memory, subject, decode_mode='autoregressive', target=target,
            teacher_forcing_ratio=1., memory_presence_mask=presence)
        changed = model(memory, subject, decode_mode='autoregressive', target=later_target,
            teacher_forcing_ratio=1., memory_presence_mask=presence)
    torch.testing.assert_close(first[:, :3], changed[:, :3])
    assert not torch.allclose(first[:, 3], changed[:, 3])
    known = torch.tensor([[True, True, False, False]])
    padding = torch.tensor([[False, True, False, False]])
    poses = torch.full((1, 4, 12), float('nan'))
    poses[:, 0] = 3
    calls = []

    def feedback(current):
        calls.append(current)
        return torch.full_like(current, 7)

    with patch.object(model, 'prepare_decoder_inputs_with_positioning',
                      wraps=model.prepare_decoder_inputs_with_positioning) as prepare:
        output = model(memory, subject, decode_mode='autoregressive',
            feedback_transform=feedback, known_camera_poses=poses, known_pose_mask=known,
            tgt_key_padding_mask=padding, memory_presence_mask=presence)
    assert len(calls) == 3 and torch.isfinite(output).all()
    assert torch.equal(prepare.call_args_list[1].args[0][:, 0], torch.full((1, 12), 3.))
    assert torch.equal(prepare.call_args_list[2].args[0][:, 1], torch.full((1, 12), 7.))


@pytest.mark.parametrize('mask', [torch.ones(2, 10), torch.ones(2, 9, dtype=torch.bool)])
def test_memory_presence_requires_a_boolean_mask_for_all_ten_fields(mask):
    model = decoder()
    memory, _, subject = inputs(model)
    with pytest.raises(ValueError, match='memory_presence_mask'):
        model(memory, subject, memory_presence_mask=mask)
