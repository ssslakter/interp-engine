"""Qwen Gated DeltaNet points are the recurrence tensors they claim to be."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from interp_engine.address import Address
from interp_engine.gdn import TokenPhase, capture_gdn, intervene_gdn
from interp_engine.model import EagerModel


def _fixture():
    transformers = pytest.importorskip("transformers")
    config_cls = transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig
    mixer_cls = transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet
    config = config_cls(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention"],
    )
    mixer = mixer_cls(config, 0).eval()
    model = object.__new__(EagerModel)
    model.arch = SimpleNamespace(attn_module=lambda _layer: mixer)
    return model, mixer


def _capture(model, mixer, x, names, *, positions=None):
    addresses = [Address(name, 0) for name in names]
    with capture_gdn(model, addresses, prompt_len=x.shape[1], positions=positions, detach=True) as finish:
        output = mixer(x)
        captured = finish()
    return output, {address.name: captured[address] for address in addresses}


def test_noop_instrumentation_preserves_the_forward_and_exposes_the_recurrence() -> None:
    torch.manual_seed(1)
    model, mixer = _fixture()
    x = torch.randn(1, 3, 16)
    baseline = mixer(x)
    names = (
        "gdn_q",
        "gdn_k",
        "gdn_v",
        "gdn_alpha",
        "gdn_beta",
        "gdn_state_write",
        "gdn_state_post",
        "gdn_read",
        "gdn_normed_read",
        "gdn_z",
        "gdn_post_gate",
    )
    output, cap = _capture(model, mixer, x, names, positions=[0, 1, 2])

    torch.testing.assert_close(output, baseline, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cap["gdn_q"].square().sum(-1), torch.ones(1, 3, 2), atol=1e-3, rtol=0)
    torch.testing.assert_close(cap["gdn_k"].square().sum(-1), torch.ones(1, 3, 2), atol=1e-3, rtol=0)
    assert cap["gdn_alpha"].dtype == cap["gdn_read"].dtype == torch.float32
    assert cap["gdn_state_post"].shape == (1, 3, 2, 4, 4)
    torch.testing.assert_close(
        cap["gdn_post_gate"],
        cap["gdn_normed_read"] * torch.nn.functional.silu(cap["gdn_z"]),
    )

    previous = torch.zeros_like(cap["gdn_state_post"][:, 0])
    for position in range(3):
        expected = previous * cap["gdn_alpha"][:, position, :, None, None] + cap["gdn_state_write"][:, position]
        torch.testing.assert_close(cap["gdn_state_post"][:, position], expected)
        previous = cap["gdn_state_post"][:, position]


def test_rank_one_state_edit_has_the_exact_predicted_read_delta() -> None:
    torch.manual_seed(2)
    model, mixer = _fixture()
    x = torch.randn(1, 3, 16)
    _, clean = _capture(model, mixer, x, ("gdn_q", "gdn_read"), positions=[1])
    u = torch.randn(4)
    v = torch.randn(4)
    scale = 0.7

    def edit(state: torch.Tensor, _context):
        return state + scale * u[:, None] * v[None, :]

    with intervene_gdn(
        model,
        {Address("gdn_state_post", 0): edit},
        prompt_token_ids=torch.zeros(1, 3, dtype=torch.long),
        phase=TokenPhase.PREFILL,
        positions=[1],
    ) as result:
        _, changed = _capture(model, mixer, x, ("gdn_q", "gdn_read"), positions=[1])

    predicted = scale * (clean["gdn_q"] @ u[..., None]).squeeze(-1)[..., None] * v / (4**0.5)
    torch.testing.assert_close(changed["gdn_read"] - clean["gdn_read"], predicted, atol=2e-6, rtol=2e-5)
    assert result.fired_positions == (1,)
    assert result.missed_positions == ()


def test_absolute_position_in_the_wrong_phase_is_reported_as_missed() -> None:
    model, mixer = _fixture()
    x = torch.randn(1, 3, 16)

    with intervene_gdn(
        model,
        {Address("gdn_v", 0): lambda value, _context: value + 1},
        prompt_token_ids=torch.zeros(1, 3, dtype=torch.long),
        phase=TokenPhase.DECODE,
        positions=[1],
    ) as result:
        mixer(x)

    assert result.fired_positions == ()
    assert result.missed_positions == (1,)


def test_decode_phase_uses_the_absolute_position_after_prefill() -> None:
    model, mixer = _fixture()
    seen = []

    def record(value, context):
        seen.append(context)
        return value

    def recurrence_inputs(seq):
        return (
            torch.randn(1, seq, 2, 4),
            torch.randn(1, seq, 2, 4),
            torch.randn(1, seq, 2, 4),
            torch.zeros(1, seq, 2),
            torch.ones(1, seq, 2),
        )

    with intervene_gdn(
        model,
        {Address("gdn_v", 0): record},
        prompt_token_ids=torch.zeros(1, 3, dtype=torch.long),
        phase=TokenPhase.DECODE,
        positions=[1, 3],
    ) as result:
        query, key, value, decay, beta = recurrence_inputs(3)
        _, state = mixer.chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        query, key, value, decay, beta = recurrence_inputs(1)
        mixer.recurrent_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
        )

    assert [context.position for context in seen] == [3]
    assert [context.phase for context in seen] == [TokenPhase.DECODE]
    assert result.fired_positions == (3,)
    assert result.missed_positions == (1,)


def test_q_callback_is_not_renormalized_after_intervention() -> None:
    model, mixer = _fixture()
    x = torch.randn(1, 2, 16)

    with intervene_gdn(
        model,
        {Address("gdn_q", 0): lambda query, _context: query * 2},
        prompt_token_ids=torch.zeros(1, 2, dtype=torch.long),
        positions=[1],
    ):
        _, captured = _capture(model, mixer, x, ("gdn_q",), positions=[1])

    torch.testing.assert_close(captured["gdn_q"].square().sum(-1), torch.full((1, 1, 2), 4.0), atol=4e-3, rtol=0)


def test_qk_are_normalized_before_the_fp32_recurrence_promotion() -> None:
    model, mixer = _fixture()
    query = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    decay = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    beta = torch.ones_like(decay)

    with capture_gdn(
        model,
        [Address("gdn_q", 0), Address("gdn_k", 0)],
        prompt_len=2,
        positions=[0, 1],
        detach=True,
    ) as finish:
        mixer.chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            use_qk_l2norm_in_kernel=True,
        )
        captured = finish()

    expected_q = (query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)).float()
    expected_k = (key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)).float()
    torch.testing.assert_close(captured[Address("gdn_q", 0)], expected_q, rtol=0, atol=0)
    torch.testing.assert_close(captured[Address("gdn_k", 0)], expected_k, rtol=0, atol=0)


def test_sparse_prefill_intervention_resumes_the_chunk_kernel_after_the_last_edit() -> None:
    model, mixer = _fixture()
    original_chunk = mixer.chunk_gated_delta_rule
    chunk_lengths = []

    def recording_chunk(*args, **kwargs):
        chunk_lengths.append(args[0].shape[1])
        return original_chunk(*args, **kwargs)

    mixer.chunk_gated_delta_rule = recording_chunk
    x = torch.randn(1, 20, 16)
    baseline = mixer(x)
    chunk_lengths.clear()

    with intervene_gdn(
        model,
        {Address("gdn_state_post", 0): lambda state, _context: state},
        prompt_token_ids=torch.zeros(1, 20, dtype=torch.long),
        phase=TokenPhase.PREFILL,
        positions=[0, 1, 2, 3, 4],
    ):
        changed = mixer(x)

    assert chunk_lengths == [15]
    # Splitting one mathematically equivalent chunk at token 5 slightly changes fp32
    # reassociation, but an identity edit must remain at reference-rounding scale.
    torch.testing.assert_close(changed, baseline, atol=3e-6, rtol=1e-4)


def test_sparse_middle_interventions_chunk_every_untouched_interval() -> None:
    model, mixer = _fixture()
    original_chunk = mixer.chunk_gated_delta_rule
    chunk_lengths = []

    def recording_chunk(*args, **kwargs):
        chunk_lengths.append(args[0].shape[1])
        return original_chunk(*args, **kwargs)

    mixer.chunk_gated_delta_rule = recording_chunk
    x = torch.randn(1, 20, 16)

    with intervene_gdn(
        model,
        {Address("gdn_state_post", 0): lambda state, _context: state},
        prompt_token_ids=torch.zeros(1, 20, dtype=torch.long),
        positions=[5, 10],
    ):
        mixer(x)

    assert chunk_lengths == [5, 4, 9]


def test_state_capture_requires_explicit_positions() -> None:
    model, mixer = _fixture()
    with (
        pytest.raises(ValueError, match="requires explicit positions"),
        capture_gdn(model, [Address("gdn_state_post", 0)], prompt_len=2, positions=None, detach=True),
    ):
        mixer(torch.randn(1, 2, 16))
