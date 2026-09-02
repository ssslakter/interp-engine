"""Qwen Gated DeltaNet capture and intervention on the recurrence itself.

The tensors exposed here are locals of fused recurrence kernels, not module boundaries. Selected
state/read positions use the same sequential fp32 step as transformers' reference decode path; the
original chunk kernel handles untouched intervals. Outside the context the mixer is untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import MethodType
from typing import Any, cast

import torch
import torch.nn.functional as F

from interp_engine.address import Address, to_address
from interp_engine.dispatch import require_eager
from interp_engine.model import EagerModel
from interp_engine.protocol import InterpModel


class TokenPhase(Enum):
    """Which model-forward rows an intervention applies to."""

    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"


def token_phase(value: TokenPhase | None) -> TokenPhase:
    """Return the default phase or reject a value that would otherwise silently mean BOTH."""
    if value is None:
        return TokenPhase.BOTH
    if not isinstance(value, TokenPhase):
        raise TypeError(f"phase must be a TokenPhase, got {type(value).__name__}")
    return value


GDN_POINTS = frozenset(
    {
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
    }
)
GDN_STATE_POINTS = frozenset({"gdn_state_write", "gdn_state_post"})
GDN_STEP_POINTS = frozenset(
    {
        "gdn_q",
        "gdn_k",
        "gdn_v",
        "gdn_alpha",
        "gdn_beta",
        "gdn_state_write",
        "gdn_state_post",
        "gdn_read",
    }
)
GDN_STATE_OR_READ_POINTS = frozenset({"gdn_state_write", "gdn_state_post", "gdn_read"})


@dataclass(frozen=True)
class GDNInterventionContext:
    """Where one callback invocation sits in the generated sequence."""

    address: Address
    position: int
    phase: TokenPhase


GDNTransform = Callable[[torch.Tensor, GDNInterventionContext], torch.Tensor]


@dataclass(frozen=True)
class GDNInterventionResult:
    """Which explicitly requested absolute positions an intervention actually reached."""

    requested_positions: tuple[int, ...] | None
    _fired: set[int]

    @property
    def fired_positions(self) -> tuple[int, ...]:
        return tuple(sorted(self._fired))

    @property
    def missed_positions(self) -> tuple[int, ...]:
        if self.requested_positions is None:
            return ()
        return tuple(position for position in self.requested_positions if position not in self._fired)


@dataclass
class _Scope:
    transforms: dict[Address, GDNTransform]
    phase: TokenPhase
    positions: frozenset[int] | None
    captures: frozenset[Address]
    capture_positions: tuple[int, ...] | None
    detach: bool
    captured: dict[Address, dict[int, torch.Tensor]]
    fired: set[int]
    prompt_len: int = 0


def _positions(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    out = tuple(int(x) for x in values)
    if any(x < 0 for x in out):
        raise ValueError(f"positions must be non-negative absolute token positions, got {out}")
    if len(set(out)) != len(out):
        raise ValueError(f"positions must not contain duplicates, got {out}")
    return out


def _phase_matches(wanted: TokenPhase, actual: TokenPhase) -> bool:
    return wanted is TokenPhase.BOTH or wanted is actual


def _validate_addresses(addresses: Sequence[Address]) -> None:
    bad_names = sorted(str(address) for address in addresses if address.name not in GDN_POINTS)
    if bad_names:
        raise ValueError(f"GDN instrumentation only accepts GDN points, got {bad_names}")
    missing_layers = sorted(str(address) for address in addresses if address.layer is None)
    if missing_layers:
        raise ValueError(f"GDN points require a layer coordinate, got {missing_layers}")
    streams = sorted(str(address) for address in addresses if address.stream is not None)
    if streams:
        raise ValueError(f"GDN points do not have a residual-stream coordinate, got {streams}")


class _Runtime:
    def __init__(self, mixer: torch.nn.Module, layer: int):
        self.mixer: Any = mixer
        self.layer = layer
        self.scopes: list[_Scope] = []
        self.original_chunk: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = cast(
            Any, mixer.chunk_gated_delta_rule
        )
        self.original_recurrent: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = cast(
            Any, mixer.recurrent_gated_delta_rule
        )
        self.norm: Any = mixer.norm
        self.original_norm_forward = self.norm.forward
        self.next_decode_position = 0
        self.pending: tuple[list[int], int, int, int] | None = None

    def install(self) -> None:
        runtime = self

        def chunk_recurrence(*args: Any, **kwargs: Any):
            return runtime.recurrence(*args, _kernel=runtime.original_chunk, **kwargs)

        def recurrent_recurrence(*args: Any, **kwargs: Any):
            return runtime.recurrence(*args, _kernel=runtime.original_recurrent, **kwargs)

        self.mixer.chunk_gated_delta_rule = chunk_recurrence
        self.mixer.recurrent_gated_delta_rule = recurrent_recurrence

        def norm_forward(_norm: torch.nn.Module, hidden: torch.Tensor, gate: torch.Tensor | None = None):
            if gate is None or runtime.pending is None:
                return (
                    runtime.original_norm_forward(hidden)
                    if gate is None
                    else runtime.original_norm_forward(hidden, gate)
                )
            positions, batch, seq, heads = runtime.pending
            runtime.pending = None
            if not any(runtime._wanted(name) for name in ("gdn_z", "gdn_normed_read", "gdn_post_gate")):
                return runtime.original_norm_forward(hidden, gate)
            original_shape = hidden.shape
            read = hidden.reshape(batch, seq, heads, -1)
            z = gate.reshape(batch, seq, heads, -1)
            eps = float(getattr(_norm, "variance_epsilon", getattr(_norm, "eps", getattr(_norm, "epsilon", 1e-6))))
            normed = read.float() * torch.rsqrt(read.float().pow(2).mean(-1, keepdim=True) + eps)
            weight = cast(torch.Tensor, _norm.weight)
            normed = (normed.to(hidden.dtype) * weight).to(hidden.dtype)
            z = runtime.apply("gdn_z", z, positions)
            normed = runtime.apply("gdn_normed_read", normed, positions)
            post = (normed.float() * F.silu(z.float())).to(hidden.dtype)
            post = runtime.apply("gdn_post_gate", post, positions)
            return post.reshape(original_shape)

        self.norm.forward = MethodType(norm_forward, self.norm)

    def restore(self) -> None:
        self.mixer.chunk_gated_delta_rule = self.original_chunk
        self.mixer.recurrent_gated_delta_rule = self.original_recurrent
        self.norm.forward = self.original_norm_forward
        delattr(self.mixer, "_interp_engine_gdn_runtime")

    def apply(self, name: str, tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
        address = Address(name, self.layer)
        rows: list[torch.Tensor] = []
        for offset, position in enumerate(positions):
            row = tensor[:, offset]
            actual = TokenPhase.PREFILL if position < self.prefill_len else TokenPhase.DECODE
            for scope in self.scopes:
                transform = scope.transforms.get(address)
                if transform is None or not _phase_matches(scope.phase, actual):
                    continue
                if scope.positions is not None and position not in scope.positions:
                    continue
                changed = transform(row, GDNInterventionContext(address, position, actual))
                if not isinstance(changed, torch.Tensor):
                    raise TypeError(f"Intervention at {address} returned {type(changed).__name__}, not a Tensor")
                if changed.shape != row.shape or changed.dtype != row.dtype or changed.device != row.device:
                    raise ValueError(
                        f"Intervention at {address} position {position} must preserve shape, dtype, and device; "
                        f"got {tuple(changed.shape)} {changed.dtype} {changed.device}, expected "
                        f"{tuple(row.shape)} {row.dtype} {row.device}"
                    )
                row = changed
                scope.fired.add(position)
            for scope in self.scopes:
                if address not in scope.captures:
                    continue
                if scope.capture_positions is not None and position not in scope.capture_positions:
                    continue
                scope.captured.setdefault(address, {})[position] = row.detach().clone() if scope.detach else row
            rows.append(row)
        return torch.stack(rows, dim=1)

    def _wanted(self, name: str) -> bool:
        address = Address(name, self.layer)
        return any(address in scope.transforms or address in scope.captures for scope in self.scopes)

    def _maybe_apply(self, name: str, tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
        return self.apply(name, tensor, positions) if self._wanted(name) else tensor

    @property
    def prefill_len(self) -> int:
        lengths = [scope.prompt_len for scope in self.scopes if scope.prompt_len]
        return lengths[0] if lengths else self.next_decode_position

    def _step_positions(self, positions: list[int]) -> set[int]:
        """Positions whose recurrence internals cannot stay inside the chunk kernel."""
        needed: set[int] = set()
        for position in positions:
            actual = TokenPhase.PREFILL if position < self.prefill_len else TokenPhase.DECODE
            for scope in self.scopes:
                if any(
                    address.name in GDN_STEP_POINTS
                    and _phase_matches(scope.phase, actual)
                    and (scope.positions is None or position in scope.positions)
                    for address in scope.transforms
                ):
                    needed.add(position)
                    break
                if any(
                    address.name in GDN_STATE_OR_READ_POINTS
                    and (scope.capture_positions is None or position in scope.capture_positions)
                    for address in scope.captures
                ):
                    needed.add(position)
                    break
        return needed

    def recurrence(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        _kernel: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("GDN capture/intervention does not support packed cu_seqlens yet")
        initial_dtype = query.dtype
        batch, seq, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        if initial_state is None:
            positions = list(range(seq))
            self.next_decode_position = seq
        else:
            start = self.next_decode_position
            positions = list(range(start, start + seq))
            self.next_decode_position += seq

        # Match the reference kernel's order exactly: Q/K are normalized in model dtype and
        # only then promoted for the fp32 recurrence.  Promoting first is mathematically close,
        # but is observably different for a bf16 checkpoint after later layers amplify it.
        if use_qk_l2norm_in_kernel:
            query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
            key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
        q = query.float()
        k = key.float()
        v = value.float()
        alpha = g.float().exp()
        b = beta.float()
        q = self._maybe_apply("gdn_q", q, positions)
        k = self._maybe_apply("gdn_k", k, positions)
        v = self._maybe_apply("gdn_v", v, positions)
        alpha = self._maybe_apply("gdn_alpha", alpha, positions)
        b = self._maybe_apply("gdn_beta", b, positions)

        state: torch.Tensor | None = (
            torch.zeros(batch, heads, key_dim, value_dim, dtype=torch.float32, device=query.device)
            if initial_state is None
            else initial_state.float()
        )
        step_positions = self._step_positions(positions)
        parts: list[torch.Tensor] = []
        cursor = 0

        def run_chunk(end: int) -> None:
            nonlocal cursor, state
            if cursor >= end:
                return
            chunk, state = _kernel(
                query[:, cursor:end],
                key[:, cursor:end],
                value[:, cursor:end],
                g=g[:, cursor:end],
                beta=beta[:, cursor:end],
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            parts.append(chunk)
            cursor = end

        for offset, position in enumerate(positions):
            if position not in step_positions:
                continue
            run_chunk(offset)
            assert state is not None
            state = state * alpha[:, offset, :, None, None]
            prediction = (state * k[:, offset, :, :, None]).sum(-2)
            update_value = (v[:, offset] - prediction) * b[:, offset, :, None]
            write = k[:, offset, :, :, None] * update_value[:, :, None, :]
            write = self._maybe_apply("gdn_state_write", write[:, None], [position])[:, 0]
            state = state + write
            state = self._maybe_apply("gdn_state_post", state[:, None], [position])[:, 0]
            read = (state * q[:, offset, :, :, None]).sum(-2) * (key_dim**-0.5)
            read = self._maybe_apply("gdn_read", read[:, None], [position])[:, 0]
            parts.append(read[:, None].to(initial_dtype))
            cursor = offset + 1
        run_chunk(seq)
        raw = torch.cat(parts, dim=1)
        self.pending = (positions, batch, seq, heads)
        return raw, state if output_final_state else None


def _mixer(model: EagerModel, layer: int) -> Any:
    mixer = model.arch.attn_module(layer)
    name = type(mixer).__name__
    required = ("chunk_gated_delta_rule", "recurrent_gated_delta_rule", "norm")
    if not name.startswith(("Qwen3NextGatedDeltaNet", "Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet")) or not all(
        hasattr(mixer, attr) for attr in required
    ):
        raise ValueError(f"Layer {layer} is {name}, not a supported Qwen3-Next/Qwen3.5 Gated DeltaNet layer")
    return mixer


@contextmanager
def _instrument(model: EagerModel, scope: _Scope, prompt_len: int) -> Iterator[_Scope]:
    scope.prompt_len = prompt_len
    layers = sorted({a.layer for a in (*scope.transforms, *scope.captures) if a.layer is not None})
    runtimes: list[_Runtime] = []
    for layer in layers:
        assert layer is not None
        mixer = _mixer(model, layer)
        runtime = getattr(mixer, "_interp_engine_gdn_runtime", None)
        if runtime is None:
            runtime = _Runtime(mixer, layer)
            mixer._interp_engine_gdn_runtime = runtime
            runtime.install()
        runtime.scopes.append(scope)
        runtimes.append(runtime)
    try:
        yield scope
    finally:
        for runtime in reversed(runtimes):
            runtime.scopes.remove(scope)
            if not runtime.scopes:
                runtime.restore()


@contextmanager
def intervene_gdn(
    model: InterpModel,
    transforms: Mapping[Address | str | tuple[str, int], GDNTransform],
    *,
    prompt_token_ids: Any,
    phase: TokenPhase = TokenPhase.BOTH,
    positions: Sequence[int] | None = None,
) -> Iterator[GDNInterventionResult]:
    """Apply arbitrary tensor transforms at semantic Qwen GDN points.

    A callback receives one absolute position at a time, with the batch axis retained, and must
    return the same shape, dtype, and device. The engine deliberately does not renormalize Q/K or
    clamp gates after the callback: its return value is the exact tensor used by the recurrence.
    ``positions`` selects absolute token positions; it never waits for a relative decode index.
    """
    require_eager(model, "intervene_gdn", capability="gdn_recurrence")
    assert isinstance(model, EagerModel)
    phase = token_phase(phase)
    normalized = {to_address(address): fn for address, fn in transforms.items()}
    _validate_addresses(tuple(normalized))
    selected = _positions(positions)
    prompt_len = int(torch.as_tensor(prompt_token_ids).shape[-1])
    scope = _Scope(
        normalized, phase, None if selected is None else frozenset(selected), frozenset(), None, True, {}, set()
    )
    with _instrument(model, scope, prompt_len):
        yield GDNInterventionResult(selected, scope.fired)


@contextmanager
def capture_gdn(
    model: EagerModel,
    addresses: Sequence[Address],
    *,
    prompt_len: int,
    positions: Sequence[int] | None,
    detach: bool,
) -> Iterator[Callable[[], dict[Address, torch.Tensor]]]:
    """Internal capture context used by :mod:`interp_engine.capture`."""
    wanted = frozenset(addresses)
    _validate_addresses(tuple(wanted))
    selected = _positions(positions)
    if selected is None and any(a.name in GDN_STATE_POINTS for a in wanted):
        raise ValueError("Capturing gdn_state_write or gdn_state_post requires explicit positions=[...]")
    scope = _Scope({}, TokenPhase.BOTH, None, wanted, selected, detach, {}, set())

    def finish() -> dict[Address, torch.Tensor]:
        order_by_address = {
            address: selected if selected is not None else tuple(sorted(scope.captured.get(address, {})))
            for address in wanted
        }
        missing = {
            address: tuple(position for position in order if position not in scope.captured.get(address, {}))
            for address, order in order_by_address.items()
        }
        missing = {address: positions for address, positions in missing.items() if positions}
        if missing:
            detail = ", ".join(
                f"{address}: {list(positions)}"
                for address, positions in sorted(missing.items(), key=lambda item: str(item[0]))
            )
            raise ValueError(f"GDN capture positions were never processed ({detail})")
        return {
            address: torch.stack([scope.captured[address][position] for position in order], dim=1)
            for address, order in order_by_address.items()
        }

    with _instrument(model, scope, prompt_len):
        yield finish
