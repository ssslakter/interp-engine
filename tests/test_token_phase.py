"""Prefill/decode steering is selected by absolute position, never batch chunk size."""

from __future__ import annotations

import torch

from interp_engine.address import Address
from interp_engine.vllm_capture._demux import _Demux
from interp_engine.vllm_capture.requests import _process_point
from interp_engine.vllm_capture.static import _apply_one_write, _WriteReq


def _hooked_decode_only(prompt_len: int) -> tuple[_Demux, Address]:
    rid = "request"
    site = Address("resid_post", 0)
    demux = _Demux(None)
    demux.registered.add(rid)
    demux.steer_mods[rid] = {site: (lambda rows: torch.ones_like(rows), (), prompt_len, False, True)}
    return demux, site


def _hooked_step(demux: _Demux, site: Address, rows: int) -> torch.Tensor:
    demux.current_meta = (["request"], [rows])
    return _process_point(demux, site, torch.zeros(rows, 2))


def test_hooked_phase_uses_absolute_cursor_for_chunked_prefill() -> None:
    demux, site = _hooked_decode_only(prompt_len=3)

    torch.testing.assert_close(_hooked_step(demux, site, 2), torch.zeros(2, 2))
    torch.testing.assert_close(_hooked_step(demux, site, 1), torch.zeros(1, 2))
    torch.testing.assert_close(_hooked_step(demux, site, 1), torch.ones(1, 2))


def test_hooked_phase_does_not_treat_a_one_token_prompt_as_decode() -> None:
    demux, site = _hooked_decode_only(prompt_len=1)

    torch.testing.assert_close(_hooked_step(demux, site, 1), torch.zeros(1, 2))
    torch.testing.assert_close(_hooked_step(demux, site, 1), torch.ones(1, 2))


def test_static_phase_uses_the_same_absolute_cursor() -> None:
    request = _WriteReq(vector=torch.ones(1, 2), prompt_len=3, steer_prefill=False)

    chunks = [torch.zeros(rows, 2) for rows in (2, 1, 1)]
    for chunk in chunks:
        _apply_one_write(chunk, None, request, len(chunk), fused=False)

    torch.testing.assert_close(chunks[0], torch.zeros(2, 2))
    torch.testing.assert_close(chunks[1], torch.zeros(1, 2))
    torch.testing.assert_close(chunks[2], torch.ones(1, 2))
