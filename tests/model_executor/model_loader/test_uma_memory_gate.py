# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest

from vllm.model_executor.model_loader import _uma_memory_gate as gate


def test_gate_uma_memory_fails_when_available_is_too_low(monkeypatch):
    monkeypatch.setattr(
        gate,
        "read_meminfo",
        lambda: {
            "MemAvailable": 1,
            "SwapTotal": 0,
            "SwapFree": 0,
        },
    )
    monkeypatch.setattr(gate, "read_memory_psi_avg10", lambda: (0.0, 0.0))

    with pytest.raises(RuntimeError, match="test_loader memory gate failed"):
        gate.gate_uma_memory(
            loader_label="test_loader",
            phase="preflight",
            min_available_gib=1.0,
            psi_gate_seconds=0.0,
            max_swap_gib=0.0,
            logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
        )


def test_gate_uma_memory_fails_when_swap_exceeds_limit(monkeypatch):
    monkeypatch.setattr(
        gate,
        "read_meminfo",
        lambda: {
            "MemAvailable": 2 * 1024**3,
            "SwapTotal": 1024,
            "SwapFree": 0,
        },
    )
    monkeypatch.setattr(gate, "read_memory_psi_avg10", lambda: (0.0, 0.0))

    with pytest.raises(RuntimeError, match="test_loader swap gate failed"):
        gate.gate_uma_memory(
            loader_label="test_loader",
            phase="preflight",
            min_available_gib=1.0,
            psi_gate_seconds=0.0,
            max_swap_gib=0.0,
            logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
        )


def test_gate_uma_memory_fails_when_psi_does_not_clear(monkeypatch):
    monkeypatch.setattr(
        gate,
        "read_meminfo",
        lambda: {
            "MemAvailable": 2 * 1024**3,
            "SwapTotal": 0,
            "SwapFree": 0,
        },
    )
    monkeypatch.setattr(gate, "read_memory_psi_avg10", lambda: (1.0, 0.5))

    with pytest.raises(RuntimeError, match="test_loader PSI gate failed"):
        gate.gate_uma_memory(
            loader_label="test_loader",
            phase="preflight",
            min_available_gib=1.0,
            psi_gate_seconds=0.0,
            max_swap_gib=0.0,
            logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
        )
