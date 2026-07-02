# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the O_DIRECT source read-window handle cache."""

import types

from vllm.model_executor.model_loader import uma_odirect_safetensors_loader as L


class _FakeODirectFile:
    def __init__(self, path, chunk_size, alignment, window_size):
        self.path = path
        self.fd = 1
        self.closed = False
        self.direct_reads = 1
        self.window_loads = 1
        self.window_hits = 0
        self.bytes_read = 100
        self.bytes_copied = 10

    def close(self):
        self.closed = True
        self.fd = -1


def _make_source(monkeypatch):
    monkeypatch.setattr(L, "_ODirectFile", _FakeODirectFile)
    source = object.__new__(L.ODirectSafetensorsWeightSource)
    source._loader = types.SimpleNamespace(
        _chunk_size=4096,
        _alignment=4096,
        _window_size=8192,
    )
    source._stats = L._SourceReadStats()
    source._bytes_since_gate = 0
    source._open_file_handle = None
    return source


def test_open_file_reuses_handle_for_same_path(monkeypatch):
    source = _make_source(monkeypatch)

    first = source._open_file("a.safetensors")
    second = source._open_file("a.safetensors")

    assert first is second
    assert first.closed is False
    assert source._stats.files_opened == 1
    # The handle stays open between reads, so per-file counters are folded
    # into the stats only once, at close time.
    assert source._stats.bytes_read == 0

    source.close_files()
    assert first.closed is True
    assert source._stats.bytes_read == 100


def test_open_file_switches_and_closes_previous_handle(monkeypatch):
    source = _make_source(monkeypatch)

    first = source._open_file("a.safetensors")
    second = source._open_file("b.safetensors")

    assert first.closed is True
    assert second.closed is False
    assert source._stats.files_opened == 2
    assert source._stats.bytes_read == 100  # first handle folded on switch

    source.close_files()
    assert second.closed is True
    assert source._stats.bytes_read == 200


def test_close_files_is_idempotent(monkeypatch):
    source = _make_source(monkeypatch)

    source.close_files()
    handle = source._open_file("a.safetensors")
    source.close_files()
    source.close_files()

    assert handle.closed is True
    assert source._stats.files_opened == 1
    assert source._stats.bytes_read == 100


def test_open_file_reopens_after_close(monkeypatch):
    source = _make_source(monkeypatch)

    first = source._open_file("a.safetensors")
    source.close_files()
    second = source._open_file("a.safetensors")

    assert first is not second
    assert second.closed is False
    assert source._stats.files_opened == 2
