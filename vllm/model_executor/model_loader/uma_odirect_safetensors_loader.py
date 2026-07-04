# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import hmac
import inspect
import json
import math
import os
import socket
import socketserver
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader._uma_memory_gate import (
    format_gib,
    gate_uma_memory,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_plan import (
    _DTYPE_MAP,
    _DTYPE_NBYTES,
    ExecutorCapability,
    TensorCatalog,
    TensorMeta,
    ReadScheduleSummary,
    TransformOp,
    WeightPlan,
    WeightPlanBuilder,
    WeightPlanEntry,
    WeightPlanExecutor,
    WeightPlanReadSegment,
    WeightPlanSourceModel,
    WeightPlanSummary,
    _normalize_single_dim_slice_selection,
    _normalize_slice_selection,
    _resolve_attr,
    _TensorRecord,
    _weight_plan_entry_target_shape,
    apply_transform_ops,
    build_auto_weight_plan_for_module,
    build_auto_weight_plan_from_catalog,
    register_weight_transform,
    resolve_weight_plan,
    resolve_weight_plan_source_hooks,
    schedule_weight_plan_reads,
    summarize_weight_plan,
    validate_weight_plan_read_segments,
    verify_loaded_weights,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

_REMOTE_MAX_HEADER_BYTES = 4 * 1024 * 1024
_REMOTE_DEFAULT_MAX_BATCH_PAYLOAD_BYTES = 128 * 1024 * 1024
_REMOTE_MAX_BATCH_ITEMS = 16_384


__all__ = [
    "ODirectSafetensorsWeightSource",
    "RemoteODirectSafetensorsWeightSource",
    "RemoteODirectSafetensorsWeightSourceServer",
    "ExecutorCapability",
    "TensorCatalog",
    "TensorMeta",
    "TransformOp",
    "UmaODirectSafetensorsModelLoader",
    "WeightPlan",
    "WeightPlanBuilder",
    "WeightPlanEntry",
    "WeightPlanExecutor",
    "WeightPlanReadSegment",
    "WeightPlanSourceModel",
    "WeightPlanSummary",
    "apply_transform_ops",
    "build_auto_weight_plan_for_module",
    "build_auto_weight_plan_from_catalog",
    "execute_weight_plan",
    "register_weight_transform",
    "resolve_weight_plan",
    "resolve_weight_plan_source_hooks",
    "schedule_weight_plan_reads",
    "summarize_weight_plan",
    "verify_loaded_weights",
]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None
_LIBC.pread.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_longlong,
]
_LIBC.pread.restype = ctypes.c_ssize_t
_LIBC.memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_LIBC.memmove.restype = ctypes.c_void_p


def _format_gib(value: float) -> str:
    return format_gib(value)


def _round_down(value: int, align: int) -> int:
    return value // align * align


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _select_contiguous_target_view(
    dst: torch.Tensor,
    target_slices: tuple[slice | int, ...] | None,
    name: str,
) -> torch.Tensor:
    if target_slices is None:
        return dst
    if len(target_slices) != dst.ndim:
        raise RuntimeError(
            f"read_into_cpu target slice rank mismatch for {name}: "
            f"got {len(target_slices)} indices for dst shape {list(dst.shape)}"
        )
    try:
        target = dst[target_slices]
    except Exception as exc:
        raise RuntimeError(
            f"read_into_cpu invalid target slice for {name}: {target_slices!r}"
        ) from exc
    if not isinstance(target, torch.Tensor):
        raise RuntimeError(f"read_into_cpu target slice for {name} is not a tensor")
    if not target.is_contiguous():
        raise RuntimeError(
            f"read_into_cpu target slice for {name} must be contiguous: "
            f"{target_slices!r}"
        )
    return target


def _segment_source_sort_key(
    record: TensorMeta,
    segment: WeightPlanReadSegment,
) -> tuple[int, int] | None:
    try:
        element_offset, element_count, _output_shape = _normalize_slice_selection(
            record.shape,
            segment.source_slices,
        )
    except ValueError:
        return None
    element_size = _DTYPE_NBYTES[record.dtype]
    return record.offset + element_offset * element_size, element_count * element_size


def _source_tensor_shape(
    record: TensorMeta,
    source_slices: tuple[slice | int, ...] | None,
) -> list[int]:
    if source_slices is None:
        return list(record.shape)
    try:
        _element_offset, _element_count, output_shape = _normalize_slice_selection(
            record.shape,
            source_slices,
        )
        return output_shape
    except ValueError as exc:
        strided = _normalize_single_dim_slice_selection(record.shape, source_slices)
        if strided is None:
            raise exc
        return strided[4]


def _source_tensor_payload_bytes(
    record: TensorMeta,
    source_slices: tuple[slice | int, ...] | None,
) -> int:
    if source_slices is None:
        return record.size
    return math.prod(_source_tensor_shape(record, source_slices)) * _DTYPE_NBYTES[
        record.dtype
    ]


def _encode_slice_selection(
    selection: tuple[slice | int, ...] | None,
) -> tuple[dict[str, int | None | str], ...] | None:
    if selection is None:
        return None
    encoded = []
    for item in selection:
        if isinstance(item, bool):
            raise RuntimeError(f"Boolean indices are not supported: {selection!r}")
        if isinstance(item, int):
            encoded.append({"kind": "int", "value": item})
        elif isinstance(item, slice):
            encoded.append(
                {
                    "kind": "slice",
                    "start": item.start,
                    "stop": item.stop,
                    "step": item.step,
                }
            )
        else:
            raise RuntimeError(f"Unsupported slice item {item!r}")
    return tuple(encoded)


def _decode_slice_selection(
    encoded: list[dict[str, int | None | str]]
    | tuple[dict[str, int | None | str], ...]
    | None,
) -> tuple[slice | int, ...] | None:
    if encoded is None:
        return None
    decoded: list[slice | int] = []
    for item in encoded:
        kind = item.get("kind")
        if kind == "int":
            value = item.get("value")
            if not isinstance(value, int):
                raise RuntimeError(f"Invalid encoded integer slice item: {item!r}")
            decoded.append(value)
        elif kind == "slice":
            decoded.append(slice(item.get("start"), item.get("stop"), item.get("step")))
        else:
            raise RuntimeError(f"Invalid encoded slice item: {item!r}")
    return tuple(decoded)


def _encode_segments(
    segments: tuple[WeightPlanReadSegment, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "source_slices": _encode_slice_selection(segment.source_slices),
            "target_slices": _encode_slice_selection(segment.target_slices),
        }
        for segment in segments
    )


def _decode_segments(
    encoded: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> tuple[WeightPlanReadSegment, ...]:
    segments: list[WeightPlanReadSegment] = []
    for item in encoded:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid encoded read segment: {item!r}")
        source_slices = _decode_slice_selection(  # type: ignore[arg-type]
            item.get("source_slices")
        )
        target_slices = _decode_slice_selection(  # type: ignore[arg-type]
            item.get("target_slices")
        )
        if source_slices is None or target_slices is None:
            raise RuntimeError(f"Invalid encoded read segment: {item!r}")
        segments.append(
            WeightPlanReadSegment(
                source_slices=source_slices,
                target_slices=target_slices,
            )
        )
    return tuple(segments)


_WIRE_DTYPE_BY_TORCH: dict[torch.dtype, str] = {
    dtype: name for name, dtype in _DTYPE_MAP.items()
}


def _dtype_to_wire(dtype: torch.dtype) -> str:
    try:
        return _WIRE_DTYPE_BY_TORCH[dtype]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported tensor dtype for remote transfer: {dtype}"
        ) from exc


def _dtype_from_wire(name: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[name]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported remote tensor dtype {name!r}") from exc


def _tensor_to_wire(tensor: torch.Tensor) -> dict[str, object]:
    if tensor.device.type != "cpu":
        raise RuntimeError(
            f"Remote weight transfer requires CPU tensor, got {tensor.device}"
        )
    contiguous = tensor.contiguous()
    return {
        "dtype": _dtype_to_wire(contiguous.dtype),
        "shape": tuple(int(dim) for dim in contiguous.shape),
        "payload": contiguous.view(torch.uint8).numpy().tobytes(),
    }


@dataclass(frozen=True)
class _RemoteReadRequest:
    checkpoint_name: str
    source_slices: tuple[slice | int, ...] | None = None
    read_segments: tuple[WeightPlanReadSegment, ...] | None = None
    staging_shape: tuple[int, ...] | None = None


def _tensor_from_wire(encoded: dict[str, object]) -> torch.Tensor:
    dtype_name = encoded.get("dtype")
    shape = encoded.get("shape")
    payload = encoded.get("payload")
    if not isinstance(dtype_name, str):
        raise RuntimeError(f"Invalid remote tensor dtype: {encoded!r}")
    if not isinstance(shape, (list, tuple)) or not all(
        isinstance(dim, int) for dim in shape
    ):
        raise RuntimeError(f"Invalid remote tensor shape: {encoded!r}")
    shape = tuple(shape)
    if not isinstance(payload, bytes):
        raise RuntimeError("Invalid remote tensor payload")
    dtype = _dtype_from_wire(dtype_name)
    expected = math.prod(shape) * _DTYPE_NBYTES[dtype]
    if len(payload) != expected:
        raise RuntimeError(
            "Remote tensor payload size mismatch: "
            f"got={len(payload)}, expected={expected}, shape={shape}, dtype={dtype}"
        )
    tensor = torch.empty(shape, dtype=dtype, device="cpu")
    if payload:
        source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        tensor.view(torch.uint8).reshape(-1).copy_(source)
    return tensor


def _split_tensor_payload(message: object) -> tuple[object, bytes]:
    if not isinstance(message, dict):
        return message, b""
    tensors = message.get("tensors")
    if isinstance(tensors, (list, tuple)):
        header = dict(message)
        tensor_headers: list[dict[str, object]] = []
        payload_parts: list[bytes] = []
        payload_offset = 0
        for tensor in tensors:
            if not isinstance(tensor, dict):
                return message, b""
            payload = tensor.get("payload")
            if not isinstance(payload, bytes):
                return message, b""
            tensor_header = dict(tensor)
            tensor_header.pop("payload")
            tensor_header["payload_offset"] = payload_offset
            tensor_header["payload_size"] = len(payload)
            tensor_headers.append(tensor_header)
            payload_parts.append(payload)
            payload_offset += len(payload)
        header["tensors"] = tensor_headers
        return header, b"".join(payload_parts)
    tensor = message.get("tensor")
    if not isinstance(tensor, dict):
        return message, b""
    payload = tensor.get("payload")
    if not isinstance(payload, bytes):
        return message, b""
    header = dict(message)
    tensor_header = dict(tensor)
    tensor_header.pop("payload")
    tensor_header["payload_size"] = len(payload)
    header["tensor"] = tensor_header
    return header, payload


def _merge_tensor_payload(message: object, payload: bytes) -> object:
    if not isinstance(message, dict):
        return message
    tensors = message.get("tensors")
    if isinstance(tensors, list):
        message = dict(message)
        decoded: list[dict[str, object]] = []
        for tensor in tensors:
            if not isinstance(tensor, dict):
                raise RuntimeError("Remote weight source frame carried invalid tensors")
            offset = tensor.get("payload_offset")
            size = tensor.get("payload_size")
            if not isinstance(offset, int) or not isinstance(size, int):
                raise RuntimeError("Remote weight source tensor payload metadata invalid")
            if offset < 0 or size < 0 or offset + size > len(payload):
                raise RuntimeError(
                    "Remote weight source tensor payload range is invalid"
                )
            tensor = dict(tensor)
            tensor.pop("payload_offset", None)
            tensor.pop("payload_size", None)
            tensor["payload"] = payload[offset : offset + size]
            decoded.append(tensor)
        message["tensors"] = decoded
        return message
    if not payload:
        return message
    tensor = message.get("tensor")
    if not isinstance(tensor, dict):
        raise RuntimeError("Remote weight source frame carried unexpected payload")
    expected = tensor.get("payload_size")
    if not isinstance(expected, int) or expected != len(payload):
        raise RuntimeError(
            "Remote weight source tensor payload size mismatch: "
            f"header={expected!r}, actual={len(payload)}"
        )
    tensor = dict(tensor)
    tensor.pop("payload_size", None)
    tensor["payload"] = payload
    message = dict(message)
    message["tensor"] = tensor
    return message


def _send_frame(sock: socket.socket, message: object) -> None:
    header, payload = _split_tensor_payload(message)
    header_bytes = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sock.sendall(
        len(header_bytes).to_bytes(8, "big")
        + header_bytes
        + len(payload).to_bytes(8, "big")
        + payload
    )


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise RuntimeError("Remote weight source connection closed mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_frame_or_none(
    sock: socket.socket,
    *,
    max_header_bytes: int = _REMOTE_MAX_HEADER_BYTES,
    max_payload_bytes: int = 0,
) -> object | None:
    raw_size = sock.recv(8)
    if not raw_size:
        return None
    while len(raw_size) < 8:
        chunk = sock.recv(8 - len(raw_size))
        if not chunk:
            raise RuntimeError("Remote weight source connection closed mid-frame")
        raw_size += chunk
    size = int.from_bytes(raw_size, "big")
    if size <= 0:
        raise RuntimeError(f"Invalid remote weight source frame size: {size}")
    if size > max_header_bytes:
        raise RuntimeError(
            "Remote weight source frame header is too large: "
            f"{size} bytes > {max_header_bytes} bytes"
        )
    header = json.loads(_recv_exact(sock, size))
    raw_payload_size = _recv_exact(sock, 8)
    payload_size = int.from_bytes(raw_payload_size, "big")
    if payload_size < 0:
        raise RuntimeError(
            f"Invalid remote weight source payload size: {payload_size}"
        )
    if payload_size > max_payload_bytes:
        raise RuntimeError(
            "Remote weight source frame payload is too large: "
            f"{payload_size} bytes > {max_payload_bytes} bytes"
        )
    payload = _recv_exact(sock, payload_size) if payload_size else b""
    return _merge_tensor_payload(header, payload)


def _recv_frame(
    sock: socket.socket,
    *,
    max_header_bytes: int = _REMOTE_MAX_HEADER_BYTES,
    max_payload_bytes: int = 0,
) -> object:
    frame = _recv_frame_or_none(
        sock,
        max_header_bytes=max_header_bytes,
        max_payload_bytes=max_payload_bytes,
    )
    if frame is None:
        raise RuntimeError("Remote weight source connection closed before frame")
    return frame


def _call_weight_loader(
    weight_loader: Callable,
    param: object,
    tensor: torch.Tensor,
    *,
    source_is_sharded: bool,
    kwargs: dict[str, object],
    entry_name: str,
) -> None:
    had_attr = hasattr(param, "is_sharded_weight")
    old_value = getattr(param, "is_sharded_weight", None)
    if source_is_sharded:
        setattr(param, "is_sharded_weight", True)
    try:
        call_kwargs = dict(kwargs)
        extra_args: list[object] = []
        expects_success = False
        if "shard_id" in call_kwargs or "expert_id" in call_kwargs:
            signature = inspect.signature(weight_loader)
            params = signature.parameters
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            if (
                "shard_id" in call_kwargs
                and "shard_id" not in params
                and not accepts_kwargs
            ):
                # Several vLLM weight_loader_v2 implementations name this
                # argument loaded_shard_id and expect it positionally.  Keep the
                # plan IR field generic while preserving the layer call
                # convention.
                if "loaded_shard_id" in params:
                    extra_args.append(call_kwargs.pop("shard_id"))
            if (
                ("expert_id" in call_kwargs or "shard_id" in call_kwargs)
                and "return_success" in params
            ):
                # Some loaders report refusal through return_success.  The plan
                # already decided this entry is required, so a refusal means the
                # plan and the loader disagree and the load must fail closed.
                call_kwargs["return_success"] = True
                expects_success = True
        result = weight_loader(param, tensor, *extra_args, **call_kwargs)
        if expects_success and not result:
            raise RuntimeError(
                "weight_loader refused a tensor the weight plan marked "
                f"local and required: {entry_name}"
            )
    finally:
        if source_is_sharded:
            if had_attr:
                setattr(param, "is_sharded_weight", old_value)
            else:
                try:
                    delattr(param, "is_sharded_weight")
                except AttributeError:
                    pass


def execute_weight_plan(
    model: nn.Module,
    source: "ODirectSafetensorsWeightSource",
    plan: WeightPlan,
) -> set[str]:
    """Execute a simple model-side WeightPlan with UMA-safe source reads."""

    plan = resolve_weight_plan(model, source.catalog, plan)
    summary = summarize_weight_plan(source.catalog, plan)
    loader = getattr(source, "_loader", None)
    schedule = schedule_weight_plan_reads(
        source.catalog,
        plan,
        chunk_size=getattr(
            source,
            "_chunk_size",
            getattr(
                loader,
                "_chunk_size",
                UmaODirectSafetensorsModelLoader.DEFAULT_CHUNK_SIZE,
            ),
        ),
        window_size=getattr(
            source,
            "_window_size",
            getattr(
                loader,
                "_window_size",
                UmaODirectSafetensorsModelLoader.DEFAULT_WINDOW_SIZE,
            ),
        ),
        alignment=getattr(
            source,
            "_alignment",
            getattr(
                loader,
                "_alignment",
                UmaODirectSafetensorsModelLoader.DEFAULT_ALIGNMENT,
            ),
        ),
    )
    schedule_summary = schedule.summary
    set_expected_read_summary = getattr(source, "set_expected_read_summary", None)
    if callable(set_expected_read_summary):
        set_expected_read_summary(schedule_summary)
    logger.info(
        "uma_odirect_safetensors weight plan: entries=%d required=%d skipped=%d "
        "missing_skipped=%d full_reads=%d sliced_reads=%d read_into=%d "
        "full_payload=%s sliced_payload=%s read_into_payload=%s "
        "skipped_payload=%s total_read_payload=%s",
        summary.entries,
        summary.required_entries,
        summary.skipped_entries,
        summary.missing_skipped_entries,
        summary.full_read_entries,
        summary.sliced_read_entries,
        summary.read_into_entries,
        _format_gib(summary.full_payload_bytes),
        _format_gib(summary.sliced_payload_bytes),
        _format_gib(summary.read_into_payload_bytes),
        _format_gib(summary.skipped_payload_bytes),
        _format_gib(summary.total_read_payload_bytes),
    )
    logger.info(
        "uma_odirect_safetensors read schedule: entries=%d required=%d "
        "read_ranges=%d expected_direct_reads=%d expected_window_loads=%d "
        "expected_window_hits=%d expected_bytes_read=%s payload=%s "
        "expected_read_amplification=%.2fx",
        schedule_summary.entries,
        schedule_summary.required_entries,
        schedule_summary.read_ranges,
        schedule_summary.expected_direct_reads,
        schedule_summary.expected_window_loads,
        schedule_summary.expected_window_hits,
        _format_gib(schedule_summary.expected_bytes_read),
        _format_gib(schedule_summary.payload_bytes),
        schedule_summary.read_amplification,
    )

    loaded: set[str] = set()
    scheduled_entries = tuple(schedule.plan)

    def prepare_entry(entry: WeightPlanEntry):
        if not entry.required:
            source.skip(
                entry.checkpoint_name,
                entry.skip_reason or "weight plan marked not required",
            )
            return None

        try:
            param = _resolve_attr(model, entry.target_name)
        except RuntimeError:
            if entry.ignore_missing:
                source.skip(entry.checkpoint_name, "weight plan target is ignored")
                return None
            raise

        weight_loader = getattr(param, "weight_loader", None)
        if not callable(weight_loader):
            if entry.loader_target_name is not None:
                loader_target = _resolve_attr(model, entry.loader_target_name)
                parent_loader = getattr(loader_target, "weight_loader", None)
                if callable(parent_loader):
                    def _parent_weight_loader(
                        param_arg: object,
                        tensor_arg: torch.Tensor,
                        *,
                        return_success: bool | None = None,
                        **kwargs: object,
                    ) -> object:
                        if return_success is not None:
                            kwargs["return_success"] = return_success
                        return parent_loader(
                            param=param_arg,
                            loaded_weight=tensor_arg,
                            **kwargs,
                        )

                    weight_loader = _parent_weight_loader
                else:
                    raise RuntimeError(
                        f"Weight plan loader target {entry.loader_target_name!r} "
                        "has no custom weight_loader"
                    )
            elif (
                entry.shard_id is not None
                or entry.expert_id is not None
                or entry.weight_name is not None
            ):
                raise RuntimeError(
                    f"Weight plan target {entry.target_name!r} requires loader "
                    "metadata but has no custom weight_loader"
                )
            else:
                weight_loader = default_weight_loader

        record = source.catalog.get(entry.checkpoint_name)

        if entry.target_slices is not None and not entry.read_into_cpu:
            raise RuntimeError(
                "WeightPlanEntry.target_slices is only supported with "
                f"read_into_cpu=True: {entry.checkpoint_name}"
            )
        if entry.read_segments is not None:
            validate_weight_plan_read_segments(record, entry)

        return entry, param, weight_loader, record

    def read_entry_tensor(prepared) -> torch.Tensor:
        entry, _param, _weight_loader, record = prepared
        # The plan was resolved above; executing without further semantic
        # inference keeps the logged summary equal to the actual reads.
        source_slices = entry.source_slices
        if entry.read_segments is not None:
            empty_cpu_shape = getattr(source, "empty_cpu_shape", None)
            if not callable(empty_cpu_shape):
                raise RuntimeError(
                    "WeightSource does not support segmented CPU staging for "
                    f"{entry.checkpoint_name}"
                )
            tensor = empty_cpu_shape(entry.checkpoint_name, entry.staging_shape)
            read_segments_into_cpu = getattr(source, "read_segments_into_cpu", None)
            if callable(read_segments_into_cpu):
                read_segments_into_cpu(
                    entry.checkpoint_name,
                    tensor,
                    entry.read_segments,
                )
            else:
                for segment in entry.read_segments:
                    source_shape = _weight_plan_entry_target_shape(
                        record,
                        segment.source_slices,
                    )
                    target = _select_contiguous_target_view(
                        tensor,
                        segment.target_slices,
                        entry.checkpoint_name,
                    )
                    if tuple(target.shape) != source_shape:
                        raise RuntimeError(
                            "WeightPlanEntry.read_segments target shape mismatch for "
                            f"{entry.checkpoint_name}: target={list(target.shape)}, "
                            f"source_slice={list(source_shape)}"
                        )
                    source.read_into_cpu(
                        entry.checkpoint_name,
                        tensor,
                        source_slices=segment.source_slices,
                        target_slices=segment.target_slices,
                    )
        elif entry.read_into_cpu:
            tensor = source.empty_cpu(
                entry.checkpoint_name,
                source_slices=None
                if entry.target_slices is not None
                else source_slices,
            )
            source.read_into_cpu(
                entry.checkpoint_name,
                tensor,
                source_slices=source_slices,
                target_slices=entry.target_slices,
            )
        elif source_slices is None:
            tensor = source.read_full_cpu(entry.checkpoint_name)
        else:
            tensor = source.read_slice_cpu(entry.checkpoint_name, source_slices)
        return tensor

    def load_entry_tensor(prepared, tensor: torch.Tensor) -> None:
        entry, param, weight_loader, _record = prepared
        if entry.transform_ops:
            tensor = apply_transform_ops(entry.transform_ops, tensor)

        kwargs = {}
        if entry.shard_id is not None:
            kwargs["shard_id"] = entry.shard_id
        if entry.expert_id is not None:
            kwargs["expert_id"] = entry.expert_id
        if entry.weight_name is not None:
            kwargs["weight_name"] = entry.weight_name
        _call_weight_loader(
            weight_loader,
            param,
            tensor,
            source_is_sharded=entry.source_is_sharded,
            kwargs=kwargs,
            entry_name=entry.checkpoint_name,
        )
        loaded.add(entry.target_name)

    read_many_cpu = getattr(source, "read_many_cpu", None)
    read_many_max_payload_bytes = getattr(source, "read_many_max_payload_bytes", None)
    if callable(read_many_cpu) and callable(read_many_max_payload_bytes):
        batch_payload_cap = int(read_many_max_payload_bytes())
    else:
        batch_payload_cap = 0
    read_many_max_items = getattr(source, "read_many_max_items", None)
    if callable(read_many_cpu) and callable(read_many_max_items):
        batch_item_cap = int(read_many_max_items())
    else:
        batch_item_cap = _REMOTE_MAX_BATCH_ITEMS

    def batchable_entry_payload(prepared) -> int | None:
        entry, _param, _weight_loader, record = prepared
        if entry.target_slices is not None:
            return None
        if entry.read_segments is not None:
            if not entry.read_into_cpu or entry.staging_shape is None:
                return None
            return math.prod(entry.staging_shape) * _DTYPE_NBYTES[record.dtype]
        if entry.read_into_cpu:
            return None
        return _source_tensor_payload_bytes(record, entry.source_slices)

    index = 0
    while index < len(scheduled_entries):
        entry = scheduled_entries[index]
        prepared = prepare_entry(entry)
        if prepared is None:
            index += 1
            continue

        read_segment_group_into_cpu = getattr(source, "read_segment_group_into_cpu", None)
        if entry.read_segments is not None and callable(read_segment_group_into_cpu):
            group = [prepared]
            next_index = index + 1
            while next_index < len(scheduled_entries):
                next_entry = scheduled_entries[next_index]
                if (
                    not next_entry.required
                    or next_entry.read_segments is None
                    or next_entry.checkpoint_name != entry.checkpoint_name
                ):
                    break
                next_prepared = prepare_entry(next_entry)
                if next_prepared is not None:
                    group.append(next_prepared)
                next_index += 1

            if len(group) > 1:
                requests = []
                tensors = []
                empty_cpu_shape = getattr(source, "empty_cpu_shape", None)
                if not callable(empty_cpu_shape):
                    raise RuntimeError(
                        "WeightSource does not support segmented CPU staging for "
                        f"{entry.checkpoint_name}"
                    )
                for group_prepared in group:
                    group_entry = group_prepared[0]
                    tensor = empty_cpu_shape(
                        group_entry.checkpoint_name,
                        group_entry.staging_shape,
                    )
                    tensors.append(tensor)
                    requests.append((tensor, group_entry.read_segments))
                read_segment_group_into_cpu(entry.checkpoint_name, tuple(requests))
                for group_prepared, tensor in zip(group, tensors):
                    load_entry_tensor(group_prepared, tensor)
                index = next_index
                continue

        if callable(read_many_cpu) and batch_payload_cap > 0:
            first_payload = batchable_entry_payload(prepared)
            if first_payload is not None and first_payload <= batch_payload_cap:
                group = [prepared]
                total_payload = first_payload
                next_index = index + 1
                while next_index < len(scheduled_entries):
                    if len(group) >= batch_item_cap:
                        break
                    next_entry = scheduled_entries[next_index]
                    if (
                        not next_entry.required
                        or next_entry.ignore_missing
                        or next_entry.target_slices is not None
                    ):
                        break
                    next_prepared = prepare_entry(next_entry)
                    next_payload = batchable_entry_payload(next_prepared)
                    if next_payload is None or next_payload > batch_payload_cap:
                        break
                    if total_payload + next_payload > batch_payload_cap:
                        break
                    group.append(next_prepared)
                    total_payload += next_payload
                    next_index += 1

                if len(group) > 1:
                    requests = tuple(
                        _RemoteReadRequest(
                            group_entry.checkpoint_name,
                            group_entry.source_slices,
                            group_entry.read_segments,
                            group_entry.staging_shape,
                        )
                        for group_entry, _param, _weight_loader, _record in group
                    )
                    tensors = read_many_cpu(requests)
                    if len(tensors) != len(group):
                        raise RuntimeError(
                            "WeightSource.read_many_cpu returned an unexpected "
                            f"tensor count: got={len(tensors)}, expected={len(group)}"
                        )
                    for group_prepared, tensor in zip(group, tensors):
                        load_entry_tensor(group_prepared, tensor)
                    index = next_index
                    continue

        tensor = read_entry_tensor(prepared)
        load_entry_tensor(prepared, tensor)
        index += 1
    return loaded


@dataclass
class _SourceReadStats:
    files_opened: int = 0
    tensors_read: int = 0
    tensors_read_full: int = 0
    tensors_read_sliced: int = 0
    tensors_skipped: int = 0
    direct_reads: int = 0
    window_loads: int = 0
    window_hits: int = 0
    bytes_read: int = 0
    bytes_copied: int = 0
    bytes_tensor_payload: int = 0
    bytes_full_tensor_payload: int = 0
    bytes_sliced_tensor_payload: int = 0
    bytes_skipped_payload: int = 0
    time_gate: float = 0.0
    time_alloc: float = 0.0
    time_read: float = 0.0
    time_consumer: float = 0.0

    def collect_file(self, file: "_ODirectFile") -> None:
        self.direct_reads += getattr(file, "direct_reads", 0)
        self.window_loads += getattr(file, "window_loads", 0)
        self.window_hits += getattr(file, "window_hits", 0)
        self.bytes_read += getattr(file, "bytes_read", 0)
        self.bytes_copied += getattr(file, "bytes_copied", 0)

    def log(self, label: str) -> None:
        logger.info(
            "uma_odirect_safetensors source stats (%s): files_opened=%d "
            "tensors_read=%d tensors_read_full=%d tensors_read_sliced=%d "
            "tensors_skipped=%d direct_reads=%d "
            "window_loads=%d window_hits=%d bytes_read=%s bytes_copied=%s "
            "tensor_payload=%s full_payload=%s sliced_payload=%s "
            "skipped_payload=%s",
            label,
            self.files_opened,
            self.tensors_read,
            self.tensors_read_full,
            self.tensors_read_sliced,
            self.tensors_skipped,
            self.direct_reads,
            self.window_loads,
            self.window_hits,
            _format_gib(self.bytes_read),
            _format_gib(self.bytes_copied),
            _format_gib(self.bytes_tensor_payload),
            _format_gib(self.bytes_full_tensor_payload),
            _format_gib(self.bytes_sliced_tensor_payload),
            _format_gib(self.bytes_skipped_payload),
        )
        logger.info(
            "uma_odirect_safetensors source timings (%s): gate=%.3fs "
            "alloc=%.3fs read_copy=%.3fs consumer=%.3fs",
            label,
            self.time_gate,
            self.time_alloc,
            self.time_read,
            self.time_consumer,
        )

    def snapshot(self) -> dict[str, int | float]:
        return {
            "files_opened": self.files_opened,
            "tensors_read": self.tensors_read,
            "tensors_read_full": self.tensors_read_full,
            "tensors_read_sliced": self.tensors_read_sliced,
            "tensors_skipped": self.tensors_skipped,
            "direct_reads": self.direct_reads,
            "window_loads": self.window_loads,
            "window_hits": self.window_hits,
            "bytes_read": self.bytes_read,
            "bytes_copied": self.bytes_copied,
            "bytes_tensor_payload": self.bytes_tensor_payload,
            "bytes_full_tensor_payload": self.bytes_full_tensor_payload,
            "bytes_sliced_tensor_payload": self.bytes_sliced_tensor_payload,
            "bytes_skipped_payload": self.bytes_skipped_payload,
            "time_gate": self.time_gate,
            "time_alloc": self.time_alloc,
            "time_read": self.time_read,
            "time_consumer": self.time_consumer,
        }


class ODirectSafetensorsWeightSource:
    """Pull-oriented source for UMA-safe safetensors loading.

    Phase 1 keeps the existing full-tensor iterator behavior, but exposes a
    catalog-first object so model-side plans can inspect metadata before any
    payload bytes are read.
    """

    def __init__(
        self,
        loader: "UmaODirectSafetensorsModelLoader",
        model_or_path: str,
    ) -> None:
        self._loader = loader
        loader._gate_memory("preflight")
        self.files = loader._prepare_files(model_or_path)
        self.catalog = loader._build_catalog(self.files)
        self._stats = _SourceReadStats()
        self._bytes_since_gate = 0
        self._open_file_handle: "_ODirectFile | None" = None
        self._open_file_path: str | None = None
        self._expected_read_summary: ReadScheduleSummary | None = None

    def _open_file(self, path: str) -> "_ODirectFile":
        """Return an open O_DIRECT handle, keeping the most recent file open.

        The read window lives on the handle, so keeping one handle open across
        plan entries lets adjacent small tensors share window loads instead of
        paying one window-sized read per tensor.  Only the most recent file
        stays open, so fd count and window-buffer residency stay bounded at one.
        """

        handle = self._open_file_handle
        if handle is not None and self._open_file_path == path:
            return handle
        self.close_files()
        handle = _ODirectFile(
            path,
            self._loader._chunk_size,
            self._loader._alignment,
            self._loader._window_size,
        )
        self._open_file_handle = handle
        self._open_file_path = path
        self._stats.files_opened += 1
        return handle

    def close_files(self) -> None:
        """Close the cached O_DIRECT handle and fold its counters into stats."""

        handle = self._open_file_handle
        if handle is None:
            return
        self._open_file_handle = None
        self._open_file_path = None
        self._stats.collect_file(handle)
        close = getattr(handle, "close", None)
        if close is not None:
            close()

    def iter_full_tensors(
        self,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        records = self.catalog.records()
        logger.info(
            "uma_odirect_safetensors using O_DIRECT source iterator: files=%d "
            "tensors=%d total=%s chunk_size=%d window_size=%d alignment=%d",
            len(self.files),
            len(records),
            _format_gib(self.catalog.total_bytes()),
            self._loader._chunk_size,
            self._loader._window_size,
            self._loader._alignment,
        )
        consumer_profile = (
            _ConsumerProfile()
            if os.environ.get("VLLM_UMA_LOAD_PROFILE", "").lower()
            in ("1", "true", "yes", "on")
            else None
        )

        try:
            for record in records:
                odirect_file = self._open_file(record.file_path)
                tensor, time_alloc, time_read = self._read_record_tensor(
                    record,
                    odirect_file,
                )
                self._stats.tensors_read += 1
                self._stats.tensors_read_full += 1
                self._stats.bytes_tensor_payload += record.size
                self._stats.bytes_full_tensor_payload += record.size
                self._stats.time_alloc += time_alloc
                self._stats.time_read += time_read

                t0 = time.perf_counter()
                yield record.name, tensor
                elapsed_consumer = time.perf_counter() - t0
                self._stats.time_consumer += elapsed_consumer
                if consumer_profile is not None:
                    consumer_profile.record(record, elapsed_consumer)
        finally:
            if sys.exc_info()[0] is None:
                self._maybe_gate("final", force=True)
            else:
                try:
                    self._maybe_gate("final", force=True)
                except Exception:
                    logger.warning(
                        "uma_odirect_safetensors final source gate failed while "
                        "another load error was already being raised",
                        exc_info=True,
                    )
            self.close_files()
            if consumer_profile is not None:
                consumer_profile.log()

    def read_full_cpu(self, name: str) -> torch.Tensor:
        record = self.catalog.get(name)
        return self._read_record_cpu(record, sliced=False)

    def read_slice_cpu(
        self,
        name: str,
        source_slices: tuple[slice | int, ...],
    ) -> torch.Tensor:
        record = self.catalog.get(name)
        try:
            element_offset, element_count, output_shape = _normalize_slice_selection(
                record.shape,
                source_slices,
            )
        except ValueError as exc:
            strided = _normalize_single_dim_slice_selection(
                record.shape,
                source_slices,
            )
            if strided is None:
                raise exc
            return self._read_strided_slice_cpu(record, strided)

        element_size = _DTYPE_NBYTES[record.dtype]
        slice_record = TensorMeta(
            file_path=record.file_path,
            name=f"{record.name}[slice]",
            dtype=record.dtype,
            shape=output_shape,
            offset=record.offset + element_offset * element_size,
            size=element_count * element_size,
        )
        return self._read_record_cpu(slice_record, sliced=True)

    def empty_cpu(
        self,
        name: str,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
    ) -> torch.Tensor:
        record = self.catalog.get(name)
        shape = _source_tensor_shape(record, source_slices)
        return self.empty_cpu_shape(name, tuple(shape))

    def empty_cpu_shape(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        record = self.catalog.get(name)
        if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise RuntimeError(f"Invalid CPU staging shape for {name}: {shape!r}")
        nbytes = math.prod(shape) * _DTYPE_NBYTES[record.dtype]
        force_allocation_gate = nbytes >= self._loader._allocation_gate_min_bytes
        if force_allocation_gate:
            self._maybe_gate(f"before allocating {name}", force=True)
        t0 = time.perf_counter()
        tensor = torch.empty(list(shape), dtype=record.dtype, device="cpu")
        self._stats.time_alloc += time.perf_counter() - t0
        if force_allocation_gate:
            self._maybe_gate(f"after allocating {name}", force=True)
        return tensor

    def read_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
        target_slices: tuple[slice | int, ...] | None = None,
        force_gate: bool = True,
    ) -> None:
        record = self.catalog.get(name)
        if dst.device.type != "cpu":
            raise RuntimeError(
                f"read_into_cpu requires a CPU destination for {name}, "
                f"got {dst.device}"
            )
        if dst.dtype != record.dtype:
            raise RuntimeError(
                f"read_into_cpu dtype mismatch for {name}: "
                f"dst={dst.dtype}, source={record.dtype}"
            )
        if not dst.is_contiguous():
            raise RuntimeError(f"read_into_cpu requires a contiguous dst for {name}")

        target = _select_contiguous_target_view(dst, target_slices, name)

        if source_slices is None:
            if tuple(target.shape) != record.shape:
                raise RuntimeError(
                    f"read_into_cpu shape mismatch for {name}: "
                    f"dst={list(target.shape)}, source={record.shape}"
                )
            self._read_record_into_cpu(
                record,
                target,
                sliced=False,
                force_gate=force_gate,
            )
            return

        try:
            element_offset, element_count, output_shape = _normalize_slice_selection(
                record.shape,
                source_slices,
            )
        except ValueError as exc:
            strided = _normalize_single_dim_slice_selection(
                record.shape,
                source_slices,
            )
            if strided is None:
                raise exc
            if list(target.shape) != strided[4]:
                raise RuntimeError(
                    f"read_into_cpu shape mismatch for {name}: "
                    f"dst={list(target.shape)}, source_slice={strided[4]}"
                )
            self._read_strided_slice_into_cpu(
                record,
                strided,
                target,
                force_gate=force_gate,
            )
            return

        if list(target.shape) != output_shape:
            raise RuntimeError(
                f"read_into_cpu shape mismatch for {name}: "
                f"dst={list(target.shape)}, source_slice={output_shape}"
            )
        element_size = _DTYPE_NBYTES[record.dtype]
        slice_record = TensorMeta(
            file_path=record.file_path,
            name=f"{record.name}[slice]",
            dtype=record.dtype,
            shape=output_shape,
            offset=record.offset + element_offset * element_size,
            size=element_count * element_size,
        )
        self._read_record_into_cpu(
            slice_record,
            target,
            sliced=True,
            force_gate=force_gate,
        )

    def read_segments_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        segments: tuple[WeightPlanReadSegment, ...],
    ) -> None:
        self.read_segment_group_into_cpu(name, ((dst, segments),))

    def read_segment_group_into_cpu(
        self,
        name: str,
        requests: tuple[
            tuple[torch.Tensor, tuple[WeightPlanReadSegment, ...]],
            ...,
        ],
    ) -> None:
        self._maybe_gate(f"before reading {name}[segments]", force=True)
        record = self.catalog.get(name)
        sortable: list[
            tuple[tuple[int, int], int, torch.Tensor, WeightPlanReadSegment]
        ] = []
        unsorted: list[tuple[int, torch.Tensor, WeightPlanReadSegment]] = []
        order = 0
        for dst, segments in requests:
            for segment in segments:
                sort_key = _segment_source_sort_key(record, segment)
                if sort_key is None:
                    unsorted.append((order, dst, segment))
                else:
                    sortable.append((sort_key, order, dst, segment))
                order += 1
        read_items = [
            (dst, segment)
            for _sort_key, _order, dst, segment in sorted(
                sortable,
                key=lambda item: (item[0], item[1]),
            )
        ]
        read_items.extend(
            (dst, segment)
            for _order, dst, segment in sorted(unsorted, key=lambda item: item[0])
        )
        for dst, segment in read_items:
            self.read_into_cpu(
                name,
                dst,
                source_slices=segment.source_slices,
                target_slices=segment.target_slices,
                force_gate=False,
            )
        self._maybe_gate(f"after reading {name}[segments]", force=True)

    def _read_strided_slice_cpu(
        self,
        record: TensorMeta,
        strided: tuple[int, int, int, int, list[int]],
    ) -> torch.Tensor:
        outer_count, source_dim, start, length, output_shape = strided
        element_size = _DTYPE_NBYTES[record.dtype]
        # Recompute inner_count from the sliced dimension instead of relying on
        # output rank equivalence. This keeps the byte math tied to the source
        # layout while the returned tensor uses output_shape.
        partial_dim = next(
            idx
            for idx, (src, out) in enumerate(zip(record.shape, output_shape))
            if src != out
        )
        inner_count = math.prod(record.shape[partial_dim + 1 :])
        segment_elements = length * inner_count
        segment_bytes = segment_elements * element_size
        if segment_bytes <= 0:
            raise RuntimeError(f"Invalid empty strided slice for {record.name}")

        total_elements = math.prod(output_shape)
        expected_elements = outer_count * segment_elements
        if total_elements != expected_elements:
            raise RuntimeError(
                f"Internal strided slice shape mismatch for {record.name}: "
                f"output={output_shape}, expected_elements={expected_elements}"
            )

        self._maybe_gate(f"before reading {record.name}[strided-slice]", force=True)
        t0 = time.perf_counter()
        tensor = torch.empty(output_shape, dtype=record.dtype, device="cpu")
        time_alloc = time.perf_counter() - t0
        self._maybe_gate(f"after allocating {record.name}[strided-slice]", force=True)

        flat = tensor.reshape(-1)
        t1 = time.perf_counter()
        odirect_file = self._open_file(record.file_path)
        for outer_idx in range(outer_count):
            source_element_offset = (
                outer_idx * source_dim * inner_count + start * inner_count
            )
            target_element_offset = outer_idx * segment_elements
            target_view = flat.narrow(0, target_element_offset, segment_elements)
            odirect_file.read_record_into_tensor(
                target_view,
                record.offset + source_element_offset * element_size,
                segment_bytes,
                gate=self._note_loaded_bytes,
            )
        time_read = time.perf_counter() - t1

        selected_bytes = total_elements * element_size
        self._stats.tensors_read += 1
        self._stats.tensors_read_sliced += 1
        self._stats.bytes_tensor_payload += selected_bytes
        self._stats.bytes_sliced_tensor_payload += selected_bytes
        self._stats.time_alloc += time_alloc
        self._stats.time_read += time_read
        self._maybe_gate(f"after reading {record.name}[strided-slice]", force=True)
        return tensor

    def _read_strided_slice_into_cpu(
        self,
        record: TensorMeta,
        strided: tuple[int, int, int, int, list[int]],
        dst: torch.Tensor,
        *,
        force_gate: bool = True,
    ) -> None:
        outer_count, source_dim, start, length, output_shape = strided
        element_size = _DTYPE_NBYTES[record.dtype]
        partial_dim = next(
            idx
            for idx, (src, out) in enumerate(zip(record.shape, output_shape))
            if src != out
        )
        inner_count = math.prod(record.shape[partial_dim + 1 :])
        segment_elements = length * inner_count
        segment_bytes = segment_elements * element_size
        if segment_bytes <= 0:
            raise RuntimeError(f"Invalid empty strided slice for {record.name}")

        total_elements = math.prod(output_shape)
        expected_elements = outer_count * segment_elements
        if total_elements != expected_elements:
            raise RuntimeError(
                f"Internal strided slice shape mismatch for {record.name}: "
                f"output={output_shape}, expected_elements={expected_elements}"
            )

        if force_gate:
            self._maybe_gate(f"before reading {record.name}[strided-slice]", force=True)
        flat = dst.reshape(-1)
        t0 = time.perf_counter()
        odirect_file = self._open_file(record.file_path)
        for outer_idx in range(outer_count):
            source_element_offset = (
                outer_idx * source_dim * inner_count + start * inner_count
            )
            target_element_offset = outer_idx * segment_elements
            target_view = flat.narrow(0, target_element_offset, segment_elements)
            odirect_file.read_record_into_tensor(
                target_view,
                record.offset + source_element_offset * element_size,
                segment_bytes,
                gate=self._note_loaded_bytes,
            )
        time_read = time.perf_counter() - t0

        selected_bytes = total_elements * element_size
        self._stats.tensors_read += 1
        self._stats.tensors_read_sliced += 1
        self._stats.bytes_tensor_payload += selected_bytes
        self._stats.bytes_sliced_tensor_payload += selected_bytes
        self._stats.time_read += time_read
        if force_gate:
            self._maybe_gate(f"after reading {record.name}[strided-slice]", force=True)

    def _read_record_cpu(self, record: TensorMeta, *, sliced: bool) -> torch.Tensor:
        self._maybe_gate(f"before reading {record.name}", force=True)
        odirect_file = self._open_file(record.file_path)
        tensor, time_alloc, time_read = self._read_record_tensor(
            record,
            odirect_file,
        )
        self._stats.tensors_read += 1
        self._stats.bytes_tensor_payload += record.size
        if sliced:
            self._stats.tensors_read_sliced += 1
            self._stats.bytes_sliced_tensor_payload += record.size
        else:
            self._stats.tensors_read_full += 1
            self._stats.bytes_full_tensor_payload += record.size
        self._stats.time_alloc += time_alloc
        self._stats.time_read += time_read
        self._maybe_gate(f"after reading {record.name}", force=True)
        return tensor

    def _read_record_tensor(
        self,
        record: TensorMeta,
        odirect_file: "_ODirectFile",
    ) -> tuple[torch.Tensor, float, float]:
        force_allocation_gate = record.size >= self._loader._allocation_gate_min_bytes
        if force_allocation_gate:
            self._maybe_gate(f"before allocating {record.name}", force=True)
        t0 = time.perf_counter()
        tensor = torch.empty(record.shape, dtype=record.dtype, device="cpu")
        time_alloc = time.perf_counter() - t0
        if force_allocation_gate:
            self._maybe_gate(f"after allocating {record.name}", force=True)

        t0 = time.perf_counter()
        odirect_file.read_record_into_tensor(
            tensor,
            record.offset,
            record.size,
            gate=(
                self._note_loaded_bytes
                if record.size > odirect_file.window_size
                else None
            ),
        )
        time_read = time.perf_counter() - t0
        if record.size <= odirect_file.window_size:
            self._note_loaded_bytes(record.size)
        return tensor, time_alloc, time_read

    def _read_record_into_cpu(
        self,
        record: TensorMeta,
        dst: torch.Tensor,
        *,
        sliced: bool,
        force_gate: bool = True,
    ) -> None:
        if force_gate:
            self._maybe_gate(f"before reading {record.name}", force=True)
        if tuple(dst.shape) != record.shape:
            raise RuntimeError(
                f"Destination shape mismatch for {record.name}: "
                f"dst={list(dst.shape)}, source={record.shape}"
            )
        odirect_file = self._open_file(record.file_path)
        t0 = time.perf_counter()
        odirect_file.read_record_into_tensor(
            dst,
            record.offset,
            record.size,
            gate=self._note_loaded_bytes,
        )
        time_read = time.perf_counter() - t0
        self._stats.tensors_read += 1
        self._stats.bytes_tensor_payload += record.size
        if sliced:
            self._stats.tensors_read_sliced += 1
            self._stats.bytes_sliced_tensor_payload += record.size
        else:
            self._stats.tensors_read_full += 1
            self._stats.bytes_full_tensor_payload += record.size
        self._stats.time_read += time_read
        if force_gate:
            self._maybe_gate(f"after reading {record.name}", force=True)

    def skip(self, name: str, reason: str) -> None:
        self._stats.tensors_skipped += 1
        if not self.catalog.has(name):
            logger.debug(
                "uma_odirect_safetensors skipping absent tensor %s reason=%s",
                name,
                reason,
            )
            return
        record = self.catalog.get(name)
        self._stats.bytes_skipped_payload += record.size
        logger.debug(
            "uma_odirect_safetensors skipping tensor %s bytes=%s reason=%s",
            name,
            _format_gib(record.size),
            reason,
        )

    def log_stats(self, label: str) -> None:
        self._stats.log(label)
        expected = self._expected_read_summary
        if expected is None or expected.expected_bytes_read <= 0:
            return
        threshold = expected.expected_bytes_read * 1.10
        if self._stats.bytes_read <= threshold:
            return
        logger.warning(
            "uma_odirect_safetensors actual read amplification exceeded "
            "schedule expectation (%s): actual_bytes_read=%s "
            "expected_bytes_read=%s ratio=%.2fx threshold=1.10x",
            label,
            _format_gib(self._stats.bytes_read),
            _format_gib(expected.expected_bytes_read),
            self._stats.bytes_read / expected.expected_bytes_read,
        )

    def set_expected_read_summary(self, summary: ReadScheduleSummary) -> None:
        self._expected_read_summary = summary

    def stats_snapshot(self) -> dict[str, int | float]:
        stats = replace(self._stats)
        if self._open_file_handle is not None:
            stats.collect_file(self._open_file_handle)
        return stats.snapshot()

    def _maybe_gate(self, reason: str, force: bool = False) -> None:
        if (
            not force
            and self._loader._gate_interval_bytes > 0
            and self._bytes_since_gate < self._loader._gate_interval_bytes
        ):
            return
        t0 = time.perf_counter()
        self._loader._gate_memory(reason)
        self._stats.time_gate += time.perf_counter() - t0
        self._bytes_since_gate = 0

    def _note_loaded_bytes(self, nbytes: int) -> None:
        self._bytes_since_gate += nbytes
        self._maybe_gate(f"after {self._bytes_since_gate} loaded bytes")


@dataclass
class _RemoteReadStats:
    tensors_read: int = 0
    tensors_read_full: int = 0
    tensors_read_sliced: int = 0
    tensors_skipped: int = 0
    batch_requests: int = 0
    batch_tensors: int = 0
    bytes_stream_recv: int = 0
    bytes_tensor_payload: int = 0
    bytes_full_tensor_payload: int = 0
    bytes_sliced_tensor_payload: int = 0
    bytes_skipped_payload: int = 0
    time_alloc: float = 0.0
    time_read: float = 0.0

    def snapshot(self) -> dict[str, int | float]:
        return {
            "tensors_read": self.tensors_read,
            "tensors_read_full": self.tensors_read_full,
            "tensors_read_sliced": self.tensors_read_sliced,
            "tensors_skipped": self.tensors_skipped,
            "batch_requests": self.batch_requests,
            "batch_tensors": self.batch_tensors,
            "bytes_stream_recv": self.bytes_stream_recv,
            "remote_stream_bytes_recv": self.bytes_stream_recv,
            "bytes_tensor_payload": self.bytes_tensor_payload,
            "bytes_full_tensor_payload": self.bytes_full_tensor_payload,
            "bytes_sliced_tensor_payload": self.bytes_sliced_tensor_payload,
            "bytes_skipped_payload": self.bytes_skipped_payload,
            "time_alloc": self.time_alloc,
            "time_read": self.time_read,
        }


class RemoteODirectSafetensorsWeightSourceServer:
    """Dedicated TCP owner for Phase-1 remote UMA O_DIRECT reads.

    The server wraps an already constructed ``ODirectSafetensorsWeightSource``.
    It is intentionally narrow: requests name a WeightSource operation, the
    owner validates through the local source/catalog path, reads with O_DIRECT,
    and returns a CPU tensor payload.  It never serves arbitrary byte ranges.
    """

    def __init__(
        self,
        source: ODirectSafetensorsWeightSource,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        auth_token: str,
        request_timeout: float = 30.0,
        max_batch_payload_bytes: int = _REMOTE_DEFAULT_MAX_BATCH_PAYLOAD_BYTES,
    ) -> None:
        if not auth_token:
            raise ValueError("Remote O_DIRECT weight source requires an auth token")
        self.source = source
        self.auth_token = auth_token
        self.request_timeout = request_timeout
        if max_batch_payload_bytes <= 0:
            raise ValueError("Remote O_DIRECT batch payload limit must be positive")
        self.max_batch_payload_bytes = max_batch_payload_bytes
        self._source_lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._connections_accepted = 0
        outer = self

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                self.request.settimeout(outer.request_timeout)
                with outer._connection_lock:
                    outer._connections_accepted += 1
                while True:
                    try:
                        request = _recv_frame_or_none(
                            self.request,
                            max_payload_bytes=0,
                        )
                        if request is None:
                            return
                        # ODirectSafetensorsWeightSource keeps one mutable
                        # O_DIRECT file/window cache and shared counters.
                        # Phase 1 is a correctness-first sync RPC path, so
                        # serialize all owner source access instead of letting
                        # handler threads race.
                        with outer._source_lock:
                            response = outer._handle_request(request)
                    except Exception as exc:  # noqa: BLE001 - propagate as RPC error.
                        response = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        _send_frame(self.request, response)
                        return
                    _send_frame(self.request, response)

        class _Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _Server((host, port), _Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address
        return str(host), int(port)

    @property
    def connections_accepted(self) -> int:
        with self._connection_lock:
            return self._connections_accepted

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="remote-odirect-weight-source",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "RemoteODirectSafetensorsWeightSourceServer":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def _check_request(self, request: object) -> dict[str, object]:
        if not isinstance(request, dict):
            raise RuntimeError("Remote O_DIRECT request must be a dict")
        auth_token = request.get("auth_token")
        if not isinstance(auth_token, str) or not hmac.compare_digest(
            auth_token,
            self.auth_token,
        ):
            raise RuntimeError("Remote O_DIRECT request failed authentication")
        op = request.get("op")
        if not isinstance(op, str) or not op:
            raise RuntimeError("Remote O_DIRECT request is missing op")
        return request

    def _handle_request(self, request_obj: object) -> dict[str, object]:
        request = self._check_request(request_obj)
        op = request["op"]
        if op == "source_capability":
            loader = getattr(self.source, "_loader", None)
            if loader is None:
                raise RuntimeError(
                    "Remote O_DIRECT owner source does not expose loader capability"
                )
            return {
                "ok": True,
                "capability": {
                    "chunk_size": int(loader._chunk_size),
                    "window_size": int(loader._window_size),
                    "alignment": int(loader._alignment),
                    "supports_strided_read": True,
                    "max_batch_payload_bytes": int(self.max_batch_payload_bytes),
                    "max_batch_items": int(_REMOTE_MAX_BATCH_ITEMS),
                },
            }
        if op == "read_full":
            name = self._request_name(request)
            return {"ok": True, "tensor": _tensor_to_wire(self.source.read_full_cpu(name))}
        if op == "read_slice":
            name = self._request_name(request)
            slices = _decode_slice_selection(  # type: ignore[arg-type]
                request.get("source_slices")
            )
            if slices is None:
                raise RuntimeError("read_slice request missing source_slices")
            return {
                "ok": True,
                "tensor": _tensor_to_wire(self.source.read_slice_cpu(name, slices)),
            }
        if op == "read_many":
            items = request.get("items")
            if not isinstance(items, list) or not items:
                raise RuntimeError("read_many request must carry non-empty items")
            if len(items) > _REMOTE_MAX_BATCH_ITEMS:
                raise RuntimeError(
                    "read_many request item count exceeds limit: "
                    f"{len(items)} > {_REMOTE_MAX_BATCH_ITEMS}"
                )
            decoded: list[_RemoteReadRequest] = []
            total_payload = 0
            for item in items:
                if not isinstance(item, dict):
                    raise RuntimeError(f"Invalid read_many item: {item!r}")
                name = self._request_name(item)
                slices = _decode_slice_selection(  # type: ignore[arg-type]
                    item.get("source_slices")
                )
                record = self.source.catalog.get(name)
                raw_segments = item.get("segments")
                if raw_segments is not None:
                    if slices is not None:
                        raise RuntimeError(
                            "read_many item cannot combine source_slices and segments"
                        )
                    shape = item.get("staging_shape")
                    if (
                        not isinstance(shape, (list, tuple))
                        or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
                    ):
                        raise RuntimeError(
                            f"Invalid read_many staging_shape for {name}: {shape!r}"
                        )
                    segments = _decode_segments(raw_segments)  # type: ignore[arg-type]
                    staging_shape = tuple(shape)
                    validate_weight_plan_read_segments(
                        record,
                        WeightPlanEntry(
                            checkpoint_name=name,
                            target_name=name,
                            read_segments=segments,
                            staging_shape=staging_shape,
                            read_into_cpu=True,
                        ),
                    )
                    payload_bytes = (
                        math.prod(staging_shape) * _DTYPE_NBYTES[record.dtype]
                    )
                    decoded.append(
                        _RemoteReadRequest(
                            name,
                            read_segments=segments,
                            staging_shape=staging_shape,
                        )
                    )
                else:
                    payload_bytes = _source_tensor_payload_bytes(record, slices)
                    decoded.append(_RemoteReadRequest(name, slices))
                total_payload += payload_bytes
                if total_payload > self.max_batch_payload_bytes:
                    raise RuntimeError(
                        "read_many request payload exceeds limit: "
                        f"{total_payload} > {self.max_batch_payload_bytes}"
                    )
            tensors: list[torch.Tensor | None] = [None] * len(decoded)
            segment_groups: dict[
                str,
                list[
                    tuple[
                        int,
                        torch.Tensor,
                        tuple[WeightPlanReadSegment, ...],
                    ]
                ],
            ] = defaultdict(list)
            for index, item in enumerate(decoded):
                if item.read_segments is not None:
                    if item.staging_shape is None:
                        raise RuntimeError(
                            f"read_many segmented item missing shape: {item}"
                        )
                    tensor = self.source.empty_cpu_shape(
                        item.checkpoint_name,
                        item.staging_shape,
                    )
                    tensors[index] = tensor
                    segment_groups[item.checkpoint_name].append(
                        (index, tensor, item.read_segments)
                    )
                elif item.source_slices is None:
                    tensors[index] = self.source.read_full_cpu(item.checkpoint_name)
                else:
                    tensors[index] = self.source.read_slice_cpu(
                        item.checkpoint_name,
                        item.source_slices,
                    )
            read_segment_group_into_cpu = getattr(
                self.source,
                "read_segment_group_into_cpu",
                None,
            )
            for name, group in segment_groups.items():
                requests = tuple((tensor, segments) for _index, tensor, segments in group)
                if callable(read_segment_group_into_cpu):
                    read_segment_group_into_cpu(name, requests)
                else:
                    for _index, tensor, segments in group:
                        self.source.read_segments_into_cpu(name, tensor, segments)
            concrete_tensors = []
            for tensor in tensors:
                if tensor is None:
                    raise RuntimeError("Internal read_many tensor was not populated")
                concrete_tensors.append(tensor)
            return {
                "ok": True,
                "tensors": [_tensor_to_wire(tensor) for tensor in concrete_tensors],
            }
        if op == "read_segments":
            name = self._request_name(request)
            shape = request.get("staging_shape")
            if (
                not isinstance(shape, (list, tuple))
                or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
            ):
                raise RuntimeError(f"Invalid read_segments staging_shape: {shape!r}")
            segments = _decode_segments(request.get("segments"))  # type: ignore[arg-type]
            record = self.source.catalog.get(name)
            entry = WeightPlanEntry(
                checkpoint_name=name,
                target_name=name,
                read_segments=segments,
                staging_shape=tuple(shape),
                read_into_cpu=True,
            )
            validate_weight_plan_read_segments(record, entry)
            tensor = self.source.empty_cpu_shape(name, tuple(shape))
            self.source.read_segments_into_cpu(name, tensor, segments)
            return {"ok": True, "tensor": _tensor_to_wire(tensor)}
        if op == "skip":
            name = self._request_name(request)
            reason = request.get("reason")
            if not isinstance(reason, str):
                raise RuntimeError("skip request missing reason")
            self.source.skip(name, reason)
            return {"ok": True}
        if op == "stats_snapshot":
            return {"ok": True, "stats": self.source.stats_snapshot()}
        if op == "close_files":
            self.source.close_files()
            return {"ok": True}
        raise RuntimeError(f"Unknown remote O_DIRECT request op: {op!r}")

    @staticmethod
    def _request_name(request: dict[str, object]) -> str:
        name = request.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Remote O_DIRECT request is missing tensor name")
        return name


class RemoteODirectSafetensorsWeightSource:
    """Remote rank WeightSource for Phase-1 TCP streaming.

    The remote source owns only metadata and a TCP endpoint.  It never opens a
    safetensors payload file; all payload materialization comes from the owner
    server.  ``read_segment_group_into_cpu`` is deliberately omitted in Phase 1
    so the executor falls back to per-entry ``read_segments_into_cpu``.
    """

    def __init__(
        self,
        catalog: TensorCatalog,
        *,
        host: str,
        port: int,
        auth_token: str,
        request_timeout: float = 30.0,
        max_batch_payload_bytes: int = _REMOTE_DEFAULT_MAX_BATCH_PAYLOAD_BYTES,
    ) -> None:
        if not auth_token:
            raise ValueError("Remote O_DIRECT weight source requires an auth token")
        self.catalog = catalog
        records = catalog.records()
        self._max_payload_bytes = max((record.size for record in records), default=0)
        if max_batch_payload_bytes <= 0:
            raise ValueError("Remote O_DIRECT batch payload limit must be positive")
        self._max_batch_payload_bytes = max_batch_payload_bytes
        self._host = host
        self._port = port
        self._auth_token = auth_token
        self._request_timeout = request_timeout
        self._stats = _RemoteReadStats()
        self._expected_read_summary: ReadScheduleSummary | None = None
        self._sock: socket.socket | None = None
        self._sock_lock = threading.Lock()
        capability = self._request_capability()
        self._chunk_size = capability["chunk_size"]
        self._window_size = capability["window_size"]
        self._alignment = capability["alignment"]
        self._supports_strided_read = capability["supports_strided_read"]
        self._max_batch_payload_bytes = min(
            self._max_batch_payload_bytes,
            capability["max_batch_payload_bytes"],
        )
        self._max_batch_items = capability["max_batch_items"]

    def read_full_cpu(self, name: str) -> torch.Tensor:
        record = self.catalog.get(name)
        t0 = time.perf_counter()
        tensor = self._request_tensor(
            {"op": "read_full", "name": name},
            max_payload_bytes=record.size,
        )
        self._stats.time_read += time.perf_counter() - t0
        self._record_tensor_read(record.size, sliced=False)
        return tensor

    def iter_full_tensors(
        self,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        logger.info(
            "uma_odirect_safetensors using remote source iterator: tensors=%d "
            "total=%s",
            len(self.catalog.records()),
            _format_gib(self.catalog.total_bytes()),
        )
        for record in self.catalog.records():
            yield record.name, self.read_full_cpu(record.name)

    def read_slice_cpu(
        self,
        name: str,
        source_slices: tuple[slice | int, ...],
    ) -> torch.Tensor:
        record = self.catalog.get(name)
        expected_shape = _source_tensor_shape(record, source_slices)
        expected_payload = math.prod(expected_shape) * _DTYPE_NBYTES[record.dtype]
        t0 = time.perf_counter()
        tensor = self._request_tensor(
            {
                "op": "read_slice",
                "name": name,
                "source_slices": _encode_slice_selection(source_slices),
            },
            max_payload_bytes=expected_payload,
        )
        self._stats.time_read += time.perf_counter() - t0
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"Remote read_slice_cpu shape mismatch for {name}: "
                f"got={list(tensor.shape)}, expected={expected_shape}"
            )
        self._record_tensor_read(tensor.numel() * tensor.element_size(), sliced=True)
        return tensor

    def read_many_max_payload_bytes(self) -> int:
        return self._max_batch_payload_bytes

    def read_many_max_items(self) -> int:
        return self._max_batch_items

    def read_many_cpu(
        self,
        requests: tuple[_RemoteReadRequest, ...],
    ) -> tuple[torch.Tensor, ...]:
        if not requests:
            return ()
        if len(requests) > self._max_batch_items:
            raise RuntimeError(
                "Remote read_many_cpu request item count exceeds limit: "
                f"{len(requests)} > {self._max_batch_items}"
            )
        items: list[dict[str, object]] = []
        expected_payloads: list[int] = []
        expected_shapes: list[list[int]] = []
        total_payload = 0
        for request in requests:
            record = self.catalog.get(request.checkpoint_name)
            if request.read_segments is not None:
                if request.source_slices is not None:
                    raise RuntimeError(
                        "Remote read_many_cpu request cannot combine "
                        "source_slices and read_segments"
                    )
                if request.staging_shape is None:
                    raise RuntimeError(
                        "Remote read_many_cpu segmented request missing staging_shape"
                    )
                validate_weight_plan_read_segments(
                    record,
                    WeightPlanEntry(
                        checkpoint_name=request.checkpoint_name,
                        target_name=request.checkpoint_name,
                        read_segments=request.read_segments,
                        staging_shape=request.staging_shape,
                        read_into_cpu=True,
                    ),
                )
                expected_shape = list(request.staging_shape)
            else:
                expected_shape = _source_tensor_shape(record, request.source_slices)
            expected_payload = math.prod(expected_shape) * _DTYPE_NBYTES[record.dtype]
            total_payload += expected_payload
            if total_payload > self._max_batch_payload_bytes:
                raise RuntimeError(
                    "Remote read_many_cpu payload exceeds limit: "
                    f"{total_payload} > {self._max_batch_payload_bytes}"
                )
            item: dict[str, object] = {
                "name": request.checkpoint_name,
                "source_slices": _encode_slice_selection(request.source_slices),
            }
            if request.read_segments is not None:
                item["segments"] = _encode_segments(request.read_segments)
                item["staging_shape"] = tuple(request.staging_shape or ())
            items.append(item)
            expected_shapes.append(expected_shape)
            expected_payloads.append(expected_payload)

        t0 = time.perf_counter()
        response = self._request(
            {"op": "read_many", "items": items},
            max_payload_bytes=total_payload,
        )
        self._stats.time_read += time.perf_counter() - t0
        encoded = response.get("tensors")
        if not isinstance(encoded, list):
            raise RuntimeError("Remote O_DIRECT owner returned no tensor batch")
        if len(encoded) != len(requests):
            raise RuntimeError(
                "Remote O_DIRECT owner returned unexpected tensor batch size: "
                f"got={len(encoded)}, expected={len(requests)}"
            )
        tensors: list[torch.Tensor] = []
        for request, item, expected_shape, expected_payload in zip(
            requests,
            encoded,
            expected_shapes,
            expected_payloads,
        ):
            if not isinstance(item, dict):
                raise RuntimeError("Remote O_DIRECT owner returned invalid tensor batch")
            tensor = _tensor_from_wire(item)
            if list(tensor.shape) != expected_shape:
                raise RuntimeError(
                    "Remote read_many_cpu shape mismatch: "
                    f"got={list(tensor.shape)}, expected={expected_shape}"
                )
            tensors.append(tensor)
            self._record_tensor_read(
                expected_payload,
                sliced=(
                    request.source_slices is not None
                    or request.read_segments is not None
                ),
            )
        self._stats.batch_requests += 1
        self._stats.batch_tensors += len(tensors)
        return tuple(tensors)

    def _request_capability(self) -> dict[str, int | bool]:
        response = self._request({"op": "source_capability"})
        capability = response.get("capability")
        if not isinstance(capability, dict):
            raise RuntimeError("Remote O_DIRECT owner returned no source capability")

        def _required_positive_int(key: str) -> int:
            value = capability.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(
                    f"Remote O_DIRECT owner returned invalid capability {key}: "
                    f"{value!r}"
                )
            return value

        chunk_size = _required_positive_int("chunk_size")
        window_size = _required_positive_int("window_size")
        alignment = _required_positive_int("alignment")
        max_batch_payload_bytes = _required_positive_int("max_batch_payload_bytes")
        max_batch_items = _required_positive_int("max_batch_items")
        supports_strided_read = capability.get("supports_strided_read")
        if not isinstance(supports_strided_read, bool):
            raise RuntimeError(
                "Remote O_DIRECT owner returned invalid capability "
                f"supports_strided_read: {supports_strided_read!r}"
            )
        if alignment & (alignment - 1) != 0:
            raise RuntimeError(
                f"Remote O_DIRECT owner returned non-power-of-two alignment: {alignment}"
            )
        if window_size < chunk_size:
            raise RuntimeError(
                "Remote O_DIRECT owner returned window_size smaller than chunk_size: "
                f"{window_size} < {chunk_size}"
            )
        return {
            "chunk_size": chunk_size,
            "window_size": window_size,
            "alignment": alignment,
            "supports_strided_read": supports_strided_read,
            "max_batch_payload_bytes": max_batch_payload_bytes,
            "max_batch_items": max_batch_items,
        }

    def read_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
        target_slices: tuple[slice | int, ...] | None = None,
        force_gate: bool = True,
    ) -> None:
        del force_gate
        record = self.catalog.get(name)
        if dst.device.type != "cpu":
            raise RuntimeError(
                f"read_into_cpu requires a CPU destination for {name}, got {dst.device}"
            )
        if dst.dtype != record.dtype:
            raise RuntimeError(
                f"read_into_cpu dtype mismatch for {name}: "
                f"dst={dst.dtype}, source={record.dtype}"
            )
        if not dst.is_contiguous():
            raise RuntimeError(f"read_into_cpu requires a contiguous dst for {name}")
        target = _select_contiguous_target_view(dst, target_slices, name)
        tensor = (
            self.read_full_cpu(name)
            if source_slices is None
            else self.read_slice_cpu(name, source_slices)
        )
        if tuple(target.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"Remote read_into_cpu shape mismatch for {name}: "
                f"target={list(target.shape)}, tensor={list(tensor.shape)}"
            )
        target.copy_(tensor)

    def read_segments_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        segments: tuple[WeightPlanReadSegment, ...],
    ) -> None:
        record = self.catalog.get(name)
        if dst.device.type != "cpu":
            raise RuntimeError(
                f"read_segments_into_cpu requires CPU destination for {name}, "
                f"got {dst.device}"
            )
        if dst.dtype != record.dtype:
            raise RuntimeError(
                f"read_segments_into_cpu dtype mismatch for {name}: "
                f"dst={dst.dtype}, source={record.dtype}"
            )
        if not dst.is_contiguous():
            raise RuntimeError(
                f"read_segments_into_cpu requires contiguous dst for {name}"
            )
        t0 = time.perf_counter()
        expected_payload = dst.numel() * dst.element_size()
        tensor = self._request_tensor(
            {
                "op": "read_segments",
                "name": name,
                "staging_shape": tuple(int(dim) for dim in dst.shape),
                "segments": _encode_segments(segments),
            },
            max_payload_bytes=expected_payload,
        )
        self._stats.time_read += time.perf_counter() - t0
        if tuple(tensor.shape) != tuple(dst.shape):
            raise RuntimeError(
                f"Remote read_segments_into_cpu shape mismatch for {name}: "
                f"got={list(tensor.shape)}, expected={list(dst.shape)}"
            )
        dst.copy_(tensor)
        self._record_tensor_read(dst.numel() * dst.element_size(), sliced=True)

    def empty_cpu(
        self,
        name: str,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
    ) -> torch.Tensor:
        record = self.catalog.get(name)
        shape = _source_tensor_shape(record, source_slices)
        return self.empty_cpu_shape(name, tuple(shape))

    def empty_cpu_shape(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        record = self.catalog.get(name)
        if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise RuntimeError(f"Invalid CPU staging shape for {name}: {shape!r}")
        t0 = time.perf_counter()
        tensor = torch.empty(shape, dtype=record.dtype, device="cpu")
        self._stats.time_alloc += time.perf_counter() - t0
        return tensor

    def skip(self, name: str, reason: str) -> None:
        self._stats.tensors_skipped += 1
        record = self.catalog.get(name) if self.catalog.has(name) else None
        if record is not None:
            self._stats.bytes_skipped_payload += record.size
        self._request({"op": "skip", "name": name, "reason": reason})

    def close_files(self) -> None:
        self._request({"op": "close_files"})

    def owner_stats_snapshot(self) -> dict[str, int | float]:
        response = self._request({"op": "stats_snapshot"})
        stats = response.get("stats")
        if not isinstance(stats, dict):
            raise RuntimeError("Remote O_DIRECT owner returned invalid stats")
        return stats  # type: ignore[return-value]

    def set_expected_read_summary(self, summary: ReadScheduleSummary) -> None:
        self._expected_read_summary = summary

    def stats_snapshot(self) -> dict[str, int | float]:
        return self._stats.snapshot()

    def log_stats(self, label: str) -> None:
        stats = self._stats.snapshot()
        logger.info(
            "uma_odirect_safetensors remote source stats (%s): "
            "tensors_read=%d tensors_read_full=%d tensors_read_sliced=%d "
            "tensors_skipped=%d batch_requests=%d batch_tensors=%d "
            "stream_recv=%s tensor_payload=%s "
            "full_payload=%s sliced_payload=%s skipped_payload=%s",
            label,
            stats["tensors_read"],
            stats["tensors_read_full"],
            stats["tensors_read_sliced"],
            stats["tensors_skipped"],
            stats["batch_requests"],
            stats["batch_tensors"],
            _format_gib(stats["remote_stream_bytes_recv"]),
            _format_gib(stats["bytes_tensor_payload"]),
            _format_gib(stats["bytes_full_tensor_payload"]),
            _format_gib(stats["bytes_sliced_tensor_payload"]),
            _format_gib(stats["bytes_skipped_payload"]),
        )
        try:
            owner_stats = self.owner_stats_snapshot()
        except Exception:
            logger.warning(
                "uma_odirect_safetensors remote source could not fetch owner "
                "stats (%s)",
                label,
                exc_info=True,
            )
            return
        logger.info(
            "uma_odirect_safetensors remote owner stats (%s): "
            "files_opened=%d tensors_read=%d tensors_read_full=%d "
            "tensors_read_sliced=%d tensors_skipped=%d direct_reads=%d "
            "window_loads=%d window_hits=%d bytes_read=%s bytes_copied=%s",
            label,
            owner_stats["files_opened"],
            owner_stats["tensors_read"],
            owner_stats["tensors_read_full"],
            owner_stats["tensors_read_sliced"],
            owner_stats["tensors_skipped"],
            owner_stats["direct_reads"],
            owner_stats["window_loads"],
            owner_stats["window_hits"],
            _format_gib(owner_stats["bytes_read"]),
            _format_gib(owner_stats["bytes_copied"]),
        )
        self._close_socket()

    def _record_tensor_read(self, payload_bytes: int, *, sliced: bool) -> None:
        self._stats.tensors_read += 1
        self._stats.bytes_tensor_payload += payload_bytes
        self._stats.bytes_stream_recv += payload_bytes
        if sliced:
            self._stats.tensors_read_sliced += 1
            self._stats.bytes_sliced_tensor_payload += payload_bytes
        else:
            self._stats.tensors_read_full += 1
            self._stats.bytes_full_tensor_payload += payload_bytes

    def _request_tensor(
        self,
        request: dict[str, object],
        *,
        max_payload_bytes: int,
    ) -> torch.Tensor:
        if max_payload_bytes < 0 or max_payload_bytes > self._max_payload_bytes:
            raise RuntimeError(
                "Invalid remote tensor response limit: "
                f"{max_payload_bytes} > catalog max {self._max_payload_bytes}"
            )
        response = self._request(request, max_payload_bytes=max_payload_bytes)
        encoded = response.get("tensor")
        if not isinstance(encoded, dict):
            raise RuntimeError("Remote O_DIRECT owner returned no tensor")
        return _tensor_from_wire(encoded)

    def _request(
        self,
        request: dict[str, object],
        *,
        max_payload_bytes: int = 0,
    ) -> dict[str, object]:
        request = dict(request)
        request["auth_token"] = self._auth_token
        with self._sock_lock:
            sock = self._ensure_socket()
            try:
                _send_frame(sock, request)
                response = _recv_frame(sock, max_payload_bytes=max_payload_bytes)
            except Exception:
                self._close_socket_locked()
                raise
        if not isinstance(response, dict):
            raise RuntimeError("Remote O_DIRECT owner returned invalid response")
        if not response.get("ok"):
            error = response.get("error")
            self._close_socket()
            raise RuntimeError(
                "Remote O_DIRECT owner rejected request"
                + (f": {error}" if isinstance(error, str) else "")
            )
        return response

    def _ensure_socket(self) -> socket.socket:
        if self._sock is None:
            sock = socket.create_connection(
                (self._host, self._port),
                timeout=self._request_timeout,
            )
            sock.settimeout(self._request_timeout)
            self._sock = sock
        return self._sock

    def _close_socket(self) -> None:
        with self._sock_lock:
            self._close_socket_locked()

    def _close_socket_locked(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


class _ConsumerProfile:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.count = 0
        self.bytes = 0
        self.by_key = defaultdict(lambda: [0, 0, 0.0])

    @staticmethod
    def _bucket(nbytes: int) -> str:
        if nbytes <= 4 * 1024:
            return "<=4KiB"
        if nbytes <= 64 * 1024:
            return "<=64KiB"
        if nbytes <= 1024 * 1024:
            return "<=1MiB"
        if nbytes <= 16 * 1024 * 1024:
            return "<=16MiB"
        return ">16MiB"

    @staticmethod
    def _name_group(name: str) -> str:
        parts = name.split(".")
        parts = ["#" if part.isdigit() else part for part in parts]
        if len(parts) >= 2 and parts[0] == "model" and parts[1] == "layers":
            return ".".join(parts[3:])
        return ".".join(parts[-3:])

    def record(self, record: _TensorRecord, elapsed: float) -> None:
        self.seconds += elapsed
        self.count += 1
        self.bytes += record.size
        key = (
            self._name_group(record.name),
            str(record.dtype),
            self._bucket(record.size),
        )
        item = self.by_key[key]
        item[0] += 1
        item[1] += record.size
        item[2] += elapsed

    def log(self) -> None:
        logger.info(
            "uma_odirect_safetensors consumer profile: total=%.3fs "
            "tensors=%d bytes=%s",
            self.seconds,
            self.count,
            _format_gib(self.bytes),
        )
        top_items = sorted(
            self.by_key.items(), key=lambda item: item[1][2], reverse=True
        )[:16]
        for (name_group, dtype, bucket), (count, nbytes, seconds) in top_items:
            logger.info(
                "uma_odirect_safetensors consumer profile item: "
                "seconds=%.3f count=%d bytes=%s name_group=%s dtype=%s bucket=%s",
                seconds,
                count,
                _format_gib(nbytes),
                name_group,
                dtype,
                bucket,
                )


class _AlignedBuffer:
    def __init__(self, size: int, alignment: int):
        self.size = size
        self.alignment = alignment
        self.ptr = ctypes.c_void_p()
        rc = _LIBC.posix_memalign(ctypes.byref(self.ptr), alignment, size)
        if rc != 0:
            raise OSError(rc, os.strerror(rc))

    def __del__(self) -> None:
        if getattr(self, "ptr", None) and self.ptr.value:
            _LIBC.free(self.ptr)
            self.ptr = ctypes.c_void_p()


class _ODirectFile:
    def __init__(
        self,
        path: str,
        chunk_size: int,
        alignment: int,
        window_size: int,
    ):
        flags = os.O_RDONLY
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError("O_DIRECT is not available on this platform")
        flags |= os.O_DIRECT
        self.path = path
        self.fd = os.open(path, flags)
        self.size = os.path.getsize(path)
        self.chunk_size = _round_up(chunk_size, alignment)
        self.alignment = alignment
        self.buffer = _AlignedBuffer(self.chunk_size + alignment, alignment)
        self.window_size = max(_round_up(window_size, alignment), self.chunk_size)
        self.window = _AlignedBuffer(self.window_size + alignment, alignment)
        self.window_start = 0
        self.window_valid = 0
        self.direct_reads = 0
        self.window_loads = 0
        self.window_hits = 0
        self.bytes_read = 0
        self.bytes_copied = 0

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "_ODirectFile":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def _pread(self, offset: int, size: int) -> int:
        return self._pread_into(self.buffer.ptr, offset, size)

    def _pread_into(self, ptr: ctypes.c_void_p, offset: int, size: int) -> int:
        ret = _LIBC.pread(self.fd, ptr, size, offset)
        if ret < 0:
            err = ctypes.get_errno()
            raise OSError(
                err,
                f"O_DIRECT pread failed for {self.path} at {offset} "
                f"size {size}: {os.strerror(err)}",
            )
        got = int(ret)
        self.direct_reads += 1
        self.bytes_read += got
        return got

    def _load_window(self, offset: int, size: int) -> None:
        read_start = _round_down(offset, self.alignment)
        required_size = _round_up((offset - read_start) + size, self.alignment)
        read_size = max(self.window_size, required_size)
        if read_size > self.window.size:
            raise RuntimeError(
                f"Internal O_DIRECT window too small: need {read_size}, "
                f"have {self.window.size}"
            )

        got = self._pread_into(self.window.ptr, read_start, read_size)
        if got == 0:
            raise EOFError(
                f"Unexpected EOF reading {self.path}: offset={read_start}, "
                f"size={read_size}"
            )
        if offset + size > read_start + got:
            raise EOFError(
                f"O_DIRECT window short read did not cover requested range for "
                f"{self.path}: requested={offset}+{size}, got={read_start}+{got}"
            )
        self.window_start = read_start
        self.window_valid = got
        self.window_loads += 1

    def _copy_from_window(self, tensor: torch.Tensor, offset: int, size: int) -> bool:
        window_end = self.window_start + self.window_valid
        if offset < self.window_start or offset + size > window_end:
            return False
        _LIBC.memmove(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(self.window.ptr.value + offset - self.window_start),
            size,
        )
        self.window_hits += 1
        self.bytes_copied += size
        return True

    def read_record_into_tensor(
        self,
        tensor: torch.Tensor,
        offset: int,
        size: int,
        gate: Callable[[int], None] | None = None,
    ) -> None:
        if size <= self.window_size:
            if not self._copy_from_window(tensor, offset, size):
                self._load_window(offset, size)
                if not self._copy_from_window(tensor, offset, size):
                    raise RuntimeError(
                        f"Internal O_DIRECT window miss after loading {self.path}: "
                        f"offset={offset}, size={size}"
                    )
            if gate is not None:
                gate(size)
            return

        self.read_into_tensor(tensor, offset, size, gate=gate)

    def read_into_tensor(
        self,
        tensor: torch.Tensor,
        offset: int,
        size: int,
        gate: Callable[[int], None] | None = None,
    ) -> None:
        if size == 0:
            return
        if offset < 0 or size < 0 or offset + size > self.size:
            raise RuntimeError(
                f"Invalid tensor byte range for {self.path}: "
                f"offset={offset}, size={size}, file_size={self.size}"
            )

        dst_base = tensor.data_ptr()
        copied = 0
        while copied < size:
            wanted_offset = offset + copied
            wanted_size = min(size - copied, self.chunk_size)
            read_start = _round_down(wanted_offset, self.alignment)
            read_end = _round_up(wanted_offset + wanted_size, self.alignment)
            read_size = read_end - read_start
            if read_size > self.buffer.size:
                raise RuntimeError(
                    f"Internal O_DIRECT buffer too small: need {read_size}, "
                    f"have {self.buffer.size}"
                )

            got = self._pread(read_start, read_size)
            if got == 0:
                raise EOFError(
                    f"Unexpected EOF reading {self.path}: offset={read_start}, "
                    f"size={read_size}"
                )

            available_start = read_start
            available_end = read_start + got
            copy_start = max(wanted_offset, available_start)
            copy_end = min(wanted_offset + wanted_size, available_end)
            if copy_end <= copy_start:
                raise EOFError(
                    f"O_DIRECT short read did not cover requested range for "
                    f"{self.path}: requested={wanted_offset}+{wanted_size}, "
                    f"got={read_start}+{got}"
                )

            src_offset = copy_start - read_start
            dst_offset = copy_start - offset
            copy_size = copy_end - copy_start
            _LIBC.memmove(
                ctypes.c_void_p(dst_base + dst_offset),
                ctypes.c_void_p(self.buffer.ptr.value + src_offset),
                copy_size,
            )
            self.bytes_copied += copy_size
            copied = dst_offset + copy_size
            if gate is not None:
                gate(copy_size)


class UmaODirectSafetensorsModelLoader(BaseModelLoader):
    """Fail-closed local safetensors loader using Linux O_DIRECT reads.

    This is intentionally narrower than vLLM's general safetensors loaders:
    local paths only, no mmap, no implicit downloads, no buffered fallback, and
    memory/PSI gates between tensors.
    """

    DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
    DEFAULT_WINDOW_SIZE = 64 * 1024 * 1024
    DEFAULT_ALIGNMENT = 4096
    DEFAULT_METADATA_LIMIT_MIB = 256
    DEFAULT_MIN_AVAILABLE_GIB = 20.0
    DEFAULT_PSI_GATE_SECONDS = 30.0
    DEFAULT_GATE_INTERVAL_MIB = 64
    DEFAULT_ALLOCATION_GATE_MIN_MIB = 16
    REMOTE_ROLE_ENV = "VLLM_UMA_ODIRECT_REMOTE_ROLE"
    REMOTE_HOST_ENV = "VLLM_UMA_ODIRECT_REMOTE_HOST"
    REMOTE_PORT_ENV = "VLLM_UMA_ODIRECT_REMOTE_PORT"
    REMOTE_TOKEN_ENV = "VLLM_UMA_ODIRECT_REMOTE_TOKEN"
    REMOTE_TIMEOUT_ENV = "VLLM_UMA_ODIRECT_REMOTE_TIMEOUT_SECONDS"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                "model_loader_extra_config must be a dict for "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )

        allowed_keys = {
            "alignment",
            "chunk_size",
            "window_size",
            "gate_interval_mib",
            "min_available_gib",
            "metadata_limit_mib",
            "allocation_gate_min_mib",
            "psi_gate_seconds",
            "max_swap_gib",
        }
        unexpected_keys = set(extra_config) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                "Unexpected extra config keys for uma_odirect_safetensors: "
                f"{unexpected_keys}"
            )

        self._alignment = self._get_positive_int(
            extra_config, "alignment", self.DEFAULT_ALIGNMENT
        )
        self._chunk_size = self._get_positive_int(
            extra_config, "chunk_size", self.DEFAULT_CHUNK_SIZE
        )
        self._window_size = self._get_positive_int(
            extra_config, "window_size", self.DEFAULT_WINDOW_SIZE
        )
        self._gate_interval_bytes = (
            self._get_non_negative_float(
                extra_config, "gate_interval_mib", self.DEFAULT_GATE_INTERVAL_MIB
            )
            * 1024
            * 1024
        )
        self._min_available_gib = self._get_non_negative_float(
            extra_config, "min_available_gib", self.DEFAULT_MIN_AVAILABLE_GIB
        )
        self._psi_gate_seconds = self._get_non_negative_float(
            extra_config, "psi_gate_seconds", self.DEFAULT_PSI_GATE_SECONDS
        )
        self._max_swap_gib = self._get_non_negative_float(
            extra_config, "max_swap_gib", 0.0
        )
        self._metadata_limit_bytes = (
            self._get_positive_int(
                extra_config, "metadata_limit_mib", self.DEFAULT_METADATA_LIMIT_MIB
            )
            * 1024
            * 1024
        )
        self._allocation_gate_min_bytes = (
            self._get_non_negative_float(
                extra_config,
                "allocation_gate_min_mib",
                self.DEFAULT_ALLOCATION_GATE_MIN_MIB,
            )
            * 1024
            * 1024
        )
        self._remote_owner_server: (
            RemoteODirectSafetensorsWeightSourceServer | None
        ) = None
        self._remote_owner_source: ODirectSafetensorsWeightSource | None = None
        if self._alignment & (self._alignment - 1) != 0:
            raise ValueError("alignment must be a power of two")
        if self._window_size < self._chunk_size:
            raise ValueError("window_size must be greater than or equal to chunk_size")
        if self._gate_interval_bytes > 0:
            if self._chunk_size > self._gate_interval_bytes:
                raise ValueError(
                    "chunk_size must be less than or equal to gate_interval_mib"
                )
            if self._window_size > self._gate_interval_bytes:
                raise ValueError(
                    "window_size must be less than or equal to gate_interval_mib"
                )
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "uma_odirect_safetensors does not support "
                "safetensors_load_strategy="
                f"{load_config.safetensors_load_strategy!r}; it uses "
                "O_DIRECT and rejects eager/prefetch paths."
            )

    @staticmethod
    def _get_positive_int(config: dict, key: str, default: int) -> int:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _get_non_negative_float(config: dict, key: str, default: float) -> float:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{key} must be a non-negative number, got {value!r}")
        return float(value)

    def _gate_memory(self, phase: str) -> None:
        gate_uma_memory(
            loader_label="uma_odirect_safetensors",
            phase=phase,
            min_available_gib=self._min_available_gib,
            psi_gate_seconds=self._psi_gate_seconds,
            max_swap_gib=self._max_swap_gib,
            logger=logger,
        )

    def _prepare_files(self, model_name_or_path: str) -> list[str]:
        if not os.path.isdir(model_name_or_path):
            raise RuntimeError(
                "uma_odirect_safetensors only accepts a local safetensors "
                f"directory, got {model_name_or_path!r}"
            )

        files = [
            os.path.join(model_name_or_path, name)
            for name in sorted(os.listdir(model_name_or_path))
            if name.endswith(".safetensors")
        ]
        if not files:
            raise RuntimeError(
                f"Cannot find any safetensors model weights in {model_name_or_path}"
            )
        return files

    def _read_records(self, files: list[str]) -> list[_TensorRecord]:
        return list(self._build_catalog(files).records())

    def _build_catalog(self, files: list[str]) -> TensorCatalog:
        return TensorCatalog.from_safetensors_files(
            files,
            metadata_limit_bytes=self._metadata_limit_bytes,
        )

    def _remote_timeout(self) -> float:
        raw = os.environ.get(self.REMOTE_TIMEOUT_ENV, "30")
        try:
            value = float(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"{self.REMOTE_TIMEOUT_ENV} must be a number, got {raw!r}"
            ) from exc
        if value <= 0:
            raise RuntimeError(
                f"{self.REMOTE_TIMEOUT_ENV} must be positive, got {raw!r}"
            )
        return value

    def _remote_auth_token(self) -> str:
        token = os.environ.get(self.REMOTE_TOKEN_ENV, "")
        if not token:
            raise RuntimeError(
                f"{self.REMOTE_TOKEN_ENV} is required for "
                "uma_odirect_safetensors remote owner/remote roles"
            )
        return token

    def _remote_port(self) -> int:
        raw = os.environ.get(self.REMOTE_PORT_ENV, "")
        try:
            port = int(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"{self.REMOTE_PORT_ENV} must be an integer, got {raw!r}"
            ) from exc
        if port <= 0 or port > 65535:
            raise RuntimeError(
                f"{self.REMOTE_PORT_ENV} must be a TCP port, got {raw!r}"
            )
        return port

    def _create_weight_source(
        self,
        model_or_path: str,
    ) -> ODirectSafetensorsWeightSource | RemoteODirectSafetensorsWeightSource:
        role = os.environ.get(self.REMOTE_ROLE_ENV, "").strip().lower()
        if not role:
            return ODirectSafetensorsWeightSource(self, model_or_path)
        if role not in {"owner", "remote"}:
            raise RuntimeError(
                f"{self.REMOTE_ROLE_ENV} must be 'owner' or 'remote', got {role!r}"
            )
        token = self._remote_auth_token()
        timeout = self._remote_timeout()
        port = self._remote_port()

        if role == "owner":
            source = ODirectSafetensorsWeightSource(self, model_or_path)
            host = os.environ.get(self.REMOTE_HOST_ENV)
            if not host:
                host = "127.0.0.1"
                logger.warning(
                    "uma_odirect_safetensors remote owner role did not set %s; "
                    "binding to loopback %s. This is safe for local loopback "
                    "tests but remote nodes will not be able to connect.",
                    self.REMOTE_HOST_ENV,
                    host,
                )
            if self._remote_owner_server is not None:
                self._remote_owner_server.close()
            server = RemoteODirectSafetensorsWeightSourceServer(
                source,
                host=host,
                port=port,
                auth_token=token,
                request_timeout=timeout,
            )
            server.start()
            actual_host, actual_port = server.address
            logger.info(
                "uma_odirect_safetensors remote owner server listening: %s:%d",
                actual_host,
                actual_port,
            )
            # Keep the server/source alive after this rank's local load returns:
            # a remote rank may still be fetching payload bytes.
            self._remote_owner_server = server
            self._remote_owner_source = source
            return source

        host = os.environ.get(self.REMOTE_HOST_ENV, "")
        if not host:
            raise RuntimeError(
                f"{self.REMOTE_HOST_ENV} is required when "
                f"{self.REMOTE_ROLE_ENV}=remote"
            )
        files = self._prepare_files(model_or_path)
        catalog = self._build_catalog(files)
        logger.info(
            "uma_odirect_safetensors using remote owner source: %s:%d "
            "files=%d tensors=%d payload=%s",
            host,
            port,
            len(files),
            len(catalog.records()),
            _format_gib(catalog.total_bytes()),
        )
        return RemoteODirectSafetensorsWeightSource(
            catalog,
            host=host,
            port=port,
            auth_token=token,
            request_timeout=timeout,
        )

    def _get_weights_iterator(
        self,
        model_or_path: str,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        source = self._create_weight_source(model_or_path)
        yield from source.iter_full_tensors()

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_files(model_config.model)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        source = self._create_weight_source(model_weights)
        source_hooks = resolve_weight_plan_source_hooks(model)
        if source_hooks is not None:
            build_weight_plan, load_weights_from_source = source_hooks
            logger.info(
                "uma_odirect_safetensors using model WeightSource path: %s",
                type(model).__name__,
            )
            plan = build_weight_plan(source.catalog)
            try:
                loaded_weights = load_weights_from_source(source, plan)
            finally:
                source.close_files()
                source.log_stats("model-source")
            if loaded_weights is None:
                raise RuntimeError(
                    "load_weights_from_source must return the loaded "
                    "parameter name set so UMA-safe loading can fail closed "
                    f"on uninitialized parameters: {type(model).__name__}"
                )
            verify_loaded_weights(model, loaded_weights)
            return

        logger.info(
            "uma_odirect_safetensors using compatibility iterator path: %s",
            type(model).__name__,
        )
        try:
            loaded_weights = model.load_weights(source.iter_full_tensors())
        finally:
            source.close_files()
            source.log_stats("compat-iterator")
        if loaded_weights is None:
            logger.warning(
                "uma_odirect_safetensors: model %s load_weights did not "
                "report a loaded set; skipping the fail-closed completeness "
                "check",
                type(model).__name__,
            )
            return
        verify_loaded_weights(model, loaded_weights)
