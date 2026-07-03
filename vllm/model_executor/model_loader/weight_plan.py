# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch import nn

_DTYPE_MAP: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_DTYPE_NBYTES: dict[torch.dtype, int] = {
    torch.bool: 1,
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_MAP["F8_E4M3"] = torch.float8_e4m3fn
    _DTYPE_NBYTES[torch.float8_e4m3fn] = 1
if hasattr(torch, "float8_e5m2"):
    _DTYPE_MAP["F8_E5M2"] = torch.float8_e5m2
    _DTYPE_NBYTES[torch.float8_e5m2] = 1


def _round_down(value: int, align: int) -> int:
    return value // align * align


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _tensor_nbytes(shape: list[int], dtype: torch.dtype) -> int:
    elements = 1
    for dim in shape:
        elements *= dim
    return elements * _DTYPE_NBYTES[dtype]


def _row_major_strides(shape: list[int]) -> list[int]:
    strides: list[int] = []
    current = 1
    for dim in reversed(shape):
        strides.append(current)
        current *= dim
    return list(reversed(strides))


def _normalize_slice_selection(
    shape: list[int],
    selection: tuple[slice | int, ...],
) -> tuple[int, int, list[int]]:
    """Return byte-independent element offset/count/shape for a contiguous slice.

    Only selections representable as one contiguous row-major range are accepted.
    This is deliberately conservative: unsupported slices fail closed instead of
    reading a full tensor and slicing it in CPU RAM.
    """

    if len(selection) != len(shape):
        raise ValueError(
            f"Slice rank mismatch: got {len(selection)} indices for shape {shape}"
        )

    strides = _row_major_strides(shape)
    first_offset = 0
    last_offset = 0
    output_shape: list[int] = []
    selected_elements = 1

    for dim, stride, item in zip(shape, strides, selection):
        if isinstance(item, bool):
            raise ValueError(f"Boolean indices are not supported: {selection!r}")
        if isinstance(item, int):
            index = item + dim if item < 0 else item
            if index < 0 or index >= dim:
                raise IndexError(
                    f"Index {item} is out of bounds for dimension of size {dim}"
                )
            first_offset += index * stride
            last_offset += index * stride
            continue
        if not isinstance(item, slice):
            raise TypeError(f"Unsupported slice item {item!r}")
        if item.step not in (None, 1):
            raise ValueError(f"Only contiguous step=1 slices are supported: {item!r}")
        start, stop, _step = item.indices(dim)
        length = max(0, stop - start)
        output_shape.append(length)
        selected_elements *= length
        if length == 0:
            continue
        first_offset += start * stride
        last_offset += (stop - 1) * stride

    if selected_elements == 0:
        return first_offset, 0, output_shape

    span_elements = last_offset - first_offset + 1
    if span_elements != selected_elements:
        raise ValueError(
            "Slice is not contiguous in row-major storage and would require "
            f"multiple reads: shape={shape}, selection={selection!r}"
        )
    return first_offset, selected_elements, output_shape


def _normalize_single_dim_slice_selection(
    shape: list[int],
    selection: tuple[slice | int, ...],
) -> tuple[int, int, int, int, list[int]] | None:
    """Return a strided row-major selection for one partially-sliced dimension.

    The return tuple is:
    ``(outer_count, source_dim, start, length, output_shape)``.
    Each outer item is one contiguous byte range of ``length * inner_count``
    elements.  This covers common input-dim TP shards such as RowParallelLinear
    weights without materializing the full checkpoint tensor.
    """

    if len(selection) != len(shape):
        raise ValueError(
            f"Slice rank mismatch: got {len(selection)} indices for shape {shape}"
        )

    partial_dim = None
    partial_start = 0
    partial_length = 0
    output_shape: list[int] = []

    for idx, (dim, item) in enumerate(zip(shape, selection)):
        if isinstance(item, bool):
            raise ValueError(f"Boolean indices are not supported: {selection!r}")
        if isinstance(item, int):
            return None
        if not isinstance(item, slice):
            raise TypeError(f"Unsupported slice item {item!r}")
        if item.step not in (None, 1):
            raise ValueError(f"Only contiguous step=1 slices are supported: {item!r}")
        start, stop, _step = item.indices(dim)
        length = max(0, stop - start)
        output_shape.append(length)
        is_full_dim = start == 0 and length == dim
        if is_full_dim:
            continue
        if partial_dim is not None:
            return None
        partial_dim = idx
        partial_start = start
        partial_length = length

    if partial_dim is None:
        return None
    if partial_length == 0:
        return None

    outer_count = math.prod(shape[:partial_dim])
    inner_count = math.prod(shape[partial_dim + 1 :])
    if outer_count <= 0 or inner_count <= 0:
        return None
    return (
        outer_count,
        shape[partial_dim],
        partial_start,
        partial_length,
        output_shape,
    )


@dataclass(frozen=True)
class TensorMeta:
    """Metadata-only view of one safetensors tensor payload."""

    file_path: str
    name: str
    dtype: torch.dtype
    shape: list[int]
    offset: int
    size: int


_TensorRecord = TensorMeta


@dataclass(frozen=True)
class WeightPlanReadSegment:
    source_slices: tuple[slice | int, ...]
    target_slices: tuple[slice | int, ...]


@dataclass(frozen=True)
class TransformOp:
    """Named, serializable tensor transform reference.

    ``op`` must name a transform registered via ``register_weight_transform``;
    ``args`` are static arguments resolved at plan-build time (never model or
    tensor references), so a plan containing ops can be serialized and diffed.
    Composition is an ordered tuple of ops on the plan entry.
    """

    op: str
    args: tuple[int | float | str | bool, ...] = ()


@dataclass(frozen=True)
class _RegisteredTransform:
    fn: Callable[..., torch.Tensor]
    extra_staging_factor: float


_TRANSFORM_REGISTRY: dict[str, _RegisteredTransform] = {}


def register_weight_transform(
    name: str,
    fn: Callable[..., torch.Tensor],
    *,
    extra_staging_factor: float,
) -> None:
    """Register a named weight transform op.

    ``extra_staging_factor`` declares the transient extra CPU staging the op
    allocates, as a multiple of its input payload bytes (0.0 for views and
    in-place ops, 1.0 for ops that materialize one same-sized output), so the
    planner can account for transform memory instead of trusting arbitrary
    code.  Re-registering the same function under the same name is a no-op;
    registering a different function under an existing name fails closed.
    """

    if not name:
        raise ValueError("Weight transform op name must be non-empty")
    if extra_staging_factor < 0:
        raise ValueError(
            f"Weight transform op {name!r} has negative staging factor"
        )
    existing = _TRANSFORM_REGISTRY.get(name)
    if existing is not None:
        if (
            existing.fn is fn
            and existing.extra_staging_factor == extra_staging_factor
        ):
            return
        raise RuntimeError(
            f"Weight transform op {name!r} is already registered with a "
            "different implementation"
        )
    _TRANSFORM_REGISTRY[name] = _RegisteredTransform(fn, extra_staging_factor)


def _resolve_transform_op(op: TransformOp) -> _RegisteredTransform:
    registered = _TRANSFORM_REGISTRY.get(op.op)
    if registered is None:
        raise RuntimeError(
            f"Unknown weight transform op {op.op!r}; transform ops must be "
            "registered before plan validation (fail closed)"
        )
    return registered


def apply_transform_ops(
    ops: tuple[TransformOp, ...],
    tensor: torch.Tensor,
) -> torch.Tensor:
    for op in ops:
        tensor = _resolve_transform_op(op).fn(tensor, *op.args)
    return tensor


def transform_ops_extra_staging_factor(ops: tuple[TransformOp, ...]) -> float:
    """Total transient staging multiple for one entry's op chain.

    Ops in a chain run serially and release their input after producing the
    output, so summing factors is a conservative upper bound.
    """

    return sum(_resolve_transform_op(op).extra_staging_factor for op in ops)


def _transform_zero_mean(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    return tensor - tensor.mean()


def _transform_squeeze(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return tensor.squeeze(dim)


def _transform_l2_normalize(
    tensor: torch.Tensor,
    dim: int = 0,
    eps: float = 1e-7,
) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor, dim=dim, p=2, eps=eps)


def _transform_qk_rope_permute(
    tensor: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    if n_heads <= 0:
        raise ValueError(f"qk_rope_permute requires n_heads > 0, got {n_heads}")
    original_ndim = tensor.ndim
    if original_ndim == 1:
        tensor = tensor.unsqueeze(-1)
    f_out, f_in = tensor.shape
    tensor = (
        tensor.view(n_heads, f_out // n_heads // 2, 2, f_in)
        .transpose(1, 2)
        .reshape(f_out, f_in)
    )
    if original_ndim == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def _transform_qk_rope_permute_2d(
    tensor: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    if n_heads <= 0:
        raise ValueError(f"qk_rope_permute_2d requires n_heads > 0, got {n_heads}")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    f_out, f_in = tensor.shape
    return (
        tensor.view(n_heads, f_out // n_heads // 2, 2, f_in)
        .transpose(1, 2)
        .reshape(f_out, f_in)
    )


def _transform_patch_embedding_reshape(
    tensor: torch.Tensor,
    patch_size: int,
    in_channels: int,
) -> torch.Tensor:
    if tensor.ndim != 2:
        return tensor
    out_channels = tensor.shape[0]
    in_features = tensor.shape[1]
    if in_features != in_channels * patch_size * patch_size:
        return tensor
    tensor = tensor.reshape(out_channels, patch_size, patch_size, in_channels)
    return tensor.permute(0, 3, 1, 2).contiguous()


register_weight_transform("zero_mean", _transform_zero_mean, extra_staging_factor=1.0)
register_weight_transform("squeeze", _transform_squeeze, extra_staging_factor=0.0)
register_weight_transform(
    "l2_normalize", _transform_l2_normalize, extra_staging_factor=1.0
)
register_weight_transform(
    "qk_rope_permute", _transform_qk_rope_permute, extra_staging_factor=1.0
)
register_weight_transform(
    "qk_rope_permute_2d", _transform_qk_rope_permute_2d, extra_staging_factor=1.0
)
register_weight_transform(
    "patch_embedding_reshape",
    _transform_patch_embedding_reshape,
    extra_staging_factor=1.0,
)


@dataclass(frozen=True)
class WeightPlanEntry:
    checkpoint_name: str
    target_name: str
    required: bool = True
    source_slices: tuple[slice | int, ...] | None = None
    target_slices: tuple[slice | int, ...] | None = None
    read_segments: tuple[WeightPlanReadSegment, ...] | None = None
    staging_shape: tuple[int, ...] | None = None
    transform_ops: tuple[TransformOp, ...] = ()
    source_is_sharded: bool = False
    read_into_cpu: bool = False
    shard_id: str | int | None = None
    expert_id: int | None = None
    weight_name: str | None = None
    loader_target_name: str | None = None
    ignore_missing: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class WeightPlan:
    entries: tuple[WeightPlanEntry, ...]

    def __iter__(self):
        return iter(self.entries)


@dataclass(frozen=True)
class WeightPlanSummary:
    entries: int
    required_entries: int
    skipped_entries: int
    missing_skipped_entries: int
    full_read_entries: int
    sliced_read_entries: int
    read_into_entries: int
    full_payload_bytes: int
    sliced_payload_bytes: int
    read_into_payload_bytes: int
    skipped_payload_bytes: int
    missing_skipped_payload_bytes: int
    peak_transform_staging_bytes: int = 0

    @property
    def total_read_payload_bytes(self) -> int:
        return (
            self.full_payload_bytes
            + self.sliced_payload_bytes
            + self.read_into_payload_bytes
        )


@dataclass(frozen=True)
class ReadScheduleSummary:
    entries: int
    required_entries: int
    read_ranges: int
    expected_direct_reads: int
    expected_window_loads: int
    expected_window_hits: int
    expected_bytes_read: int
    payload_bytes: int

    @property
    def read_amplification(self) -> float:
        if self.payload_bytes == 0:
            return 0.0
        return self.expected_bytes_read / self.payload_bytes


@dataclass(frozen=True)
class ReadSchedulePlan:
    plan: WeightPlan
    summary: ReadScheduleSummary


@dataclass(frozen=True)
class _PlanReadRange:
    file_path: str
    offset: int
    size: int
    repeat: int = 1
    stride: int = 0

    @property
    def first_offset(self) -> int:
        return self.offset

    @property
    def range_count(self) -> int:
        return self.repeat if self.size > 0 else 0

    @property
    def payload_bytes(self) -> int:
        return self.size * self.range_count


@dataclass(frozen=True)
class _ScheduledEntry:
    original_index: int
    entry: WeightPlanEntry
    ranges: tuple[_PlanReadRange, ...]

    @property
    def first_read_key(self) -> tuple[str, int, int]:
        if not self.ranges:
            return ("", 0, self.original_index)
        first = min(self.ranges, key=lambda item: (item.file_path, item.first_offset))
        return (first.file_path, first.first_offset, self.original_index)


class TensorCatalog:
    """Validated, metadata-only catalog of safetensors records."""

    def __init__(self, records: list[TensorMeta]) -> None:
        self._records = tuple(records)
        self._by_name = {record.name: record for record in records}
        if len(self._by_name) != len(records):
            raise RuntimeError("Duplicate tensor names in TensorCatalog")

    @classmethod
    def from_safetensors_files(
        cls,
        files: list[str],
        *,
        metadata_limit_bytes: int,
    ) -> "TensorCatalog":
        records: list[TensorMeta] = []
        seen_names: dict[str, str] = {}
        for path in files:
            file_size = os.path.getsize(path)
            with open(path, "rb", buffering=0) as f:
                raw_size = f.read(8)
                if len(raw_size) != 8:
                    raise RuntimeError(f"Invalid safetensors header in {path}")
                metadata_size = int.from_bytes(raw_size, "little")
                if metadata_size > metadata_limit_bytes:
                    raise RuntimeError(
                        f"Safetensors metadata too large in {path}: "
                        f"{metadata_size} bytes > {metadata_limit_bytes} bytes"
                    )
                if metadata_size > file_size - 8:
                    raise RuntimeError(
                        f"Invalid safetensors metadata size in {path}: "
                        f"{metadata_size} bytes exceeds file payload"
                    )
                metadata_raw = f.read(metadata_size)
                if len(metadata_raw) != metadata_size:
                    raise RuntimeError(f"Short safetensors metadata read in {path}")
            metadata = json.loads(metadata_raw)
            data_start = 8 + metadata_size
            file_ranges: list[tuple[int, int, str]] = []
            for name, info in metadata.items():
                if name == "__metadata__":
                    continue
                if name in seen_names:
                    raise RuntimeError(
                        f"Duplicate safetensors tensor name {name!r}: "
                        f"{seen_names[name]} and {path}"
                    )
                seen_names[name] = path
                if not isinstance(info, dict):
                    raise RuntimeError(f"Invalid safetensors metadata for {name}")
                try:
                    dtype_name = info["dtype"]
                    data_offsets = info["data_offsets"]
                    shape_raw = info["shape"]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Missing safetensors metadata key {exc.args[0]!r} for {name}"
                    ) from exc
                if dtype_name not in _DTYPE_MAP:
                    raise RuntimeError(
                        f"Unsupported safetensors dtype {dtype_name!r} for {name}"
                    )
                dtype = _DTYPE_MAP[dtype_name]
                if not isinstance(data_offsets, list) or len(data_offsets) != 2:
                    raise RuntimeError(
                        f"Invalid safetensors data_offsets for {name}: "
                        f"{data_offsets!r}"
                    )
                if not isinstance(shape_raw, list):
                    raise RuntimeError(
                        f"Invalid safetensors shape for {name}: {shape_raw!r}"
                    )
                start, end = data_offsets
                shape = list(shape_raw)
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end < start
                    or data_start + end > file_size
                ):
                    raise RuntimeError(
                        f"Invalid safetensors byte range for {name}: "
                        f"start={start}, end={end}, file_size={file_size}, "
                        f"data_start={data_start}"
                    )
                if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
                    raise RuntimeError(f"Invalid safetensors shape for {name}: {shape}")
                size = end - start
                expected_size = _tensor_nbytes(shape, dtype)
                if expected_size != size:
                    raise RuntimeError(
                        f"Tensor size mismatch for {name}: metadata has {size} "
                        f"bytes, shape/dtype imply {expected_size} bytes"
                    )
                file_ranges.append((start, end, name))
                records.append(
                    TensorMeta(
                        file_path=path,
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        offset=data_start + start,
                        size=size,
                    )
                )
            file_ranges.sort(key=lambda item: item[0])
            previous_end = 0
            previous_name = ""
            for start, _end, name in file_ranges:
                if start < previous_end:
                    raise RuntimeError(
                        f"Overlapping safetensors data ranges in {path}: "
                        f"{previous_name} ends at {previous_end}, "
                        f"{name} starts at {start}"
                    )
                previous_end = _end
                previous_name = name
        records.sort(key=lambda r: (r.file_path, r.offset))
        return cls(records)

    def records(self) -> tuple[TensorMeta, ...]:
        return self._records

    def names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self._records)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> TensorMeta:
        return self._by_name[name]

    def numel(self, name: str) -> int:
        elements = 1
        for dim in self.get(name).shape:
            elements *= dim
        return elements

    def total_bytes(self) -> int:
        return sum(record.size for record in self._records)


class WeightPlanBuilder(Protocol):
    """Model-side contract for constructing a metadata-only placement plan."""

    def build_weight_plan(self, catalog: TensorCatalog) -> WeightPlan:
        ...


class WeightPlanExecutor(Protocol):
    """Model-side contract for executing a plan through a weight source."""

    def load_weights_from_source(
        self,
        source: object,
        plan: WeightPlan,
    ) -> set[str]:
        ...


class WeightPlanSourceModel(WeightPlanBuilder, WeightPlanExecutor, Protocol):
    """Model that supports the planner/executor WeightSource path."""


@dataclass(frozen=True)
class ExecutorCapability:
    """Executor features used to validate a plan before scheduling reads."""

    supports_partial_read: bool
    supports_strided_read: bool
    requires_alignment: bool
    allows_mmap: bool
    supports_full_tensor_fallback: bool
    fail_closed: bool
    max_staging_bytes: int | None = None

    @classmethod
    def uma_odirect(
        cls,
        *,
        max_staging_bytes: int | None = None,
        supports_strided_read: bool = True,
    ) -> "ExecutorCapability":
        return cls(
            supports_partial_read=True,
            supports_strided_read=supports_strided_read,
            requires_alignment=True,
            allows_mmap=False,
            supports_full_tensor_fallback=False,
            fail_closed=True,
            max_staging_bytes=max_staging_bytes,
        )


WeightPlanBuildFn = Callable[[TensorCatalog], WeightPlan]
WeightPlanExecuteFn = Callable[[object, WeightPlan], set[str]]


def resolve_weight_plan_source_hooks(
    model: object,
) -> tuple[WeightPlanBuildFn, WeightPlanExecuteFn] | None:
    """Return model-side planner/executor hooks, or fail on a partial contract."""

    build_weight_plan = getattr(model, "build_weight_plan", None)
    load_weights_from_source = getattr(model, "load_weights_from_source", None)
    if not callable(build_weight_plan) and not callable(load_weights_from_source):
        return None
    if not callable(build_weight_plan) or not callable(load_weights_from_source):
        raise RuntimeError(
            "Models using WeightSource loading must implement both "
            "build_weight_plan(catalog) and load_weights_from_source(source, plan)"
        )
    return build_weight_plan, load_weights_from_source


def _resolve_attr(root: object, path: str) -> object:
    current = root
    for part in path.split("."):
        if isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
            continue
        if not hasattr(current, part):
            raise RuntimeError(f"Cannot resolve weight plan target {path!r}")
        current = getattr(current, part)
    return current


def _infer_output_dim_source_slice(
    param: object,
    record: TensorMeta,
    *,
    shard_id: int | str | tuple[int, ...] | None = None,
    weight_loader: Callable | None = None,
) -> tuple[slice | int, ...] | None:
    """Infer a safe source-side TP slice for tensor-parallel params.

    vLLM fused loaders (QKV/MergedColumn) normally receive a full checkpoint
    shard and narrow it to this rank. If we can prove the source tensor is
    exactly ``tp_size`` copies of the local shard along output dim, we can read
    only this rank's source rows and temporarily mark the input as already
    sharded before delegating to the existing weight_loader.  RowParallel-like
    input-dim shards use the same proof but may require strided source reads.
    """
    if getattr(param, "is_sharded_weight", False):
        return None
    if getattr(param, "use_bitsandbytes_4bit", False):
        return None

    param_data = getattr(param, "data", None)
    if param_data is None:
        return None
    param_shape = list(param_data.shape)
    if len(record.shape) != len(param_shape) or not param_shape:
        return None

    tp_rank = getattr(param, "tp_rank", None)
    tp_size = getattr(param, "tp_size", None)
    if not isinstance(tp_rank, int) or not isinstance(tp_size, int):
        return None
    if tp_size <= 1 or tp_rank < 0 or tp_rank >= tp_size:
        return None

    output_dim = getattr(param, "output_dim", None)
    if output_dim == 0:
        if getattr(param, "packed_dim", None) != output_dim:
            shard_size = _infer_output_dim_local_shard_size(
                param,
                record,
                shard_id=shard_id,
                weight_loader=weight_loader,
                param_output_size=param_shape[output_dim],
            )
            output_slice = _infer_dim_tp_source_slice(
                record,
                param_shape,
                output_dim,
                shard_size,
                tp_rank,
                tp_size,
            )
            if output_slice is not None:
                return output_slice

    input_dim = getattr(param, "input_dim", None)
    if shard_id is None and isinstance(input_dim, int):
        if getattr(param, "packed_dim", None) != input_dim:
            normalized_input_dim = input_dim
            if normalized_input_dim < 0:
                normalized_input_dim += len(param_shape)
            input_slice = _infer_dim_tp_source_slice(
                record,
                param_shape,
                normalized_input_dim,
                (
                    param_shape[normalized_input_dim]
                    if 0 <= normalized_input_dim < len(param_shape)
                    else None
                ),
                tp_rank,
                tp_size,
            )
            if input_slice is not None:
                return input_slice

    return None


def _infer_dim_tp_source_slice(
    record: TensorMeta,
    param_shape: list[int],
    dim: int,
    shard_size: int | None,
    tp_rank: int,
    tp_size: int,
) -> tuple[slice | int, ...] | None:
    if dim < 0:
        dim += len(param_shape)
    if dim < 0 or dim >= len(param_shape):
        return None
    if shard_size is None or shard_size <= 0:
        return None
    if record.shape[dim] != shard_size * tp_size:
        return None
    for idx, (source_size, target_size) in enumerate(zip(record.shape, param_shape)):
        if idx == dim:
            continue
        if source_size != target_size:
            return None

    start = tp_rank * shard_size
    slices: list[slice | int] = [slice(None)] * len(param_shape)
    slices[dim] = slice(start, start + shard_size)
    return tuple(slices)


def _infer_output_dim_local_shard_size(
    param: object,
    record: TensorMeta,
    *,
    shard_id: int | str | tuple[int, ...] | None,
    weight_loader: Callable | None,
    param_output_size: int,
) -> int | None:
    if shard_id is None:
        return param_output_size

    if isinstance(shard_id, tuple):
        return None

    owner = getattr(weight_loader, "__self__", None)
    if owner is None:
        return None

    get_size = getattr(owner, "_get_shard_size_mapping", None)
    if callable(get_size):
        try:
            shard_size = get_size(shard_id)
        except Exception:
            shard_size = None
        if isinstance(shard_size, int) and shard_size > 0:
            if shard_size <= param_output_size:
                return shard_size
            return None

    if isinstance(shard_id, int):
        output_sizes = getattr(owner, "output_sizes", None)
        if (
            isinstance(output_sizes, (list, tuple))
            and 0 <= shard_id < len(output_sizes)
        ):
            total_size = output_sizes[shard_id]
            tp_size = getattr(param, "tp_size", None)
            if isinstance(total_size, int) and isinstance(tp_size, int):
                if tp_size > 1 and total_size > 0 and total_size % tp_size == 0:
                    shard_size = total_size // tp_size
                    if shard_size <= param_output_size:
                        return shard_size

    return None


def resolve_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    plan: WeightPlan,
) -> WeightPlan:
    """Resolve implicit placement decisions before any payload read.

    Fills in the tensor-parallel source slices that executors used to infer at
    execution time, so entries arrive fully resolved, the plan summary matches
    executed reads byte-for-byte, and executors never perform semantic
    inference.  Entries that are skipped, already sliced, segmented, routed to
    an expert loader, or missing from the catalog are left untouched
    (`summarize_weight_plan` raises the canonical error for missing required
    tensors).
    """

    resolved: list[WeightPlanEntry] = []
    changed = False
    for entry in plan:
        if (
            not entry.required
            or entry.source_slices is not None
            or entry.read_segments is not None
            or entry.expert_id is not None
            or not catalog.has(entry.checkpoint_name)
        ):
            resolved.append(entry)
            continue
        try:
            param = _resolve_attr(model, entry.target_name)
        except RuntimeError:
            if entry.ignore_missing:
                resolved.append(entry)
                continue
            raise
        source_slices = _infer_output_dim_source_slice(
            param,
            catalog.get(entry.checkpoint_name),
            shard_id=entry.shard_id,
            weight_loader=getattr(param, "weight_loader", None),
        )
        if source_slices is None:
            resolved.append(entry)
            continue
        changed = True
        resolved.append(
            replace(entry, source_slices=source_slices, source_is_sharded=True)
        )
    if not changed:
        return plan
    return WeightPlan(tuple(resolved))


def verify_loaded_weights(model: nn.Module, loaded_weights: set[str]) -> None:
    """Fail closed when loading left model parameters uninitialized.

    Minimal early version of the ResolvedWeightBinding completeness check.
    Mirrors the exemptions of the default loader's weight tracking: modules
    whose quant method materializes or rewrites weights after loading may have
    parameters that legitimately never appear in a checkpoint.
    """

    loaded = set(loaded_weights)
    for module_name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        has_online_quant = getattr(quant_method, "uses_meta_device", False)
        has_postprocess_quant = getattr(
            quant_method, "process_weights_after_loading", None
        )
        if has_online_quant or has_postprocess_quant:
            for param_name, _ in module.named_parameters():
                full_name = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )
                loaded.add(full_name)
    weights_not_loaded = {name for name, _ in model.named_parameters()} - loaded
    if weights_not_loaded:
        preview = ", ".join(sorted(weights_not_loaded)[:8])
        suffix = ", ..." if len(weights_not_loaded) > 8 else ""
        raise RuntimeError(
            f"Weight loading left {len(weights_not_loaded)} parameters "
            f"uninitialized (fail closed): {preview}{suffix}"
        )


_ROTARY_EMBEDS_UNUSED_WEIGHTS = (
    "rotary_pos_emb.inv_freq",
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
)


def _weight_plan_entry_payload_size(
    record: TensorMeta,
    source_slices: tuple[slice | int, ...] | None,
) -> int:
    if source_slices is None:
        return record.size
    try:
        _element_offset, element_count, _output_shape = _normalize_slice_selection(
            record.shape,
            source_slices,
        )
    except ValueError as exc:
        strided = _normalize_single_dim_slice_selection(record.shape, source_slices)
        if strided is None:
            raise exc
        element_count = math.prod(strided[4])
    return element_count * _DTYPE_NBYTES[record.dtype]


def _weight_plan_entry_read_ranges(
    record: TensorMeta,
    source_slices: tuple[slice | int, ...] | None,
) -> tuple[_PlanReadRange, ...]:
    if source_slices is None:
        return (_PlanReadRange(record.file_path, record.offset, record.size),)
    try:
        element_offset, element_count, _output_shape = _normalize_slice_selection(
            record.shape,
            source_slices,
        )
    except ValueError as exc:
        strided = _normalize_single_dim_slice_selection(record.shape, source_slices)
        if strided is None:
            raise exc
        outer_count, source_dim, start, length, output_shape = strided
        partial_dim = next(
            idx
            for idx, (src, out) in enumerate(zip(record.shape, output_shape))
            if src != out
        )
        inner_count = math.prod(record.shape[partial_dim + 1 :])
        segment_elements = length * inner_count
        element_size = _DTYPE_NBYTES[record.dtype]
        segment_bytes = segment_elements * element_size
        stride = source_dim * inner_count * element_size
        offset = record.offset + start * inner_count * element_size
        return (
            _PlanReadRange(
                record.file_path,
                offset,
                segment_bytes,
                repeat=outer_count,
                stride=stride,
            ),
        )
    element_size = _DTYPE_NBYTES[record.dtype]
    return (
        _PlanReadRange(
            record.file_path,
            record.offset + element_offset * element_size,
            element_count * element_size,
        ),
    )


def _weight_plan_entry_target_shape(
    record: TensorMeta,
    source_slices: tuple[slice | int, ...] | None,
) -> tuple[int, ...]:
    if source_slices is None:
        return tuple(record.shape)
    try:
        _element_offset, _element_count, output_shape = _normalize_slice_selection(
            record.shape,
            source_slices,
        )
        return tuple(output_shape)
    except ValueError as exc:
        strided = _normalize_single_dim_slice_selection(record.shape, source_slices)
        if strided is None:
            raise exc
        return tuple(strided[4])


def summarize_weight_plan(
    catalog: TensorCatalog,
    plan: WeightPlan,
) -> WeightPlanSummary:
    """Summarize planned payload reads from metadata only."""

    required_entries = 0
    skipped_entries = 0
    missing_skipped_entries = 0
    full_read_entries = 0
    sliced_read_entries = 0
    read_into_entries = 0
    full_payload_bytes = 0
    sliced_payload_bytes = 0
    read_into_payload_bytes = 0
    skipped_payload_bytes = 0
    missing_skipped_payload_bytes = 0
    peak_transform_staging_bytes = 0

    for entry in plan:
        has_record = catalog.has(entry.checkpoint_name)
        if not entry.required:
            skipped_entries += 1
            if has_record:
                skipped_payload_bytes += catalog.get(entry.checkpoint_name).size
            else:
                missing_skipped_entries += 1
            continue

        if not has_record:
            raise RuntimeError(
                f"Weight plan requires missing tensor {entry.checkpoint_name!r}"
            )

        required_entries += 1
        record = catalog.get(entry.checkpoint_name)
        if entry.read_segments is not None:
            if not entry.read_into_cpu:
                raise RuntimeError(
                    "WeightPlanEntry.read_segments is only supported with "
                    f"read_into_cpu=True: {entry.checkpoint_name}"
                )
            if entry.source_slices is not None or entry.target_slices is not None:
                raise RuntimeError(
                    "WeightPlanEntry.read_segments cannot be combined with "
                    f"source_slices/target_slices: {entry.checkpoint_name}"
                )
            read_into_entries += 1
            payload_size = sum(
                _weight_plan_entry_payload_size(record, segment.source_slices)
                for segment in entry.read_segments
            )
            read_into_payload_bytes += payload_size
        else:
            payload_size = _weight_plan_entry_payload_size(record, entry.source_slices)
            if entry.read_into_cpu:
                read_into_entries += 1
                read_into_payload_bytes += payload_size
            elif entry.source_slices is not None:
                sliced_read_entries += 1
                sliced_payload_bytes += payload_size
            else:
                full_read_entries += 1
                full_payload_bytes += payload_size

        if entry.transform_ops:
            # Resolving ops here also fails closed on unregistered op names
            # before any payload byte is read.
            extra_factor = transform_ops_extra_staging_factor(entry.transform_ops)
            peak_transform_staging_bytes = max(
                peak_transform_staging_bytes,
                int(payload_size * extra_factor),
            )

    return WeightPlanSummary(
        entries=len(plan.entries),
        required_entries=required_entries,
        skipped_entries=skipped_entries,
        missing_skipped_entries=missing_skipped_entries,
        full_read_entries=full_read_entries,
        sliced_read_entries=sliced_read_entries,
        read_into_entries=read_into_entries,
        full_payload_bytes=full_payload_bytes,
        sliced_payload_bytes=sliced_payload_bytes,
        read_into_payload_bytes=read_into_payload_bytes,
        skipped_payload_bytes=skipped_payload_bytes,
        missing_skipped_payload_bytes=missing_skipped_payload_bytes,
        peak_transform_staging_bytes=peak_transform_staging_bytes,
    )


def _weight_plan_read_ranges(
    catalog: TensorCatalog,
    entry: WeightPlanEntry,
) -> tuple[_PlanReadRange, ...]:
    if not entry.required or not catalog.has(entry.checkpoint_name):
        return ()
    record = catalog.get(entry.checkpoint_name)
    if entry.read_segments is not None:
        ranges: list[_PlanReadRange] = []
        for segment in entry.read_segments:
            ranges.extend(
                _weight_plan_entry_read_ranges(
                    record,
                    segment.source_slices,
                )
            )
        return tuple(ranges)
    return _weight_plan_entry_read_ranges(record, entry.source_slices)


def _file_sizes_from_catalog(catalog: TensorCatalog) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for record in catalog.records():
        sizes[record.file_path] = max(
            sizes.get(record.file_path, 0),
            record.offset + record.size,
        )
    return sizes


def _simulate_odirect_reads(
    ranges: tuple[_PlanReadRange, ...],
    *,
    file_sizes: dict[str, int],
    chunk_size: int,
    window_size: int,
    alignment: int,
) -> tuple[int, int, int, int]:
    chunk_size = max(_round_up(chunk_size, alignment), alignment)
    window_size = max(_round_up(window_size, alignment), chunk_size)
    current_path: str | None = None
    window_start = 0
    window_valid = 0
    direct_reads = 0
    window_loads = 0
    window_hits = 0
    bytes_read = 0

    def simulate_windowed_range(
        path: str,
        offset: int,
        size: int,
        repeat: int,
        stride: int,
    ) -> None:
        nonlocal current_path
        nonlocal window_start
        nonlocal window_valid
        nonlocal direct_reads
        nonlocal window_loads
        nonlocal window_hits
        nonlocal bytes_read

        if repeat <= 0 or size == 0:
            return
        if stride == 0:
            stride = size
        if path != current_path:
            current_path = path
            window_start = 0
            window_valid = 0
        file_size = file_sizes[path]
        index = 0
        while index < repeat:
            range_offset = offset + index * stride
            window_end = window_start + window_valid
            if range_offset < window_start or range_offset + size > window_end:
                read_start = _round_down(range_offset, alignment)
                required_size = _round_up(
                    (range_offset - read_start) + size,
                    alignment,
                )
                read_size = max(window_size, required_size)
                got = min(read_size, max(file_size - read_start, 0))
                direct_reads += 1
                window_loads += 1
                bytes_read += got
                window_start = read_start
                window_valid = got
                window_end = window_start + window_valid

            max_covered_index = (window_end - size - offset) // stride
            covered = max(1, min(repeat, max_covered_index + 1) - index)
            window_hits += covered
            index += covered

    def simulate_direct_range(
        path: str,
        offset: int,
        size: int,
        repeat: int,
        stride: int,
    ) -> None:
        nonlocal current_path
        nonlocal window_start
        nonlocal window_valid
        nonlocal direct_reads
        nonlocal bytes_read

        if repeat <= 0 or size == 0:
            return
        if stride == 0:
            stride = size
        if path != current_path:
            current_path = path
            window_start = 0
            window_valid = 0
        file_size = file_sizes[path]
        for index in range(repeat):
            offset_i = offset + index * stride
            chunks = _round_up(size, chunk_size) // chunk_size
            if offset_i % alignment == 0 and size % alignment == 0:
                direct_reads += chunks
                bytes_read += size
                continue
            copied = 0
            while copied < size:
                wanted_offset = offset_i + copied
                wanted_size = min(size - copied, chunk_size)
                read_start = _round_down(wanted_offset, alignment)
                read_end = _round_up(wanted_offset + wanted_size, alignment)
                read_size = read_end - read_start
                got = min(read_size, max(file_size - read_start, 0))
                direct_reads += 1
                bytes_read += got
                available_start = read_start
                available_end = read_start + got
                copy_start = max(wanted_offset, available_start)
                copy_end = min(wanted_offset + wanted_size, available_end)
                if copy_end <= copy_start:
                    break
                copied = (copy_start - offset_i) + (copy_end - copy_start)

    for read_range in ranges:
        if read_range.size == 0:
            continue
        if read_range.size <= window_size:
            simulate_windowed_range(
                read_range.file_path,
                read_range.offset,
                read_range.size,
                read_range.repeat,
                read_range.stride,
            )
        else:
            simulate_direct_range(
                read_range.file_path,
                read_range.offset,
                read_range.size,
                read_range.repeat,
                read_range.stride,
            )

    return direct_reads, window_loads, window_hits, bytes_read


def schedule_weight_plan_reads(
    catalog: TensorCatalog,
    plan: WeightPlan,
    *,
    chunk_size: int,
    window_size: int,
    alignment: int,
) -> ReadSchedulePlan:
    """Order required plan entries by file offset and estimate O_DIRECT reads."""

    required: list[_ScheduledEntry] = []
    skipped: list[WeightPlanEntry] = []
    for index, entry in enumerate(plan.entries):
        if entry.required:
            required.append(
                _ScheduledEntry(
                    original_index=index,
                    entry=entry,
                    ranges=_weight_plan_read_ranges(catalog, entry),
                )
            )
        else:
            skipped.append(entry)

    scheduled_entries = sorted(required, key=lambda item: item.first_read_key)
    scheduled_required = [item.entry for item in scheduled_entries]
    scheduled_plan = WeightPlan(tuple([*scheduled_required, *skipped]))
    ranges: list[_PlanReadRange] = []
    payload_bytes = 0
    read_ranges = 0
    for scheduled_entry in scheduled_entries:
        ranges.extend(scheduled_entry.ranges)
        read_ranges += sum(item.range_count for item in scheduled_entry.ranges)
        payload_bytes += sum(item.payload_bytes for item in scheduled_entry.ranges)
    direct_reads, window_loads, window_hits, bytes_read = _simulate_odirect_reads(
        tuple(ranges),
        file_sizes=_file_sizes_from_catalog(catalog),
        chunk_size=chunk_size,
        window_size=window_size,
        alignment=alignment,
    )
    return ReadSchedulePlan(
        plan=scheduled_plan,
        summary=ReadScheduleSummary(
            entries=len(scheduled_plan.entries),
            required_entries=len(scheduled_required),
            read_ranges=read_ranges,
            expected_direct_reads=direct_reads,
            expected_window_loads=window_loads,
            expected_window_hits=window_hits,
            expected_bytes_read=bytes_read,
            payload_bytes=payload_bytes,
        ),
    )


def build_auto_weight_plan_from_catalog(
    catalog: TensorCatalog,
    *,
    mapper: object | None = None,
    name_transform: (
        Callable[
            [str],
            tuple[str, tuple[TransformOp, ...] | None]
            | None,
        ]
        | None
    ) = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
    skip_predicate: Callable[[str], bool] | None = None,
    ignore_unexpected_suffixes: list[str] | None = None,
) -> WeightPlan:
    """Build a trivial name-mapped compatibility plan.

    Keep this helper narrow.  It is intended for exact-name, no-fusion,
    no-quant, no-expert-routing cases plus explicit skip/name-mapping rules.
    More complex model semantics should be declared by model/module hooks and
    resolved into an explicit rank-local plan instead of being inferred here.
    """

    prefixes = skip_prefixes or []
    substrs = [*(skip_substrs or []), *_ROTARY_EMBEDS_UNUSED_WEIGHTS]
    ignored_suffixes = ignore_unexpected_suffixes or []
    map_name_with_shard = getattr(mapper, "_map_name_with_shard", None)
    entries: list[WeightPlanEntry] = []
    for checkpoint_name in catalog.names():
        name = checkpoint_name
        transform_ops: tuple[TransformOp, ...] = ()
        if name_transform is not None:
            transformed = name_transform(checkpoint_name)
            if transformed is None:
                entries.append(
                    WeightPlanEntry(
                        checkpoint_name=checkpoint_name,
                        target_name=checkpoint_name,
                        required=False,
                    )
                )
                continue
            name, resolved_transform_ops = transformed
            if resolved_transform_ops is not None:
                transform_ops = resolved_transform_ops
        if skip_predicate is not None and skip_predicate(name):
            entries.append(
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=name,
                    required=False,
                    transform_ops=transform_ops,
                )
            )
            continue
        if any(name.startswith(prefix) for prefix in prefixes) or any(
            substr in name for substr in substrs
        ):
            entries.append(
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=name,
                    required=False,
                    transform_ops=transform_ops,
                )
            )
            continue

        target_name = name
        shard_id = None
        if callable(map_name_with_shard):
            mapped = map_name_with_shard(name)
            if mapped is None:
                entries.append(
                    WeightPlanEntry(
                        checkpoint_name=checkpoint_name,
                        target_name=name,
                        required=False,
                        transform_ops=transform_ops,
                    )
                )
                continue
            target_name, shard_id = mapped

        entries.append(
            WeightPlanEntry(
                checkpoint_name=checkpoint_name,
                target_name=target_name,
                transform_ops=transform_ops,
                shard_id=shard_id,
                ignore_missing=any(
                    target_name.endswith(suffix) for suffix in ignored_suffixes
                ),
            )
        )
    return WeightPlan(tuple(entries))


def build_auto_weight_plan_for_module(
    module: nn.Module,
    catalog: TensorCatalog,
    *,
    mapper: object | None = None,
    name_transform: (
        Callable[
            [str], tuple[str, tuple[TransformOp, ...] | None] | None
        ]
        | None
    ) = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
    skip_predicate: Callable[[str], bool] | None = None,
    ignore_unexpected_suffixes: list[str] | None = None,
) -> WeightPlan:
    """Build an AutoWeightsLoader-like plan for a real vLLM module."""

    merged_ignore_unexpected_suffixes = [".bias", *(ignore_unexpected_suffixes or [])]
    modules = (module, *module.children())
    iterator = (m.quant_config for m in modules if hasattr(m, "quant_config"))
    if quant_config := next(iterator, None):
        cache_scale_mapper = quant_config.get_cache_scale_mapper()
        if cache_scale_mapper is not None:
            mapper = (
                mapper | cache_scale_mapper
                if mapper is not None
                else cache_scale_mapper
            )
        merged_ignore_unexpected_suffixes.extend(
            quant_config._ignore_unexpected_suffixes
        )

    return build_auto_weight_plan_from_catalog(
        catalog,
        mapper=mapper,
        name_transform=name_transform,
        skip_prefixes=skip_prefixes,
        skip_substrs=skip_substrs,
        skip_predicate=skip_predicate,
        ignore_unexpected_suffixes=merged_ignore_unexpected_suffixes,
    )
