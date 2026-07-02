# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

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
class WeightPlanEntry:
    checkpoint_name: str
    target_name: str
    required: bool = True
    source_slices: tuple[slice | int, ...] | None = None
    target_slices: tuple[slice | int, ...] | None = None
    read_segments: tuple[WeightPlanReadSegment, ...] | None = None
    staging_shape: tuple[int, ...] | None = None
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    source_is_sharded: bool = False
    read_into_cpu: bool = False
    shard_id: str | int | None = None
    expert_id: int | None = None
    weight_name: str | None = None
    ignore_missing: bool = False


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

    @property
    def total_read_payload_bytes(self) -> int:
        return (
            self.full_payload_bytes
            + self.sliced_payload_bytes
            + self.read_into_payload_bytes
        )


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

    def total_bytes(self) -> int:
        return sum(record.size for record in self._records)


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
    )


def build_auto_weight_plan_from_catalog(
    catalog: TensorCatalog,
    *,
    mapper: object | None = None,
    name_transform: (
        Callable[
            [str], tuple[str, Callable[[torch.Tensor], torch.Tensor] | None] | None
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
        transform = None
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
            name, transform = transformed
        if skip_predicate is not None and skip_predicate(name):
            entries.append(
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=name,
                    required=False,
                    transform=transform,
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
                    transform=transform,
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
                        transform=transform,
                    )
                )
                continue
            target_name, shard_id = mapped

        entries.append(
            WeightPlanEntry(
                checkpoint_name=checkpoint_name,
                target_name=target_name,
                transform=transform,
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
            [str], tuple[str, Callable[[torch.Tensor], torch.Tensor] | None] | None
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
