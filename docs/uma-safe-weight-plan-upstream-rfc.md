# RFC Draft: WeightPlan IR for Explicit Model Weight Loading

Status: draft for upstream discussion

## Summary

This RFC proposes a small, serializable `WeightPlan` IR that separates model
weight-loading semantics from loader execution policy.

Today, loader implementations often rediscover model semantics while iterating
checkpoint tensors: stacked projections, tensor-parallel slices, routed MoE
expert ownership, fused tensors, optional weights, and per-family transforms.
That works when there is one standard loader path, but it becomes costly and
fragile when adding new execution policies such as direct I/O, range scheduling,
streaming, or memory-constrained loading.

The proposed split is:

- model code builds a `WeightPlan` from checkpoint metadata and module state;
- loader executors consume the same plan using their own read/application
  strategy;
- model-family quirks are represented as data: entries, read segments, shard
  metadata, named transforms, and skip reasons.

The vendor branch used for this RFC validates the design with a UMA-safe
O_DIRECT executor, but the IR itself has no O_DIRECT dependency.

## Motivation

### Loader-specific semantic dispatch does not scale

The current fork started as a UMA-safe safetensors loader for DGX Spark, where
CPU RAM and VRAM share the same physical memory pool.  In that environment,
normal eager reads, mmap/page-cache growth, or broad staging buffers can cause
memory PSI stalls or swap even when the model nominally fits.

The first direct-I/O loader worked, but it exposed a design problem: once the
loader bypasses the ordinary checkpoint iterator, it must still know exactly
how each model expects checkpoint tensors to be mapped into parameters.  That
knowledge was spread across model-specific hooks:

- checkpoint name parsing for routed experts;
- gate/up and q/k/v projection fusion;
- local versus non-local expert filtering;
- source slices for fused tensors;
- tensor transforms such as rotary QK permutes or patch reshapes;
- optional or legitimately missing checkpoint tensors.

If each new loader format reimplements these decisions, vLLM grows a new model
semantic dispatcher per load format.

### Memory-safe loading needs a pre-execution plan

For memory-constrained execution, the loader needs a metadata-only view before
payload reads start:

- how many bytes will be read;
- how much staging memory transforms may require;
- whether reads are contiguous, sliced, or segmented;
- whether skipped tensors are truly optional/non-local or missing by mistake;
- whether the planned read order causes avoidable amplification.

An explicit `WeightPlan` makes those properties observable and testable before
any tensor payload is loaded.

## Goals

- Represent model weight-loading semantics in a loader-neutral plan.
- Keep the plan serializable enough for golden tests and reviewable diffs.
- Support dense, tensor-parallel, stacked, fused, and routed-MoE cases.
- Let different loader executors share one model plan.
- Preserve fail-closed validation for missing tensors, invalid slices, unknown
  transforms, unsupported segment shapes, and unsafe filesystem metadata.
- Provide stable read-amplification accounting for memory-sensitive loaders.

## Non-Goals

- This RFC does not require every upstream model to migrate at once.
- It does not require all loaders to implement direct I/O or range scheduling.
- It does not propose a general graph IR for arbitrary tensor rewrites.
- It does not make model loading lazy at inference time.
- It does not replace quantization methods or parameter `weight_loader`
  conventions.  The plan records the data needed to call them consistently.

## Proposed IR

### TensorCatalog

`TensorCatalog` is a metadata-only view of checkpoint tensors:

```python
@dataclass(frozen=True)
class TensorMeta:
    file_path: str
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int
    size: int
```

It is responsible for checkpoint metadata validation: duplicate tensor names,
record bounds, overlaps, dtype support, path regularity, and path safety.  Plan
builders consume the catalog without reading tensor payloads.

### WeightPlan

```python
@dataclass(frozen=True)
class WeightPlan:
    entries: tuple[WeightPlanEntry, ...]
```

A plan is the complete model-side loading contract for one rank.  It contains
required entries, optional/skipped entries, source slices, read segments,
transform operations, and metadata needed by the target parameter loader.

### WeightPlanEntry

Conceptually, an entry says:

> read this checkpoint tensor or tensor region, optionally transform it, then
> apply it to this model target with these loader arguments.

Important fields:

```python
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
```

This supports common upstream loading patterns:

- dense one-to-one tensor loads;
- tensor-parallel source slices;
- fused q/k/v or gate/up stacked projections with shard IDs;
- local/non-local routed expert entries with `expert_id`;
- module-owned custom loaders via `loader_target_name`;
- optional tensors and explicit skip reasons.

Entry-level invariants:

- `source_slices` and `read_segments` are mutually exclusive;
- entries with `read_segments` must provide `staging_shape`;
- `target_slices` only applies to read-into/staged execution;
- `transform_ops` must contain registered, serializable ops;
- skipped entries should carry an explicit `skip_reason` or
  `ignore_missing=True`.

### Read Segments

`WeightPlanReadSegment` expresses "which source bytes land where" for cases
that cannot be represented as one contiguous row-major slice:

```python
@dataclass(frozen=True)
class WeightPlanReadSegment:
    source_slices: tuple[slice | int, ...]
    target_slices: tuple[slice | int, ...]
```

Segmented entries require an explicit `staging_shape`.  This keeps the executor
from silently full-reading a fused tensor just to slice it in CPU RAM.

Examples:

- interleaved fused qkv where q/k/v rows are gathered with stride;
- fused expert tensors where local expert ranges and projection halves are
  copied into a compact staging tensor;
- TP layouts where one source tensor contributes multiple disjoint ranges.

The responsibility split is intentionally narrow:

- `WeightPlanReadSegment` says which bytes to copy where;
- `TransformOp` says what element operation to apply after bytes are staged.

This avoids a broad reshape/reorder primitive whose memory and byte-accounting
semantics are harder to reason about.

### TransformOp

```python
@dataclass(frozen=True)
class TransformOp:
    op: str
    args: tuple[int | float | str | bool, ...] = ()
```

Transforms are named, registered, and serializable.  Each registered transform
declares an `extra_staging_factor`, which lets plan summaries account for
temporary memory before execution.

Examples from the prototype:

- `zero_mean`
- `squeeze(dim)`
- `l2_normalize(dim, eps)`
- `qk_rope_permute(n_heads)`
- `qk_rope_permute_2d(n_heads)`
- `patch_embedding_reshape(patch_size, in_channels)`
- `transpose_last_two`

Unknown transform names fail before payload reads start.

## Declarative Model Specs

The prototype moved 24 routed families onto shared declarations plus common IR.
The useful primitives were orthogonal and small:

- `RoutedExpertPattern`: parse layer, expert, projection, and suffix from
  conventional checkpoint names;
- `RoutedProjectionMap`: map source projection names to target parameter names
  and shard IDs;
- `NameRewrite`: ordered, anchored literal rewrites before pattern matching;
- `StackedProjectionMap`: data form of upstream-style stacked projection
  mappings such as q/k/v to qkv or gate/up to gate_up;
- `SliceRule`: produce one or more sliced or segmented `WeightPlanEntry`
  records from one checkpoint tensor.

Model builders can still contain builder-side conditionals for true runtime
model-layout choices.  The important property is that the emitted plan remains
concrete data.

## Executor Behavior

A loader executor consumes `WeightPlan` entries in plan order or in a
semantics-preserving schedule.  The O_DIRECT reference executor validates:

- required checkpoint tensors exist;
- skipped or ignored entries have explicit reasons;
- source slices are contiguous or segment-supported;
- segmented entries provide `staging_shape`;
- transform ops are registered;
- actual loaded targets match the plan's loaded-set expectation;
- actual direct bytes read do not drift beyond scheduled expectation.

Other executors can start simpler: they can execute the same plan using normal
safetensors reads, then add scheduling or direct placement later.

## Read Accounting

The prototype includes a metadata-only read scheduler.  It simulates planned
direct reads using file offsets, tensor ranges, read-window size, contiguous
slices, and closed-form strided segment ranges.  It reports:

- expected direct reads;
- expected window loads and hits;
- expected bytes read;
- payload bytes;
- expected read amplification.

After execution, source stats compare actual `bytes_read` against expected
bytes and warn when actual exceeds expected by more than 10 percent.

This matters because read amplification can regress without changing model
correctness.  In the prototype, a handle-lifetime bug caused each small tensor
entry to pay one full read window.  The worst tiny-MoE case went from
150.39 GiB read for 1.25 GiB of payload in 61.47 s to 1.25 GiB read in
0.77 s after handle reuse and scheduling made the desired property explicit
and testable.

## Prototype Validation

The current vendor branch validates the proposed split with:

- a neutral `weight_plan.py` module with no fork-only dependencies;
- a UMA-safe O_DIRECT executor as the reference memory-sensitive executor;
- golden plan snapshots for representative families;
- registry/unit coverage for transform equivalence, routed parsing, slice
  rules, stacked mappings, and segmented entries;
- 24 routed families migrated to declaration specs plus shared IR;
- no remaining non-`WeightPlan` composite routed execution contracts.

DGX Spark model-load smoke after Phase 3:

```text
Model                         Expected  Actual   Amplification  Load time
tiny-random-qwen3.5 MoE        0.01GiB  0.01GiB         1.00x      1.55s
PrimeIntellect tiny MoE        1.25GiB  1.25GiB         1.00x      1.49s
Qwen3.6 35B NVFP4             22.23GiB 22.23GiB         1.02x     25.19s
```

Here `Amplification` is actual bytes read divided by payload bytes needed, not
actual divided by expected.  For the 35B row, actual and expected both equal
22.23 GiB, while payload is 21.73 GiB.

All three runs matched scheduled bytes exactly, emitted no actual-vs-expected
amplification warning, kept swap at 0, and kept memory PSI at 0.  The 35B run
peaked at 39.95 GiB used+buff/cache with 82.53 GiB minimum local available
RAM.

Known validation gap: the existing three-model smoke set does not exercise a
real checkpoint that uses segmented read entries.  TeleChat2, HunYuan, or
Llama4 checkpoints should be added to the smoke matrix when available.  The
segment shapes are covered by registry and golden tests today.

End-to-end `ready` startup is intentionally not used as the loader success
criterion in this environment.  A post-load FlashInfer/CUDA JIT compile storm
can consume nearly all host RAM independent of the weight loader.  That is a
separate runtime/cache issue, not a loader IR result.

## Incremental Adoption Path

1. Land the neutral data types and validation helpers.
2. Add plan construction for the default safetensors path without changing
   execution behavior.
3. Migrate a small set of dense and routed families to produce golden
   `WeightPlan` snapshots.
4. Add a normal safetensors executor that consumes `WeightPlan`.
5. Add optional advanced executors: O_DIRECT, range-scheduled reads, streaming,
   or other memory-sensitive policies.
6. Gradually replace model-specific parser hooks with declarative spec helpers.
7. Require new model families to expose plan semantics once the helper set is
   stable.

This lets upstream review the IR separately from any UMA-specific executor.

## Relationship to Existing Mechanisms

`WeightPlan` is meant to reuse existing vLLM model-loading conventions, not
replace them wholesale.

- Existing parameter `weight_loader` methods remain the application boundary.
  A plan entry records the arguments needed to call them consistently:
  `weight_name`, `shard_id`, `expert_id`, and `source_is_sharded`.
- `WeightsMapper` and upstream `stacked_params_mapping` remain useful inputs to
  plan construction.  The prototype's `StackedProjectionMap` deliberately
  mirrors that shape and can bridge back to `WeightsMapper` while model
  families migrate.
- Auto/default loaders can continue to build one-to-one entries.  The plan only
  makes the mapping explicit before payload reads.
- The part this proposal replaces is loader-side rediscovery of model
  semantics: parsing checkpoint names, guessing stacked projections, deciding
  local expert ownership, or full-reading fused tensors because the loader does
  not know the intended slice/segment.

## Compatibility

The IR does not change parameter loader APIs.  It records enough metadata to
call existing loaders consistently:

- `weight_name`
- `shard_id`
- `expert_id`
- `source_is_sharded`
- `loader_target_name`

Existing models can keep their current `load_weights` path until they opt into
plan construction.  Existing loaders can ignore `WeightPlan` until an executor
is added for them.

## Open Questions

- Which module should own the neutral IR long-term:
  `model_executor/model_loader/weight_plan.py`, a new planning package, or a
  more general model-semantics module?
- How much `TensorCatalog` validation should be safetensors-specific versus
  generic checkpoint metadata?
- How should pre-sharded per-rank checkpoints represent duplicate tensor names?
- What is the minimum transform op vocabulary acceptable upstream?
- Should segment execution allow direct destination placement for CUDA tensors,
  or should CPU staging remain the only portable baseline?
- How should quantization methods declare auxiliary tensors and packed layouts?
- What golden snapshot format should be stable enough for CI without becoming
  too noisy across legitimate model changes?

## Proposed First Upstream PR Set

1. Add `TensorMeta`, `TensorCatalog`, `WeightPlan`, `WeightPlanEntry`,
   `WeightPlanReadSegment`, and `TransformOp` with unit tests.
2. Add metadata-only summary and read-accounting tests.
3. Add one or two model builders that produce plans but still execute through
   the existing loader path.
4. Add golden plan snapshots for those builders.
5. Add a simple executor for normal safetensors reads.

The UMA/O_DIRECT executor can remain out of tree until the neutral plan is
accepted.
