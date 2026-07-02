# Weight Plan IR Roadmap for UMA-safe Loading

Date: 2026-07-03

Status: design note / roadmap

This document reframes the current `uma_odirect_safetensors` work as a
prototype for a more general vLLM weight-loading architecture.  The immediate
branch still exists to make loading safer on UMA systems such as DGX Spark, but
the longer-term goal is to avoid creating another model-loader side path that
accumulates model-specific rules.

## Summary

The current UMA loader is useful, but it is not the final architecture.

Current shape:

```text
model-specific *_uma.py helpers
  -> UMA-specific WeightPlan
  -> O_DIRECT safetensors executor
```

Target shape:

```text
model / module / parameter loading semantics
  -> rank-local PlacementPlan / WeightPlan IR
  -> executor: O_DIRECT safetensors | ordinary safetensors | sharded_state
             | Run:ai | InstantTensor | runtime weight swap
```

In the target shape, model-specific knowledge lives in declarative model-side
loading semantics, not in storage loaders.  Storage loaders execute a plan; they
do not infer whether a tensor is QKV, gate/up, a local expert, a quantization
scale, a tied embedding, or a pipeline-missing layer.

## Why the Current vLLM Shape Is Hard to Make UMA-safe

The legacy loading contract is roughly:

```text
storage loader -> Iterable[(checkpoint_name, full CPU tensor)]
               -> model.load_weights()
               -> model-specific placement / slicing / fusion / skipping
```

That order is convenient for compatibility, but it is backwards for UMA safety:

- the storage layer reads a full tensor before knowing whether this rank needs
  it;
- TP/PP/EP locality is discovered after CPU staging;
- fused checkpoint tensors may be read before only one shard is used;
- non-local experts can be materialized and then discarded;
- quantization side tensors and model-specific transforms are handled through
  ad hoc branches;
- every new load format risks reimplementing the same model-family knowledge.

This is not just a DGX Spark performance issue.  It is an architectural
pressure: if each new loader path has to rediscover model placement rules, each
loader path will grow its own mapping tables and special cases.

## Root Cause: Loader as Semantic Dispatcher

The deeper issue is not simply that an I/O decision point happens too late.  The
problem is that loader implementations can become semantic dispatchers.

An I/O loader should primarily decide:

- which file or object to read from;
- which byte range or tensor payload to read;
- which staging buffer or target device receives the bytes;
- which I/O strategy, scheduling policy, and safety gates apply;
- which execution order is safe for memory and throughput.

In practice, loader paths often accumulate responsibilities that belong to a
semantic planning layer:

- checkpoint name to vLLM internal name resolution;
- fused QKV or gate/up split and merge rules;
- tensor-parallel shard axis and shard offset selection;
- pipeline-rank missing layer handling;
- expert-parallel global expert to local expert routing;
- quantization payload, scale, zero, and side-state attachment;
- tied embedding and vocabulary padding policy;
- model-family specific exceptions;
- pre-sharded versus logical full-tensor checkpoint handling.

Those jobs are necessary, but they should not live in the same layer as file
I/O.  Once a loader owns semantic dispatch, every new load format risks
rebuilding the same semantic tables.  A new fast path then becomes a parallel
model-loading architecture rather than an executor for the same model plan.

This is why small operational fixes do not fully solve the problem.  Smaller
O_DIRECT windows, coalesced reads, or per-worker state-dict loading can reduce
symptoms, but they still operate downstream of a missing planning phase unless
the semantics are resolved before payload reads.

The intended ordering is:

```text
read checkpoint manifest/header only
build model weight schema
resolve checkpoint names to logical weights
resolve TP/PP/EP rank-local placement
resolve quantization and expert relationships
build LoadPlan / PlacementPlan
schedule byte ranges
execute I/O
```

In other words, tensor treatment must be decided before tensor payload is read.
For UMA systems this is a safety requirement.  For vLLM as a whole it is the
only way to prevent each load format from growing its own model-specific
dispatcher.

## Existing Improvement Vectors

Several upstream-facing ideas move in the same direction, but none is enough by
itself unless they share a common placement IR.

### Per-worker state dict loading

This moves loading closer to "each worker reads only what it needs".  That is
directionally correct, but it still needs a way to compute what each worker
needs.  If that computation is loader-specific, the same expert/fusion/shard
knowledge will be rebuilt inside the new path.

### InstantTensor-style load format

A dedicated high-throughput backend can improve I/O behavior, but a new load
format by itself is another side path.  Without a shared plan, it can become a
parallel implementation of the same model-specific decisions.

### RunTimeWeightLoader / runtime weight swap

Runtime replacement naturally wants a "model asks for what it needs" contract.
It should use the same plan IR as startup loading, otherwise runtime and startup
loading will diverge.

### `process_weights_after_loading`

This is useful for post-load model fixes, but it happens too late to prevent
unsafe full-tensor reads.  It should remain a post-processing hook, not the
primary place where source tensor locality is decided.

## Design Principle

The loader should provide I/O and safety policy.  The model should declare
loading semantics.  A planner should turn model semantics plus checkpoint
metadata into a rank-local plan.  An executor should perform reads and
placements.

```text
Checkpoint metadata
      +
Model loading semantics
      +
Parallel config / quant config / pipeline rank
      |
      v
PlacementPlan
      |
      v
Executor
```

This separates responsibilities:

- model code knows model semantics;
- planner knows how to resolve semantics against checkpoint metadata and rank
  state;
- executor knows how to safely read bytes and place tensors;
- load formats become executor implementations rather than independent loading
  architectures.

More explicitly, the design should split the current fat-loader behavior into
separate stages:

```text
CheckpointReader
  - reads manifests, headers, indexes, and tensor metadata
  - does not read tensor payload by default

ModelWeightSchema
  - declares expected logical weights, runtime parameter names, shapes, fusion
    groups, quant roles, and optional/tied weights

NameResolver
  - maps checkpoint names to logical weight IDs
  - handles architecture-specific naming differences without doing I/O

ShardingPlanner
  - maps logical weights to TP/PP/EP rank-local source and target slices

QuantPlanner
  - resolves weight/scale/zero/quant-state relationships and attachment points

ExpertPlanner
  - maps global experts to local experts and source ranges

LoadPlanner
  - combines the above into an executable PlacementPlan

ReadScheduler
  - groups nearby byte ranges, chooses read order, caps staging bytes, and
    controls reuse/release points

LoadExecutor
  - executes the schedule using an I/O strategy
  - does not know model-family semantics
```

The current prototype has pieces of this split, but not the full separation.
`TensorCatalog` is close to `CheckpointReader`, `WeightPlan` is close to
`PlacementPlan`, and `ODirectSafetensorsWeightSource` is close to a
LoadExecutor.  The missing parts are first-class semantic planning and a
model-independent read scheduler.

## Proposed IR Layers

### 1. TensorCatalog

Metadata-only checkpoint view.

Responsibilities:

- enumerate checkpoint tensors;
- validate safetensors metadata before payload reads;
- reject duplicate names, overlaps, malformed ranges, symlinked files, and
  unsupported dtypes;
- expose shape, dtype, file, and byte range.

This started as a prototype inside `uma_odirect_safetensors_loader.py` and is
now being extracted into neutral model-loader IR modules.

### 2. SemanticSpec / Weight Semantics

Model/module/parameter-side declaration of how checkpoint names map to runtime
parameters.

Examples:

```python
ColumnShard(source="layers.{i}.self_attn.q_proj.weight", dim=0)
RowShard(source="layers.{i}.mlp.down_proj.weight", dim=1)
FusedQKV(
    sources=("q_proj.weight", "k_proj.weight", "v_proj.weight"),
    target="qkv_proj.weight",
)
FusedGateUp(
    sources=("gate_proj.weight", "up_proj.weight"),
    target="gate_up_proj.weight",
)
RoutedExpert(
    source="experts.{expert}.gate_proj.weight",
    expert_id=expert,
    locality="rank-local",
)
PackedExperts(
    source="experts.gate_up_proj.weight",
    expert_axis=0,
    local_expert_range=(start, stop),
)
Transform(source="q_proj.weight", op="llama4_q_rope_permute")
Optional(source="lm_head.weight", reason="tied embedding")
```

The important property is not the exact class names.  The important property is
that these are inspectable declarations rather than arbitrary file I/O or
loader-specific Python branches.

At this layer, the declaration is still abstract.  It should not decide which
checkpoint tensor actually exists, which file offset will be read, or which
rank-local slice this worker owns.  Those decisions belong to later planning
stages.

### 3. ResolvedWeightBinding

Catalog-resolved binding between semantic declarations and checkpoint metadata.

It answers:

- which checkpoint tensors match each semantic source pattern;
- whether a tensor is missing, optional, tied, duplicate, or ambiguous;
- which legacy name mapping, if any, was used;
- which target runtime parameter each source group binds to;
- whether the binding is valid before rank-local placement is considered.

This prevents model-name resolution from leaking into storage executors.  A
storage executor should receive resolved records and byte ranges, not infer
whether `q_proj.weight`, `query.weight`, or a fused checkpoint tensor is the
right source for a model family.

### 4. RankLocalPlacementPlan / WeightPlan

Rank-local, executable IR.

It answers:

- which checkpoint byte ranges this worker must read;
- where the result goes;
- whether a source tensor is required, optional, skipped, or local-only;
- whether the read can be contiguous, segmented, strided, or must fail closed;
- whether a bounded CPU staging tensor is needed;
- which transform, if any, is applied before placement;
- how much payload will be read before execution starts.

The current `WeightPlan`, `WeightPlanEntry`, and `WeightPlanReadSegment` are a
good prototype.  `WeightPlan` is intentionally kept as the Phase 1 name to
avoid churn, but it should narrow toward `RankLocalPlacementPlan` semantics over
time.  Later phases should introduce aliases or replacements for clearer names
such as `SemanticWeightSpec`, `ResolvedWeightBinding`, `PlacementPlan`, and
`ReadSchedule`.

### 5. ExecutorCapability

Executor capability contract used to validate whether a placement plan can be
scheduled safely.

Examples:

```text
ODirectSafetensorsPlanExecutor:
  supports_partial_read = true
  supports_strided_read = false
  requires_alignment = true
  max_staging_bytes = configured
  allows_mmap = false
  fail_closed = true

OrdinarySafetensorsPlanExecutor:
  supports_partial_read = maybe
  supports_full_tensor_fallback = true
  allows_page_cache = true
  fail_closed = false
```

The planner should combine:

```text
PlacementPlan + ExecutorCapability
  -> ExecutableReadSchedule
```

This keeps unsupported slices, transforms, layouts, staging sizes, and fallback
policies out of ad hoc executor branches.  UMA-safe executors should reject
unsupported plans before payload reads.

### 6. PlanExecutor

Executes a `PlacementPlan`.

Possible executors:

- `ODirectSafetensorsPlanExecutor`
- `SafetensorsPlanExecutor`
- `RunaiPlanExecutor`
- `InstantTensorPlanExecutor`
- `ShardedStatePlanExecutor`
- `RuntimeWeightSwapPlanExecutor`

UMA-safe execution is one executor policy, not the only use of the plan.

UMA-specific executor requirements:

- fail closed on unsupported reads;
- avoid mmap/prefetch/eager fallback;
- gate allocation and read loops on `MemAvailable`, swap, and memory PSI;
- report `peak used`, `peak buff/cache`, `used + buff/cache`, and skipped
  bytes during tests;
- never silently fall back to full-tensor CPU staging.

Non-UMA executors may choose faster or more permissive behavior while using the
same plan.

### 7. ReadSchedulePlan

Executor-facing schedule derived from a `PlacementPlan`.

This is where small-tensor read amplification should be solved.  The executor
should not blindly process one logical tensor at a time if many required
tensors are adjacent in the same safetensors file.

Responsibilities:

- sort required payload ranges by file and offset when semantic ordering allows;
- coalesce nearby ranges while respecting a maximum staging-buffer size;
- preserve explicit ordering for entries with side effects or shared
  source-tensor reuse;
- expose expected read amplification before execution;
- gate before and after group reads;
- release staging buffers as soon as all group entries are dispatched.

This keeps range coalescing model-independent.  It also avoids treating
`window_size` tuning as the main solution.  Window tuning is an executor knob;
read scheduling is the architectural layer that prevents repeated large reads
for many small tensors.

Placement and scheduling should remain separate.  Placement describes what this
rank needs and where it goes; scheduling describes how an executor groups,
orders, gates, and releases concrete reads.

## What This Means for the Current Branch

The current branch is valid as an operational mitigation, but it should be
treated as a prototype with clear boundaries.

Keep doing:

- enforce fail-closed behavior for UMA-safe loads;
- keep O_DIRECT reads and metadata-first catalog validation;
- keep adding targeted model hooks when they unblock real testing;
- keep tests around source slicing, local expert selection, and skipped reads.

Avoid doing:

- adding storage-loader branches such as `if qwen`, `if llama4`, or
  `if deepseek`;
- treating `*_uma.py` files as the final stable model-loading interface;
- adding a new external loader path without making it consume the same plan;
- allowing hidden fallback from plan execution to the legacy full-tensor
  iterator in UMA-safe mode.
- growing `build_auto_weight_plan_*` into an implicit semantic dispatcher.
  Auto-plan helpers should remain limited to trivial exact-name compatibility
  cases; fused, quantized, expert-parallel, transformed, or non-local layouts
  should require explicit semantic specs or model hooks.

Near-term model hooks should be written as if they are future plan builders, not
as one-off loaders.  They should build entries, summarize read volume, and then
let the executor do all I/O.

## Migration Plan

### Phase 0: Stabilize the current UMA prototype

Goal: keep DGX Spark usable and prevent unsafe fallback.

Tasks:

- keep `uma_odirect_safetensors` fail-closed;
- maintain strict metadata validation;
- keep adding only the model hooks needed for actual tests;
- document unsupported cases explicitly;
- keep measuring `used + buff/cache`, not just allocated tensors.

Exit criteria:

- Qwen 35B and selected MoE models load without memory PSI;
- tests cover malformed metadata, duplicate names, unsafe slices, local experts,
  transforms, and plan summaries.

### Phase 1: Extract the IR from the O_DIRECT loader

Goal: make the plan representation independent from the O_DIRECT executor.

Tasks:

- move `TensorCatalog`, `WeightPlan`, `WeightPlanEntry`, and
  `WeightPlanReadSegment` into a neutral module, for example:
  `vllm/model_executor/model_loader/weight_plan.py`;
- keep the current class names initially to avoid churn;
- make `ODirectSafetensorsWeightSource` consume the neutral IR;
- add tests that build a plan without instantiating the O_DIRECT loader.

Exit criteria:

- the O_DIRECT loader imports the IR instead of owning it;
- model hooks import the neutral IR;
- tests can validate plan construction and plan accounting separately from
  Linux O_DIRECT reads.

### Phase 2: Split planner and executor contracts

Goal: make it clear which code creates a plan and which code executes it.

Tasks:

- define a `WeightPlanBuilder` protocol or model hook contract;
- define a `WeightPlanExecutor` protocol;
- rename or wrap `load_weights_from_source(source, plan)` into a plan executor
  call;
- keep legacy iterator loading as a compatibility executor/fallback outside
  UMA-safe mode.

Exit criteria:

- model code never calls raw file I/O;
- executor code never contains model-family mappings;
- plan summary can run before payload reads and before executor selection.

### Phase 3: Move from model-level hooks to module/parameter semantics

Goal: prevent `*_uma.py` files from becoming the new mapping-table pile.

Tasks:

- introduce small inspectable loading-spec objects for common patterns:
  column shard, row shard, fused qkv, fused gate/up, routed expert, packed
  expert, tied embedding, optional tensor, named transform;
- let modules expose these specs directly where possible;
- generate most `WeightPlanEntry` objects from specs rather than hand-written
  per-model loops;
- keep model-level plan builders only for genuinely model-global decisions.

Exit criteria:

- adding a conventional Transformer block should require mostly module spec
  declarations, not a new model-specific planner;
- common MoE layouts share spec helpers;
- QKV, gate/up, and routed expert handling are not reimplemented per load
  format.

### Phase 4: Share the plan across load formats

Goal: avoid a separate architecture per loader.

Tasks:

- add a non-O_DIRECT executor for the same plan;
- make sharded/per-worker loading consume the same plan;
- evaluate whether Run:ai or InstantTensor can execute plan entries or expose
  equivalent range/locality primitives;
- make runtime weight swap use the same model semantics when possible.

Exit criteria:

- at least two loader formats execute the same plan IR;
- model-side semantics do not change when switching executor;
- UMA-safe mode is a strict executor policy, not a separate model-loading
  knowledge graph.

## Open Design Questions

- What is the smallest set of first-class spec objects that covers most dense,
  fused, and routed-MoE models?
- Which transforms must become named IR operations instead of Python callables?
- How should quantization methods attach auxiliary tensors, scales, and
  packed layouts to the plan?
- Can direct destination placement be expressed safely for CUDA tensors, or
  should CPU segmented staging remain the only safe portable primitive?
- How should error messages expose "unsupported by executor" versus "invalid
  model plan" versus "missing checkpoint tensor"?
- How much of this can be proposed upstream without requiring every model to
  migrate at once?

## Practical Next Steps

For this branch, the next useful work is:

1. keep current UMA loader working and tested;
2. avoid adding new model-specific logic to the storage loader;
3. extract the current plan dataclasses into a neutral module;
4. update existing model hooks to import the neutral plan types;
5. add a short design note in each future model hook explaining which generic
   spec pattern it should eventually become.

This lets the branch keep solving the immediate UMA safety problem while moving
toward a loader architecture that does not grow a new special-case path for
every model and every load format.

## Implementation Notes

### 2026-07-03 Phase 1 start

The first extraction step moved the metadata and plan representation out of the
O_DIRECT safetensors loader into:

```text
vllm/model_executor/model_loader/weight_plan.py
```

The neutral module now owns:

- `TensorMeta`
- `TensorCatalog`
- `WeightPlan`
- `WeightPlanEntry`
- `WeightPlanReadSegment`
- `WeightPlanSummary`
- `summarize_weight_plan`
- `build_auto_weight_plan_from_catalog`
- `build_auto_weight_plan_for_module`

`uma_odirect_safetensors_loader.py` imports and re-exports these names for
compatibility, so existing model hooks can continue importing from the old
loader path while future hooks can import the neutral IR directly.  A small
unit test was added for plan construction and summary accounting without
instantiating the O_DIRECT loader.

### 2026-07-03 Phase 2 start

The neutral `weight_plan.py` module now defines the first explicit planner and
executor contracts:

- `WeightPlanBuilder`
- `WeightPlanExecutor`
- `WeightPlanSourceModel`

These protocols name the existing model hook contract:

```text
build_weight_plan(catalog) -> WeightPlan
load_weights_from_source(source, plan) -> set[str]
```

This does not change execution behavior yet.  It gives the current
`build_weight_plan` / `load_weights_from_source` convention a neutral API home
so future executors can depend on the contract without importing the O_DIRECT
loader implementation.

The first helper on top of these contracts is
`resolve_weight_plan_source_hooks(model)`, which centralizes the fail-closed
check that a model implements both hooks or neither.  The O_DIRECT loader now
uses this neutral helper instead of open-coding the hook detection.

The neutral module also defines the initial `ExecutorCapability` dataclass and
an `ExecutorCapability.uma_odirect()` constructor.  This is not wired into read
scheduling yet, but it gives future schedule validation a model-independent
place to express fail-closed behavior, mmap policy, alignment requirements, and
staging limits.
