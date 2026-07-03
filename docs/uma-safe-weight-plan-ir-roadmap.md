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

One known gap in the current extraction:

- the catalog rejects duplicate tensor names across all files.  That is
  correct for standard HF sharded checkpoints but wrong for pre-sharded
  per-rank checkpoints (`sharded_state`), where the same name legitimately
  appears in each rank file.  Phase 4 needs a rank/namespace dimension on the
  catalog, or per-rank catalogs with explicit merge rules.

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

Transforms are part of this rule, and this is now a decision rather than an
open question: `Transform(op=...)` names an operation in a transform registry.
Opaque Python callables are a prototype convenience only — they make a plan
impossible to serialize, to diff in golden-plan tests, or to validate against
executor capabilities.  Each registered op must declare its extra staging
factor (for example, an op that concatenates two staged tensors temporarily
doubles staging memory), so the planner and `ExecutorCapability` validation
can account for transform memory instead of trusting arbitrary code.  Today a
`transform` callable can allocate unbounded CPU memory invisibly to the
planner, which is a UMA-safety hole.

#### Transform op registry design (2026-07-03)

Inventory of every tensor transform in the tree today (name-only rewrites in
`name_transform` hooks need no ops and are Phase 3 NameResolver work, not
transform work):

```text
zero_mean       sarvam, param2moe   x - x.mean()            allocates ~2x
squeeze(dim=0)  ernie45_moe         view                    ~1x
l2_normalize    bailing_moe         F.normalize(dim=0)      allocates ~2x
qk_rope_permute llama4, mistral     pure n_heads arg        allocates ~2x
patch_reshape   bagel               vit patch embedding     ~1x (view/reshape)
(composition)   bailing_moe         chained callables       product
```

(The mistral and bagel entries were found during migration, in the model
files rather than the `*_uma.py` hooks — the inventory must include model
files that implement `build_weight_plan` directly.)

The vocabulary is four ops plus ordered composition, which answers the open
question about the minimal op set.  Design:

- `TransformOp(op: str, args: tuple[int | float | str | bool, ...] = ())` —
  frozen, serializable; composition is an ordered tuple of ops on the entry
  (`WeightPlanEntry.transform_ops`), replacing callable chaining;
- a module-level registry in `weight_plan.py`:
  `register_weight_transform(name, fn, *, extra_staging_factor)`; duplicate
  names fail; generic ops (`zero_mean`, `squeeze`, `l2_normalize`,
  `qk_rope_permute`, `patch_embedding_reshape`) register at import with
  **pure arguments computed at build time** (head counts, patch size, channel
  counts) — implementations must not capture the model;
- unknown op names fail during plan validation (before payload reads), not
  at execution;
- `WeightPlanSummary` gains transform staging accounting
  (payload × (factor − 1) per entry, peak across entries since execution is
  serial), which `ExecutorCapability.max_staging_bytes` can later veto;
- `name_transform` hook results use `(name, tuple[TransformOp, ...])`; opaque
  callable transforms are not part of `WeightPlanEntry`.

Migration order: (1) registry + entry field + executor application +
validation, legacy field kept temporarily; (2) migrate the three generic-op
families; (3) extract model-bound transform closures into pure-args ops:
llama4 and mistral use shared `qk_rope_permute(n_heads)`, and bagel uses
`patch_embedding_reshape(patch_size, in_channels)`; (4) delete the legacy
`transform` field and callable `name_transform` compatibility.

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

Two hardening rules for the entry type:

- entry validity constraints (for example `read_segments` requiring
  `read_into_cpu=True` plus `staging_shape` and excluding
  `source_slices`/`target_slices`) must be enforced at construction time —
  via `__post_init__` validation or by splitting the entry into explicit
  variants (full read, sliced read, segmented read, skip) — not discovered as
  runtime errors just before payload reads;
- an entry must arrive fully resolved: executors must not infer slices, shard
  sizes, or locality from parameter attributes at execution time.

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

There are two levels of coalescing:

1. Range scheduling without new IR.  The executor can flatten required entry
   ranges, group them by checkpoint file/name, and read source offsets in
   ascending order when the destination staging tensors are independent.  This
   is the preferred first step because it preserves the existing `WeightPlan`
   schema and only changes executor scheduling.
2. A fused-source read primitive.  If range scheduling cannot remove repeated
   sweeps for entries that share one large checkpoint tensor, add an explicit
   grouped-read IR node whose children are ordinary `WeightPlanEntry` targets.
   This is higher risk because it changes the serialized plan shape and must be
   justified in the upstream RFC.

The first concrete target is fused checkpoint tensors split into multiple
entries: HunYuan q/k/v from one interleaved QKV tensor, TeleChat2 k/v from one
interleaved tensor, and Llama4 gate/up splits.  The failure mode is repeated
single-window sweeps over the same large tensor when each logical entry is
executed independently.  This should be visible as expected read amplification
before implementation; actual checkpoint validation is still required because
the standard Qwen smoke set does not exercise real `read_segments` bytes.

Placement and scheduling should remain separate.  Placement describes what this
rank needs and where it goes; scheduling describes how an executor groups,
orders, gates, and releases concrete reads.

## Known Deviations in the Current Prototype

A review of the code against this design (2026-07-03) found places where the
implementation already violates the principles above.  They are recorded here
so the migration phases can close them explicitly instead of leaving them as
folklore.

### Executor-side TP slice inference

`_infer_output_dim_source_slice` and its helpers in the O_DIRECT loader infer
tensor-parallel source slices at execution time from parameter attributes
(`output_dim`, `input_dim`, `tp_rank`, `_get_shard_size_mapping`,
`output_sizes`).  This is exactly the semantic dispatch this document argues
against, and it has a concrete cost: `summarize_weight_plan` accounts entries
with `source_slices=None` as full reads even when the executor will later read
only a slice, so the plan summary can over-report payload bytes.  The plan is
not yet the source of truth for "how much will be read".  Phase 2 moves this
inference into plan construction.

Status: closed 2026-07-03.  The inference now lives in
`resolve_weight_plan()` in the neutral `weight_plan.py`; `execute_weight_plan`
resolves the plan before summarizing and performs no inference of its own.

### `RoutedMoeSourcePlan` bypasses the plan IR

The shared MoE hook helper returns a composite plan (`auto_plan` plus
`routed_entries`) and executes the routed part through its own loop that calls
`read_full_cpu` directly.  Consequences:

- the `WeightPlanExecutor` protocol declares `plan: WeightPlan`, but its main
  users pass a different type, so the contract is effectively untyped;
- routed expert payloads — the largest reads in MoE models — are invisible to
  `summarize_weight_plan`, so read-volume accounting is wrong exactly where it
  matters most;
- a future non-O_DIRECT executor cannot execute the routed part without
  reimplementing the loop, which recreates the side-path problem.

Phase 1 folds routed entries into first-class `WeightPlanEntry` records; the
fields needed for it (`expert_id`, `shard_id`, `weight_name`) already exist.

Status: closed 2026-07-03.
Routed entries are folded into a single `WeightPlan` during plan construction.
The temporary `RoutedMoeSourcePlan` compatibility class and conversion helper
were removed after all current family hooks were audited.  Routed expert target
paths are now derived from `model.named_modules(remove_duplicate=False)` at
build time, and each routed entry carries an explicit `loader_target_name` when
the custom loader lives on the expert module rather than the parameter object.
The executor no longer guesses `routed_experts` children or parent loaders at
execution time.

### No load completeness check

`load_weights()` discards the loaded-parameter set returned by
`load_weights_from_source`, and `build_auto_weight_plan_from_catalog` silently
marks unmapped names `required=False`.  A mapper bug or an unexpected
checkpoint name can therefore leave runtime parameters at their initial values
without any error.  The legacy vLLM loader has an unloaded-parameter check;
the plan path currently does not.  This is the largest fail-closed gap and is
now a Phase 0 task.

Status: closed 2026-07-03.  `verify_loaded_weights()` in `weight_plan.py`
compares the loaded set against `model.named_parameters()` (with the default
loader's quant-method exemptions) after every plan execution, and the plan
path additionally fails if a hook returns no loaded set at all.  The
compatibility iterator path warns instead of failing when a legacy model does
not report its loaded set.

### Safety validation split across layers

The symlink rejection this document attributes to `TensorCatalog` originally
lived in the loader's `_prepare_files`; the extracted catalog therefore
depended on caller discipline for one of its safety guarantees.

Status: closed 2026-07-03.  `TensorCatalog.from_safetensors_files()` now
rejects symlinked and non-file safetensors paths before metadata reads.  The
O_DIRECT loader and metadata audit script only enumerate candidate files; path
safety is centralized in the catalog.

## What This Means for the Current Branch

The current branch is valid as an operational mitigation, but it should be
treated as a prototype with clear boundaries.

Keep doing:

- enforce fail-closed behavior for UMA-safe loads;
- verify the loaded-parameter set against the model's expected parameters
  after every plan execution;
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
- add a loaded-set completeness check: compare the parameter names returned by
  `load_weights_from_source` against `model.named_parameters()` and fail on
  unloaded parameters (a minimal, early version of the
  `ResolvedWeightBinding` completeness validation);
- keep adding only the model hooks needed for actual tests;
- document unsupported cases explicitly;
- keep measuring `used + buff/cache`, not just allocated tensors.

Exit criteria:

- Qwen 35B and selected MoE models load without memory PSI;
- a plan that silently drops a required parameter fails loudly instead of
  leaving initial values in place;
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
- fold `RoutedMoeSourcePlan.routed_entries` into first-class
  `WeightPlanEntry` records so MoE hooks return a plain `WeightPlan` and the
  routed executor loop disappears;
- move symlink/path validation into `TensorCatalog` construction;
- add tests that build a plan without instantiating the O_DIRECT loader.

Exit criteria:

- the O_DIRECT loader imports the IR instead of owning it;
- model hooks import the neutral IR;
- every model hook returns a single `WeightPlan`, and `summarize_weight_plan`
  accounts for all payload bytes including routed experts;
- tests can validate plan construction and plan accounting separately from
  Linux O_DIRECT reads.

### Phase 2: Split planner and executor contracts

Goal: make it clear which code creates a plan and which code executes it.

Tasks:

- define a `WeightPlanBuilder` protocol or model hook contract;
- define a `WeightPlanExecutor` protocol;
- move executor-side TP shard-slice inference
  (`_infer_output_dim_source_slice` and helpers) into plan construction so
  entries arrive fully resolved;
- wire `ExecutorCapability` into a validation step
  (`validate_plan(plan, capability)`) that runs before payload reads, and
  revisit its boolean fields — the design sketch already needs a "maybe" for
  `supports_partial_read`, so tri-state values or a constraint set are likely
  needed;
- rename or wrap `load_weights_from_source(source, plan)` into a plan executor
  call;
- keep legacy iterator loading as a compatibility executor/fallback outside
  UMA-safe mode.

Exit criteria:

- model code never calls raw file I/O;
- executor code never contains model-family mappings;
- executors perform no semantic inference at execution time: the plan summary
  matches executed reads byte-for-byte;
- plans are validated against executor capability before any payload read;
- plan summary can run before payload reads and before executor selection.

### Phase 2.5: Read scheduling and read-amplification accounting

Goal: give the `ReadSchedulePlan` layer an owner.  It is described above but
was previously assigned to no phase.  Routed expert loading — one read per
layer per expert per projection — is the main small-read amplification case
today.

Tasks:

- derive a `ReadSchedulePlan` from a `PlacementPlan` plus `ExecutorCapability`;
- sort entries by source file offset where execution semantics allow, and
  coalesce nearby byte ranges for expected-read accounting;
- report expected read amplification (payload bytes read / payload bytes
  needed) in the plan summary before execution;
- keep placement and scheduling as separate artifacts.

Exit criteria:

- routed-MoE loads no longer rely on incidental auto-plan catalog ordering to
  avoid a second pass over expert regions;
- read amplification is reported for every load and tracked as a regression
  metric.

### Phase 3: Move from model-level hooks to module/parameter semantics

Goal: shrink the existing `*_uma.py` mapping-table pile.

This is not a future risk.  As of 2026-07-03 there are 27 `*_uma.py` model
hooks totalling roughly 5,900 lines.  The shared `routed_moe_uma.py` helper is
the right direction, but every family still hand-writes `parse_name`
(checkpoint-name string splitting) and `resolve_routed_experts` (model
traversal) — both are exactly what the SemanticSpec pattern declarations
should replace, so this phase is more urgent than its position in the
sequence suggests.

Tasks:

- replace per-family `parse_name` functions with declarative source patterns
  (for example
  `RoutedExpert(source="layers.{i}.mlp.experts.{e}.{proj}.{suffix}")`)
  resolved by a shared matcher;
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
- model-specific lines per newly added model family stay under an agreed
  budget, tracked per addition;
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

## Success Metrics

The phase exit criteria are mostly qualitative.  These quantitative metrics
detect regression toward a semantic-dispatcher loader and should be tracked
across phases:

- read amplification: payload bytes read / payload bytes needed per load;
- peak `used + buff/cache` during load on the UMA target machine;
- model-specific lines added per new model family (should fall phase over
  phase);
- plan construction time (metadata-only, must stay trivially cheap);
- number of plan entries whose treatment is decided at execution time (must
  reach zero at Phase 2 and stay there).

## Upstreaming Posture

This tree is a vendor fork; 27 added files under `models/` plus loader changes
carry real rebase cost, and that cost grows the longer Phase 3 is deferred.
Working stance until an upstream RFC exists:

- `weight_plan.py` (the neutral IR) is the upstream RFC candidate and must
  stay free of fork-only dependencies;
- `*_uma.py` hooks are disposable prototypes: they are not preserved across
  rebases at the cost of IR clarity, and no external code should import them;
- the O_DIRECT executor is the reference UMA-safe executor — useful upstream,
  but secondary to the IR proposal itself.

## Open Design Questions

- What is the smallest set of first-class spec objects that covers most dense,
  fused, and routed-MoE models?
- What is the minimal transform-op vocabulary, and how does each op declare
  its staging-memory factor?  (Named ops themselves are decided; see the
  SemanticSpec section.)
- How should `TensorCatalog` represent pre-sharded per-rank checkpoints where
  the same tensor name legitimately appears in each rank file?
- How should quantization methods attach auxiliary tensors, scales, and
  packed layouts to the plan?
- Can direct destination placement be expressed safely for CUDA tensors, or
  should CPU segmented staging remain the only safe portable primitive?
- How should error messages expose "unsupported by executor" versus "invalid
  model plan" versus "missing checkpoint tensor"?
- How much of this can be proposed upstream without requiring every model to
  migrate at once?

## Practical Next Steps

For this branch, Phase 3 is closed as of 2026-07-03.  The completed set now
includes the loaded-set completeness check, the routed-plan fold, TP slice
inference in plan resolution, Phase 2.5 read-window reuse and read scheduling,
build-side routed-plan unification, compatibility layer removal, build-time
routed target path derivation, named transform ops, golden plan snapshots,
TensorCatalog path validation, Phase 3 declarative routed specs, composite
routed side-path removal, and real-load smoke validation.

The next useful work is, in priority order:

1. review and refine the upstream RFC draft in
   `docs/uma-safe-weight-plan-upstream-rfc.md`;
2. keep `read_segments` real-checkpoint validation as a TODO for TeleChat2,
   HunYuan, or Llama4 when a suitable checkpoint is available;
3. treat FlashInfer/CUDA JIT ready-start memory spikes as a separate runtime
   track, not a loader-IR blocker.

This lets the branch move from proving the UMA-safe loader architecture to
preparing the neutral `WeightPlan` IR for upstream discussion.

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
an `ExecutorCapability.for_aligned_direct_io()` constructor.  This is not wired
into read scheduling yet, but it gives future schedule validation a
model-independent place to express fail-closed behavior, mmap policy,
alignment requirements, and staging limits.

### 2026-07-03 design review

A review of the code against this document added the "Known Deviations in the
Current Prototype" section and reprioritized the phases.  The
highest-priority gaps found:

- no loaded-set completeness check — `load_weights()` discards the loaded set
  and auto plans silently skip unmapped names (now a Phase 0 task);
- `RoutedMoeSourcePlan` bypasses the plan IR and its summary accounting, so
  routed expert bytes are invisible to `summarize_weight_plan` (now a Phase 1
  task);
- executor-side TP slice inference makes the plan summary diverge from actual
  reads (now a Phase 2 task).

Transforms-as-named-ops was promoted from an open question to a decision,
Phase 2.5 was added for read scheduling and read-amplification accounting,
and quantitative success metrics plus an upstreaming posture were added.

### 2026-07-03 review fixes implemented

The three highest-priority gaps from the review were closed in code:

- **Loaded-set completeness check (Phase 0).**  `verify_loaded_weights()` was
  added to `weight_plan.py` and is called by
  `UmaODirectSafetensorsModelLoader.load_weights()` on both the plan path and
  the compatibility iterator path.  The plan path fails closed if a model hook
  returns no loaded set; the compat path logs a warning for legacy models
  that return `None`.
- **Routed MoE entries folded into the plan IR (Phase 1).**
  `routed_moe_uma.py` now folds routed entries into a single `WeightPlan` at
  build time.  Local routed entries become required `WeightPlanEntry` records
  with target paths derived from registered model modules; loader metadata is
  carried in `shard_id`, `expert_id`, `weight_name`, and `loader_target_name`.
  Non-local entries become skips with a `skip_reason`.  Routed expert bytes are
  included in `summarize_weight_plan` accounting, the side executor loop is
  gone, and the temporary `RoutedMoeSourcePlan` compatibility layer has been
  removed.
- **Executor-side TP slice inference moved to plan resolution (Phase 2).**
  `_infer_output_dim_source_slice` and helpers moved to `weight_plan.py`, and
  a new `resolve_weight_plan(model, catalog, plan)` fills TP source slices
  before summary and execution.  `execute_weight_plan` resolves first, then
  executes without inference, so the logged summary matches actual reads.
  Entries with `expert_id` are exempt from inference (expert loaders narrow
  internally).

Supporting changes: `WeightPlanEntry.skip_reason` was added and is surfaced
in skip logging; `_call_weight_loader` now validates `return_success` for
expert entries whose loader supports it, so a loader refusing a tensor the
plan marked local fails closed instead of being silently ignored.  Unit tests
cover plan resolution, completeness verification, routed plan folding, and
the refusal path
(`tests/model_executor/model_loader/test_weight_plan.py`,
`tests/model_executor/model_loader/test_routed_moe_uma_plan.py`).

### 2026-07-03 runtime validation

The review fixes were validated on DGX Spark after explicit pre-clean
(`drop_caches` and swap cycle on both nodes) and a final preflight check:
no backend-like processes, local `MemAvailable` 109 GiB, remote
`MemAvailable` 112 GiB, swap 0 on both nodes, and memory PSI avg10 0.

Runs used `load_format=uma_odirect_safetensors`, `max_model_len=128`,
`max_num_seqs=1`, `enforce_eager=True`, `min_available_gib=20`,
`max_swap_gib=0`, and `psi_gate_seconds=30`.  The vLLM engine was stopped
immediately after model-load completion to avoid unrelated post-load profile
JIT failures in the local test environment.  Full logs and RAM/PSI CSV files
were written under `/tmp/vllm-uma-load-tests/`.

Results:

```text
tiny-random-qwen3-moe
  plan entries: 46 required, 0 skipped
  total_read_payload: 0.02 GiB
  source bytes_read / bytes_copied: 0.03 GiB / 0.02 GiB
  model load: 0.02 GiB, 2.24 s
  peak used + buff/cache: 16.44 GiB
  min available: 107.49 GiB
  swap: 0
  memory PSI avg10 max: 0 / 0

PrimeIntellect-qwen3-moe-tiny
  plan entries: 1325 required, 0 skipped
  total_read_payload: 1.25 GiB
  source bytes_read / bytes_copied: 150.39 GiB / 1.25 GiB
  model load: 1.26 GiB, 61.47 s
  peak used + buff/cache: 17.44 GiB
  min available: 106.52 GiB
  swap: 0
  memory PSI avg10 max: 0 / 0

tiny-random-qwen3.5-moe
  plan entries: 2017 required, 17 skipped
  total_read_payload: 0.01 GiB
  source bytes_read / bytes_copied: 2.24 GiB / 0.01 GiB
  model load: 0.02 GiB, 86.39 s
  peak used + buff/cache: 17.66 GiB
  min available: 106.32 GiB
  swap: 0
  memory PSI avg10 max: 0 / 0

Qwen3.6-35B-A3B-heretic-NVFP4
  settings difference: chunk_size=1 MiB, window_size=1 MiB,
    gate_interval_mib=16, metadata_limit_mib=32
  plan entries: 124306 required, 0 skipped
  total_read_payload: 21.73 GiB
  source bytes_read / bytes_copied: 125.91 GiB / 21.73 GiB
  model load: 21.88 GiB, 90.99 s
  peak used + buff/cache: 40.00 GiB
  min available: 84.06 GiB
  swap: 0
  memory PSI avg10 max: 0 / 0
  IO PSI avg10 max: 10.01 / 10.01
```

No loaded-set completeness false positive was observed in these model-side
plan runs.  The most important metric is read amplification: the
PrimeIntellect tiny MoE run read 150.39 GiB for 1.25 GiB of payload with the
128 MiB window setting, while the 35B NVFP4 run read 125.91 GiB for
21.73 GiB of payload after reducing the O_DIRECT window to 1 MiB.  This
validates the Phase 2.5 priority: read scheduling and coalescing should become
a first-class regression metric rather than relying on window-size tuning.

### 2026-07-03 Phase 2.5 stage 1: read-window reuse across plan entries

The runtime numbers above identified the amplification mechanism exactly:
every plan-path read (`_read_record_cpu`, `_read_record_into_cpu`, and both
strided readers) opened its own `_ODirectFile` per call, and the read window
lives on that handle, so each entry paid at least one window-sized pread.
The measurements match "one window load per entry" almost perfectly:

```text
35B NVFP4: 124,306 entries x 1 MiB window   ~ 121 GiB  vs 125.91 GiB read
tiny MoE:    1,325 entries x 128 MiB window ~ 166 GiB  vs 150.39 GiB read
qwen3.5:     2,017 entries x ~1.1 MiB file  ~ 2.2 GiB  vs   2.24 GiB read
```

(The compat iterator path already kept one handle open per file, which is why
this never showed up before the plan path became the default.)

Fix: `ODirectSafetensorsWeightSource` now caches the most recently used
`_ODirectFile` (`_open_file` / `close_files`), and all plan-path readers plus
`iter_full_tensors` share it.  Only one file stays open, so fd count and
window-buffer residency remain bounded at one — peak memory behavior is
unchanged.  Because auto plans iterate the catalog in (file, offset) order,
adjacent small tensors now hit the same window instead of each paying a full
window read.  Expected result: amplification drops to roughly
`bytes_read ~ file bytes touched` for offset-ordered plan segments; routed
entries appended after the auto plan can still cause a second pass over
expert regions (bounded ~2x), which is what the stage 2 `ReadSchedulePlan`
(entry ordering and range coalescing) will remove.

DGX Spark re-measurement after the fix:

```text
Model                         Payload   bytes_read  Window loads  Load time
PrimeIntellect tiny MoE       1.25 GiB    1.85 GiB            11     0.90 s
tiny-random-qwen3.5 MoE       0.01 GiB    0.01 GiB             1     0.43 s
Qwen3.6 35B NVFP4            21.73 GiB   28.90 GiB           218    23.93 s
```

Safety counters stayed clean: swap remained 0 and memory PSI stayed 0 for all
three runs.  The 35B run was measured with the normal 128 MiB window again;
the previous 1 MiB workaround is no longer required for amplification control.
Per-file counters are now folded into source stats when the handle is closed or
switched, and `stats_snapshot()` includes the currently open handle so registry
tests and mid-load snapshots see live counters.

### 2026-07-03 Phase 2.5 stage 2: read schedule summary

Stage 2 now has a first concrete `ReadSchedulePlan`: required plan entries are
ordered by `(file, first_offset)` before the common executor reads them, skipped
entries remain recorded but move after required reads, and a metadata-only
O_DIRECT simulation reports expected direct reads, window loads, window hits,
bytes read, payload bytes, and expected read amplification before execution.

This makes the low-amplification property explicit instead of depending on the
current auto-plan construction order.  The initial stage does not introduce a
new bulk group-read API; the actual read path still uses the Stage 1 handle
cache, while the schedule gives logs and CI a stable expected-amplification
metric and prepares the IR for later range-group execution if it becomes worth
the extra staging complexity.

Stage 2 follow-up: schedule construction now computes read ranges once and
shares them across ordering, payload accounting, and simulation.  Strided reads
are represented as `(offset, size, repeat, stride)` instead of enumerating every
row/column segment, which keeps tensor-parallel slice planning bounded by
entries rather than rows.  Source stats also compare actual `bytes_read` to the
scheduled expectation after execution and warn when actual exceeds expected by
more than 10%, making read-amplification regression detection permanent in the
load logs.

DGX Spark validation after the Stage 2 follow-up:

```text
Model                         Expected   Actual   Amplification  Load time
PrimeIntellect tiny MoE        1.25 GiB  1.25 GiB         1.00x     0.77 s
tiny-random-qwen3.5 MoE        0.01 GiB  0.01 GiB         1.00x     0.40 s
Qwen3.6 35B NVFP4             22.23 GiB 22.23 GiB         1.02x    23.46 s
```

All three runs matched the scheduled expectation exactly, no
`actual read amplification exceeded` warnings were emitted, swap stayed at 0,
and memory PSI stayed at 0.  The PrimeIntellect tiny MoE case dropped from the
Stage 1 value of 1.85 GiB to 1.25 GiB, confirming that routed reads no longer
depend on incidental auto-plan catalog ordering.  Phase 2.5 is therefore closed
for the current branch: expected read amplification is reported before
execution and actual-vs-expected drift is now a standing load-log regression
signal.

### 2026-07-03 build-side routed-plan unification

`build_routed_moe_weight_plan()` now returns a plain `WeightPlan`.  Routed MoE
entries are folded during plan construction, not between build and load, so
`WeightPlanExecutor` again sees the declared `build_weight_plan(catalog) ->
WeightPlan` contract for pass-through MoE families.

The model-family hooks that used to expose `RoutedMoeSourcePlan` as their
source plan type now alias their source plan to `WeightPlan`.  The post-process
families that rewrote `auto_plan` / `routed_entries` between build and load
were migrated to operate on first-class `WeightPlanEntry` records:

- HunYuan v1 keeps only fused-qkv side entries outside the common plan;
- Llama4 keeps only fused expert source-slice entries outside the common plan;
- MiniMax M2, MiMo v2, Param2MoE, Granite MoE, LongCat Flash, and OpenPangu
  now filter or annotate `WeightPlan.entries` directly.

Local routed entries carry `shard_id`, `expert_id`, and `weight_name` on the
plan entry; non-local routed entries are ordinary skipped entries with the same
routed metadata and `skip_reason="non-local routed expert"`.  The executor also
accepts routed expert wrappers where the custom loader lives on the parent
expert module rather than the parameter object, preserving the routed loader
call convention after unification.

Follow-up on 2026-07-03 removed the old `RoutedMoeSourcePlan` class and
`routed_moe_source_plan_to_weight_plan()` helper entirely.  The build helper now
requires routed experts to be registered model modules and derives their target
path with `named_modules(remove_duplicate=False)`, preferring paths that match
the expert module's `layer_name`.  Routed entries that need a module-level
custom loader carry `loader_target_name`, so `execute_weight_plan()` no longer
has a hard-coded `routed_experts` fallback or parent-loader guessing logic.

### 2026-07-03 transform op registry, stages 1-3

`weight_plan.py` now owns the named transform registry:

- `TransformOp(op, args)` — frozen, serializable op reference; composition is
  an ordered tuple in the new `WeightPlanEntry.transform_ops` field;
- `register_weight_transform(name, fn, *, extra_staging_factor)` — idempotent
  for identical re-registration, fails closed on conflicting names;
- generic ops registered at import: `zero_mean`, `squeeze(dim)`,
  `l2_normalize(dim, eps)`, `qk_rope_permute(n_heads)`, and
  `patch_embedding_reshape(patch_size, in_channels)`;
- `summarize_weight_plan` resolves each required entry's ops (unknown op
  names fail before any payload read) and reports
  `peak_transform_staging_bytes` from declared staging factors;
- the executor applies only `transform_ops`; opaque callable transforms were
  removed from `WeightPlanEntry` in stage 4.

Migrated to named ops: sarvam (`zero_mean`, including the model-file copy in
`sarvam.py`), param2moe (`zero_mean`), ernie45_moe (`squeeze`), bailing_moe
(`l2_normalize`, with `_compose_name_transform` now concatenating op tuples
instead of chaining callables), llama4 and mistral (`qk_rope_permute`), and
bagel (`patch_embedding_reshape`).  mimo_v2's attention-sink rewrite passes
`transform_ops` through.

Stage 3 removed the remaining model-bound transform closures: llama4 no
longer captures `model.permute_qk_weight_for_rotary`, mistral shares the same
rope permute op with pure `n_heads`, and bagel's patch reshape uses static
patch/channel arguments.

Stage 4 removed the legacy `WeightPlanEntry.transform` field and the callable
`name_transform` compatibility path.  `WeightPlanEntry` is now data-only for
transforms, with all transform behavior represented by serializable
`TransformOp` tuples.  This unblocks golden-plan serialization tests.

### 2026-07-03 golden plan test seed

`tests/model_executor/model_loader/test_weight_plan_golden.py` adds the first
inline golden snapshots for representative plan builders:

- Qwen3 MoE routed local/non-local expert entries, including
  `loader_target_name` and skip metadata;
- Mistral Q projection remap with shared `qk_rope_permute`;
- Bagel patch embedding with `patch_embedding_reshape`;
- Llama4 dense Q transform, routed local/non-local experts, and fused expert
  source slices.

The snapshot serializer intentionally records stable plan semantics rather
than environment-specific file paths or runtime tensors: checkpoint name,
target name, required/skipped state, slices, read segments, transform ops,
routed metadata, loader target, and skip reason.  This is the safety net for
Phase 3 parse-name spec migration: a declarative rewrite should either keep
these snapshots byte-for-byte equivalent or produce a small, reviewable diff.

### 2026-07-03 Mistral qscale shape and transform-track closeout

A review of the transform op migration found one legacy semantic difference
between Llama4 and Mistral: 1-D Mistral `qscale_weight` tensors used
`_permute_mistral_weight(..., attn_out=1)` and returned a 2-D `(attn_in, 1)`
tensor, while Llama4's rotary permutation squeezed 1-D scale tensors back to
1-D.  The registry now exposes two explicit ops:

- `qk_rope_permute(n_heads)` preserves the Llama4 1-D-in/1-D-out behavior;
- `qk_rope_permute_2d(n_heads)` preserves the Mistral qscale 1-D-in/2-D-out
  behavior.

Mistral qscale entries use the 2-D op and are covered at three levels:
generic transform unit tests, the Mistral registry test against the legacy
method, and the Mistral golden snapshot.  `TensorCatalog.numel(name)` replaced
the duplicated local `_tensor_numel` helpers, and Mistral's catalog argument is
now required so qscale transform selection cannot silently skip shape checks.

DGX Spark real-load smoke after stage 4, golden snapshots, and the Mistral
qscale fix:

```text
Model                         Expected   Actual   Amplification  Load time
tiny-random-qwen3.5 MoE        0.01 GiB  0.01 GiB         1.00x     0.41 s
PrimeIntellect tiny MoE        1.25 GiB  1.25 GiB         1.00x     0.77 s
Qwen3.6 35B NVFP4             22.86 GiB 22.86 GiB         1.05x    24.20 s
```

All three runs used the model WeightSource path, matched scheduled reads
exactly, emitted no actual-vs-expected amplification warning, kept swap at 0,
and kept memory PSI at 0.  The vLLM engine failed later during profile/JIT in
the local host environment (`Python.h`/`ninja` unavailable), after model-load
completion, so these runs validate the loader path but not end-to-end serving.

### 2026-07-03 TensorCatalog path validation

`TensorCatalog.from_safetensors_files()` now rejects symlinked and non-file
`.safetensors` paths before reading metadata.  The O_DIRECT loader's
`_prepare_files()` and the metadata audit script only enumerate candidates;
the catalog owns the path safety guarantee together with metadata validation,
duplicate-name checks, overlap checks, dtype checks, and byte-range checks.

### 2026-07-03 Phase 3 parse-name spec investigation

The current UMA hook layer has 25 `*_uma.py` files plus the shared
`routed_moe_uma.py` helper, about 5k lines total.  The repeated structure is
clear enough to introduce a declarative spec in stages, starting with standard
routed MoE families and leaving fused/paired tensor cases as explicit
extensions until the DSL proves itself.

The proposed first-pass spec should cover these primitives:

- `NameRewrite`: ordered literal replacements plus optional skip rules, e.g.
  Param2MoE/OpenPangu/HunYuan checkpoint names and rotary-cache/MTP skips.
- `StackedAlias`: existing mapper-style rules for q/k/v and gate/up stacking,
  including shard IDs (`q`, `k`, `v`, `w1`, `w2`, `w3`).
- `RoutedExpertPattern`: tokenized path pattern with bindings for
  `layer_id`, `expert_id`, `projection`, and `suffix`, plus a projection map
  to `(param_name, shard_id)`.
- `LayerResolver`: declarative path from model root to layer list and routed
  expert module, with PPMissingLayer skip behavior and fail-closed loader
  presence checks.
- `TransformRule`: checkpoint-name predicates plus `TransformOp` tuples, using
  catalog shape predicates where needed.
- `SliceRule`: metadata-derived source slices for fused qkv/gate-up/shared
  expert tensors.  This should be a second layer on top of the simple routed
  pattern because it needs catalog shapes, TP rank/size, or local expert maps.

Families that should be migrated first:

1. Qwen/Mixtral/Jamba/Laguna/Sarvam/Bailing/Ernie45/AFMoE/EXAONE/Nemotron-H:
   standard routed expert pattern plus small name transform or transform op.
2. MiMoV2/Param2MoE/OpenPangu/HunYuan: standard routed pattern plus stacked
   aliases, skips, and one or two metadata-derived slice rules.
3. DeepSeek/GLM4/Granite/Llama4: shared/fused expert sources and local expert
   slicing; migrate after the simpler spec is covered by golden snapshots.

Golden snapshots should be added or expanded before each family moves.  The
first implementation target should be Qwen or Mixtral because their parser is
almost pure `RoutedExpertPattern`; Mistral/Bagel already exercise transform
ops but are not routed-pattern migrations.

### 2026-07-03 Phase 3 stage 1: Qwen routed expert pattern

The first declarative parse-name primitive is now implemented in
`routed_moe_uma.py`:

- `RoutedExpertPattern` parses common
  `...layers.<layer_id>.<module_path>.<expert_id>.<projection>.<suffix>`
  checkpoint names into `(layer_id, expert_id, projection, suffix)`;
- `RoutedProjectionMap` maps projection names to `(param_name, shard_id)` via
  data rules such as `gate_proj -> w13_<suffix>, w1`.

`qwen_moe_uma.py` no longer owns a hand-written routed parser or projection
mapper.  It declares:

```python
RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
RoutedProjectionMap((
    RoutedProjectionRule("gate_proj", "w13", "w1"),
    RoutedProjectionRule("up_proj", "w13", "w3"),
    RoutedProjectionRule("down_proj", "w2", "w2"),
))
```

The existing Qwen golden snapshot stayed unchanged, which is the intended
Phase 3 migration contract: replacing parser code with spec data must preserve
the serialized `WeightPlan`.  This pattern should be reusable for Mixtral,
Jamba, Laguna, Sarvam, Bailing, Ernie45, AFMoE, EXAONE, and Nemotron-H before
moving on to families that need shape-derived `SliceRule`s.

### 2026-07-03 Phase 3 stage 1b: standard routed expert patterns

`mixtral_uma.py`, `jamba_uma.py`, and `laguna_uma.py` now use the same
declarative primitive as Qwen instead of owning family-local parsers and
projection mappers.

Mixtral keeps its checkpoint spelling (`w1`/`w2`/`w3`) as data:

```python
RoutedExpertPattern(
    module_path=("block_sparse_moe", "experts"),
    projections=("w1", "w2", "w3"),
)
RoutedProjectionMap((
    RoutedProjectionRule("w1", "w13", "w1"),
    RoutedProjectionRule("w3", "w13", "w3"),
    RoutedProjectionRule("w2", "w2", "w2"),
))
```

Jamba differs only in the module path:

```python
RoutedExpertPattern(
    module_path=("feed_forward", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
```

Laguna uses the same `mlp.experts` pattern as Qwen, with family-specific layer
resolution and skip behavior left in ordinary Python for now.

The existing Mixtral, Jamba, and Laguna plan tests still exercise the actual
plan paths, and the generic routed-pattern unit test now covers Qwen-style
`mlp.experts`, Mixtral-style `block_sparse_moe.experts`, and Jamba-style
`feed_forward.experts` names.

### 2026-07-03 Phase 3 stage 1c: standard pattern with transforms/mappers

`sarvam_uma.py`, `bailing_moe_uma.py`, and `ernie45_moe_uma.py` now also use
`RoutedExpertPattern(module_path=("mlp", "experts"), ...)` plus the standard
gate/up/down projection map.  Their family-specific behavior remains outside
the pattern primitive:

- Sarvam keeps the `zero_mean` gate-bias transform;
- Bailing keeps its auto-path `gate_up_proj` mapper and optional
  `l2_normalize` head transform;
- Ernie 4.5 keeps its QKV/gate-up mapper, `squeeze` gate-bias transform, MTP
  skip, and tied-head skip.

This is the intended granularity for the first declarative pass: parser and
projection-map boilerplate become data, while transform, skip, mapper, and
layer-resolution policy stay as explicit Python until their own spec
primitives are introduced.

### 2026-07-03 Phase 3 stage 1d: mapper-backed standard patterns

`afmoe_uma.py`, `exaone_moe_uma.py`, and `nemotron_h_uma.py` have joined the
same declarative parser/projection-map path.  AFMoE and EXAONE use the
standard `mlp.experts` + `gate_proj`/`up_proj`/`down_proj` pattern while
keeping their mapper and skip options outside the primitive.  Nemotron-H uses
the same parser with a different module path and a two-projection map:

```python
RoutedExpertPattern(
    module_path=("mixer", "experts"),
    projections=("up_proj", "down_proj"),
)
RoutedProjectionMap((
    RoutedProjectionRule("up_proj", "w13", "w1"),
    RoutedProjectionRule("down_proj", "w2", "w2"),
))
```

This leaves dense-layer rejection in AFMoE and mapper replay in Nemotron-H as
family policy while removing the repeated checkpoint-token parser from both.

### 2026-07-03 Phase 3 stage 1e: mapper and skip-wrapper patterns

`lfm2_moe_uma.py`, `kimi_linear_uma.py`, and `hy_v3_uma.py` now use
`RoutedExpertPattern` as well:

- LFM2 uses `feed_forward.experts` with `w1`/`w2`/`w3`, plus its existing
  name transform and mapper;
- Kimi Linear uses `block_sparse_moe.experts` with `w1`/`w2`/`w3`, plus its
  speculative-layer skip predicate;
- HYV3 uses `mlp.experts` with the standard gate/up/down projection map, while
  retaining its wrapper that suppresses routed parsing for names skipped by
  the family skip predicate.

At this point the first-pass routed parser primitive covers the standard MoE
families that only need tokenized path matching plus projection-map data.  The
remaining handwritten parsers mostly involve name rewrites, fused/shared
expert entries, or source-slice rules and should move after those primitives
are explicit.

### 2026-07-03 Phase 3 stage 2 design: three orthogonal deviation primitives

A cross-family survey of the nine remaining handwritten parsers
(Param2MoE, OpenPangu, HunYuan v1, GLM4 MoE, MiMo V2, MiniMax M2, LongCat,
DeepSeek, Llama4) shows their deviations from `RoutedExpertPattern` decompose
into three orthogonal axes. They should be modeled as three independent
optional fields on a family spec — not one mega-primitive — so each is
testable alone and explainable separately in the upstream RFC.

| Axis | What it expresses | Families |
| --- | --- | --- |
| `NameRewrite` | literal substitutions applied before pattern matching (`.attention.` → `.self_attn.`, `gate_proj_bias` → `gate_proj.bias`, `model.` prefix) | Param2MoE, HunYuan, OpenPangu, MiniMax M2, DeepSeek |
| `StackedProjection` | two source projections folded into one fused target param with a shard index (`gate_proj`/`up_proj` → `gate_up_proj` 0/1, `q_a_proj`/`kv_a_proj_with_mqa` → `fused_qkv_a_proj` 0/1) | GLM4, LongCat, OpenPangu, HunYuan, DeepSeek |
| `SliceRule` | one checkpoint tensor fissioned into N entries with `source_slices` (fused qkv row split, `gate_and_up_proj` halves, MiMo attention-sink head slice) | Param2MoE, HunYuan, MiMo V2, (Llama4 `fused_expert_entries`) |

Design decisions:

- `NameRewrite` is an ordered tuple of `(old, new, count=1)` literal
  substitutions — pure data, serializable. Substitution strings must be
  anchored with surrounding dots (or an explicit prefix/suffix position) to
  avoid mid-token matches; bare-token replacements are a spec violation.
  Param2MoE's parser already calls its name transform before pattern
  matching, proving rewrite→pattern composition works.
- `StackedProjection` extends `RoutedProjectionMap` with a
  `(fused_target, source_projection, shard_index)` table. It is the same
  shape as upstream vLLM's `stacked_params_mapping`, so naming should mirror
  it for RFC credibility. GLM4/DeepSeek gate shared-expert fusion on
  `rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()`; that runtime
  branch stays in the builder and selects between specs — plan output remains
  concrete data either way.
- `SliceRule` maps one source name to `[(target, shard_id, slices)]` where
  slice boundaries come from config-derived sizes (q_split/kv_split/half)
  resolved to concrete values at build time. Plans are built per-rank, so
  MiMo's TP-dependent head slicing fits the same shape. Golden-plan tests pin
  the concrete values.

Migration order: (1) `NameRewrite` — cheapest, immediately unblocks Param2MoE
and MiniMax M2 with the existing pattern; (2) `StackedProjection` — highest
leverage (five families) and the strongest upstream-RFC story; (3)
`SliceRule` — hardest, config-dependent; (4) fold Llama4's
`fused_expert_entries` composite last, once slices are declarative.

### 2026-07-03 Phase 3 stage 2a: NameRewrite primitive

`routed_moe_uma.py` now includes the first deviation primitive:

- `NameRewriteRule(old, new, count=1)` for one literal substitution;
- `NameRewriter(rules)` for ordered composition.

Rules fail closed unless `old` is boundary-aware: either dot-anchored for
internal replacements (for example `.attention.`) or explicit prefix-style
(`model.` / `model.word_embeddings.`).  Prefix-style rules only apply at the
start of a checkpoint name, so a middle-token occurrence such as
`prefix.model.word_embeddings.weight` is not rewritten.

Param2MoE now uses `NameRewriter` for its attention/embedding checkpoint
spelling changes before applying `RoutedExpertPattern`.  MiniMax M2 uses the
same primitive for its `model.` prefix strip before routed pattern parsing and
auto-plan mapping.  Family-specific transform behavior still remains outside
the rewrite primitive: Param2MoE's expert-bias `zero_mean` transform is kept in
its existing `name_transform`.

### 2026-07-03 Phase 3 stage 2b: StackedProjection primitive

`routed_moe_uma.py` now has `StackedProjectionRule` and
`StackedProjectionMap`.  The map converts directly to a `WeightsMapper` with
`orig_to_new_stacked`, deliberately mirroring upstream vLLM's
`stacked_params_mapping` shape: source projection token, fused target token,
and shard id.

Initial migrations:

- OpenPangu: standard QKV + gate/up stacking, plus optional MLA
  `q_a_proj`/`kv_a_proj_with_mqa` fusion selected by the existing builder
  branch;
- GLM4 MoE: standard QKV + gate/up stacking, plus optional MLA fusion selected
  by the existing `include_mla` flag; fused shared expert slicing remains
  explicit until `SliceRule`;
- LongCat Flash: MLA and dense-`mlps` gate/up stacking; the existing guard that
  avoids applying dense stacking to routed `mlp` remains in the wrapper;
- HunYuan v1: standard QKV + gate/up stacking, plus NameRewrite-backed
  `gate_proj_bias`/`up_proj_bias`/`mlp.gate.wg` spelling fixes.  Fused qkv and
  `gate_and_up_proj` splitting remain explicit until `SliceRule`.

These migrations also move the corresponding routed expert parser boilerplate
to `RoutedExpertPattern` and `RoutedProjectionMap`.  DeepSeek has the same
projection vocabulary, but its mapper also checks target parameter existence
and controls shared-expert fallback behavior, so it is intentionally left for
a dedicated pass rather than mixed into the first StackedProjection commit.

### 2026-07-03 Phase 3 stage 2c: SliceRule primitive

`routed_moe_uma.py` now has the first concrete slice primitive:

- `SliceRuleEntry(target_name, source_slices, shard_id=None)`;
- `SliceRule(checkpoint_name, entries).to_weight_plan_entries()`.

This deliberately represents **resolved** slices, not shape formulas.  Family
builders still compute q/k/v split sizes, half points, and TP rank-local head
ranges from config/catalog state, then emit serializable concrete slice
entries.  That keeps the `WeightPlan` snapshot stable and avoids embedding
model-specific shape algebra in the executor.

Initial migrations:

- Param2MoE fused qkv row split now emits three concrete `SliceRuleEntry`
  records for q/k/v;
- HunYuan `gate_and_up_proj` half split now emits two concrete entries for
  gate/up shards.  HunYuan fused qkv remains an explicit full-read reshape
  path because it is not a simple source slice yet;
- MiMoV2 attention-sink TP head slicing now uses a one-entry `SliceRule` after
  the rank-local head range is computed.  MiMoV2's routed parser and stacked
  projection table were moved to the existing declarative primitives at the
  same time.

The remaining slice work is to decide whether HunYuan fused qkv and Llama4
fused expert entries should become simple `SliceRule`s or a slightly richer
variant that can express reshape/reorder semantics without hiding extra
staging memory.

### 2026-07-03 Phase 3 stage 2d design: no reshape/reorder primitive — segments

Decision for the question left open by stage 2c: HunYuan fused qkv (and the
Llama4 fused expert composite) do NOT need a reshape/reorder primitive, and we
deliberately will not add one.

The HunYuan `_split_fused_qkv_shards` path — full read, then
`reshape(num_kv_heads, groups + 2, head_dim, hidden)` + `split(dim=1)` — is
not an element reorder. It is a strided row gather: q takes, per kv head, a
block of `groups * head_dim` rows at stride `(groups + 2) * head_dim`; k and v
take one `head_dim`-row block each at fixed offsets inside the same stride.
Row gathers are exactly what the IR's existing
`WeightPlanReadSegment(source_slices, target_slices)` expresses:

- the executor already supports segment entries (with mandatory
  `staging_shape`, fail-closed shape verification);
- the read scheduler already accounts for strided segments in closed form
  (`_PlanReadRange(offset, size, repeat, stride)`), so expected/actual byte
  accounting stays exact — unlike the current full-read path, whose staging
  cost lives outside the accounting;
- `telechat2.py` already builds interleaved K/V row-gather entries from a
  fused qkv checkpoint via `read_segments`, so this is precedent, not new
  machinery.

Division of responsibility, final: **segments say which bytes land where;
`TransformOp` does element math** (rope permute etc.). Every remaining family
is a composition of the two. A general reshape/reorder primitive would
duplicate both responsibilities and muddy the byte-accounting story the
upstream RFC depends on.

Plan:

1. extend `SliceRule` with a segments form (`SliceRuleEntry` gains optional
   `segments` + `staging_shape`, or a sibling `SegmentRule`), plus a shared
   interleaved-row-gather generator
   (`group_count`, per-group row counts → segment tuples) reused by telechat2
   and HunYuan;
2. migrate HunYuan fused qkv off the full-read path — this also brings its
   staging under scheduler accounting;
3. DeepSeek: the mapper's target-existence check and shared-expert fallback
   control become builder-side conditional spec selection (same pattern as
   GLM4's rocm_aiter gate) — the branch stays in code, the plan output stays
   concrete data;
4. Llama4 `fused_expert_entries` folds last, once the segments form exists.

### 2026-07-03 Phase 3 stage 2e: segmented SliceRule entries

`SliceRuleEntry` now supports the existing segmented-read IR:
`read_segments` plus `staging_shape`, with fail-closed validation that exactly
one of `source_slices` or `read_segments` is present.  A shared
`build_interleaved_row_gather_segments(...)` helper generates the common
row-gather segment tuples used by TeleChat2 and HunYuan.

TeleChat2's existing key/value fused qkv split now uses the shared helper,
preserving the prior `WeightPlanReadSegment` output.  HunYuan fused qkv has
moved off its old side path entirely: instead of full-reading the qkv tensor
and calling `_split_fused_qkv_shards`, the builder emits three segmented
`WeightPlanEntry` records for q/k/v.  The temporary `fused_qkv_names`
side-channel is gone, so HunYuan load execution is now a single ordinary
`execute_weight_plan()` call and the fused qkv staging/read volume is visible
to plan summary and scheduler accounting.

### 2026-07-03 Phase 3 stage 2f: DeepSeek conditional stacked specs

DeepSeek now uses the shared declaration primitives for the portions that are
pure data:

- `StackedProjectionMap` tables for gate/up, wk/weights, MHA q/k/v, and MLA
  q_a/kv_a fusion;
- `RoutedExpertPattern` for ordinary routed experts;
- `RoutedProjectionMap` for routed/shared expert projection names.

The DeepSeek-specific target-existence checks remain in the builder-side
wrapper where they belong.  If an MLA fused target is absent, mapping falls
back to the original name; if a shared expert target exists as a normal model
parameter, the shared-expert fallback entries are not emitted.  This follows
the same shape as GLM4's rocm_aiter gate: conditions stay in Python, while the
resulting plan remains concrete data.

### 2026-07-03 Phase 3 stage 2g: Llama4 fused experts folded into WeightPlan

Llama4's remaining composite side path is now ordinary IR data.  The
`Llama4FusedExpertEntry` wrapper and custom dispatch loop are gone;
`Llama4SourcePlan` is a plain `WeightPlan`, and
`load_llama4_weights_from_source()` delegates directly to
`execute_weight_plan()`.

The fused `gate_up_proj` checkpoint tensor emits two segmented
`WeightPlanEntry` records, one for `w1` and one for `w3`.  Each entry reads
only its half of the last dimension into an explicit staging tensor via
`WeightPlanReadSegment`, so the reads stay visible to plan summary and
scheduler accounting.  The fused `down_proj` tensor uses the same ordinary
entry machinery with a local expert slice.  The old tensor transpose in the
custom loader is represented by a serializable `TransformOp("transpose_last_two")`.

With HunYuan, DeepSeek, and Llama4 covered, the known fused/composite routed
paths no longer require non-`WeightPlan` execution contracts.

### 2026-07-03 Phase 3 routed IR host smoke attempt

A host `ready` smoke was attempted after installing system `python3-dev` and
`ninja-build`, using the tiny Qwen3.5 MoE checkpoint and
`uma_odirect_safetensors`.

Loader result was clean:

```text
tiny-random-qwen3.5 MoE
  plan entries: 2034 total, 2017 required, 17 skipped
  expected bytes_read: 0.01 GiB
  actual bytes_read:   0.01 GiB
  expected read amplification: 1.00x
  model load: 0.02 GiB, 1.30 s
```

The end-to-end `ready` phase was stopped before completion for machine safety:
post-load FlashInfer/CUDA JIT compilation spawned many `nvcc` processes and
drove host RAM to `min available=1.50 GiB`, `peak swap=3.16 GiB`, and memory
PSI avg10 `some/full=2.85/2.54`.  Remote RAM stayed idle.  This is outside the
DGX Spark safety envelope, so the remaining ready smokes were not run.

Conclusion: the Phase 3 loader path passed expected/actual byte accounting,
but host `ready` startup with uncached FlashInfer/CUDA JIT is not safe in this
configuration.  Further large-model validation should use `model_load` stop
unless the JIT memory spike is mitigated or caches are safely prebuilt.

After clearing swap, the standard three-model DGX Spark smoke was repeated with
`VLLM_TEST_STOP_AFTER=model_load` to avoid the unsafe post-load JIT phase:

```text
Model                         Expected  Actual   Amplification  Load time
tiny-random-qwen3.5 MoE        0.01GiB  0.01GiB         1.00x      1.55s
PrimeIntellect tiny MoE        1.25GiB  1.25GiB         1.00x      1.49s
Qwen3.6 35B NVFP4             22.23GiB 22.23GiB         1.02x     25.19s
```

All three model-load runs matched scheduled bytes exactly, emitted no
actual-vs-expected amplification warning, kept swap at 0, and kept memory PSI
at 0.  The 35B run peaked at `39.95GiB` used+buff/cache with
`82.53GiB` minimum local available RAM.  This validates the Phase 3 routed IR
loader path for the existing smoke set, while leaving end-to-end `ready`
blocked on the separate FlashInfer/CUDA JIT memory spike.

### 2026-07-03 upstream RFC draft

The upstream-facing RFC draft is now in
`docs/uma-safe-weight-plan-upstream-rfc.md`.  It frames the work as a neutral
`WeightPlan` IR proposal rather than an O_DIRECT-specific feature: motivation,
IR shape, entry/segment/transform responsibilities, read-accounting validation,
incremental adoption, compatibility, open questions, and a first PR sequence.

Remaining validation note for the RFC: the standard three-model smoke does not
exercise real `read_segments` checkpoint bytes.  TeleChat2, HunYuan, or Llama4
should be added to the smoke matrix when a suitable checkpoint is available;
registry and golden tests cover the segment shape in the meantime.

### 2026-07-03 review hardening

The high/medium issues from the branch review are addressed before the RFC
track moves forward:

- Routed and shared-expert plan building now applies explicit skip predicates
  before parsing routed names.  This keeps DeepSeek MTP/nextn checkpoint
  tensors in the auto skipped-entry path instead of resolving out-of-range
  model layers.
- `read_segments` now has an executor-level final guard: segment targets must
  cover the staging tensor exactly once, with no overlaps or unwritten gaps.
  The same validation is used by summary, scheduling, and execution.
- TP output-dim slice inference now fails closed if the private
  `_get_shard_size_mapping()` helper raises, instead of silently falling back
  to a full-tensor read.
- The neutral capability helper is named
  `ExecutorCapability.for_aligned_direct_io()` instead of using the fork-only
  UMA/O_DIRECT name.
- DeepSeek FP8 `indexer.wk` manual two-source dequantization is still a
  model-side postprocess, but its weight and scale reads are represented by a
  supplemental `WeightPlan` for expected-byte accounting.
- `return_success` refusal checks now cover shard-only loader calls as well as
  expert-routed calls.

The first lower-priority follow-up is also closed: O_DIRECT segmented staging
now batches forced memory gates at the entry level.  Individual segment reads
still feed byte counts into the normal interval gate, but the expensive
`/proc/meminfo` and PSI reads no longer run before and after every tiny
segment.  This protects Llama4-style per-row segment plans from millions of
forced gate probes while preserving pre/post safety checks around the whole
staging read.

The O_DIRECT unit coverage now includes one real `_ODirectFile` test on
filesystems that support it.  It verifies window-backed unaligned record reads
and fail-closed short-read handling without faking `pread()`.

The shared memory/PSI gate is now factored into `_uma_memory_gate.py` and used
by both `uma_safetensors` and `uma_odirect_safetensors`.  The loader-specific
labels and error messages remain intact, while `/proc/meminfo`, memory PSI,
swap-limit, and GiB formatting logic have a single implementation and unit
coverage.

Remaining lower-priority review items: possible fused-entry coalescing to
reduce window-cache sweeps and adding a real segment-family checkpoint to the
smoke matrix.

### 2026-07-04 truncated segment fixtures

Before downloading full HunYuan/Llama4/TeleChat2 checkpoints, the segment byte
boundaries are now covered by lightweight real-safetensors fixtures.  These
tests write tiny safetensors files with the same tensor ranks and segment
layout assumptions as the real families, then read them through the real
`_ODirectFile` path instead of a fake source:

- TeleChat2 interleaved `key_value.weight` splits into K and V staging tensors.
- HunYuan interleaved fused QKV uses a `num_heads=4, num_kv_heads=2` fixture so
  the Q/K/V reads cross multiple KV groups, not just one contiguous block.
- Llama4 fused `gate_up_proj` reads the local expert slice into separate w1/w3
  staging tensors.

This is not a substitute for a real checkpoint smoke: it does not validate full
model construction, quantization metadata, or end-to-end `weight_loader`
contracts.  It does catch the highest-risk off-by-one and source/target slice
boundary mistakes before spending time and RAM on multi-hundred-GB checkpoints.
OpenPangu remains outside this specific segment fixture set because its current
UMA path uses stacked/source-slice entries rather than `read_segments`.

### 2026-07-04 Stage A fused-source coalescing

Stage A is implemented without changing the `WeightPlan` IR.  Consecutive
required `read_segments` entries that share the same checkpoint tensor are read
as one source group: each entry still owns its own staging tensor, but the
O_DIRECT source flattens all group segments and reads them in source-offset
order.  After the grouped read completes, the executor dispatches the existing
per-entry `weight_loader` calls in the scheduled entry order, so loader
semantics and loaded-weight accounting remain unchanged.

The read scheduler mirrors the same grouped range order before simulating
O_DIRECT windows, keeping expected and actual byte accounting aligned.  A small
real-safetensors fixture fixes the regression case: q/k/v entries sharing one
fused tensor would previously sweep rows in entry order; with coalescing, the
same values are loaded into the same staging tensors while window loads drop
from the entry-order pattern to the source-order pattern.

This remains an executor scheduling optimization, not a new serialized plan
primitive.  A grouped fused-source IR node is still reserved for a later stage
only if real checkpoint smokes show amplification that cannot be removed by
range scheduling alone.

### 2026-07-04 Hunyuan-A13B-Instruct-GPTQ-Int4 real-checkpoint smoke

The first real quantized routed-MoE checkpoint smoke passed on
`--skip-tokenizer-init model_load` stop mode:

```
plan: 25634 entries, required 25634
expected bytes read: 40.09 GiB
actual bytes read:   40.09 GiB
payload:             39.74 GiB
amplification:       1.01x
model load:          39.61 GiB, 20.46s
local min available: 41.25 GiB
local peak used + buff/cache: 80.94 GiB (used 78.38 GiB + buff/cache 2.56 GiB)
swap: 0.00 GiB, memory PSI: 0.00/0.00, IO PSI: 22.41/22.16
```

Page cache grew only 0.08 GiB (2.48 -> 2.56 GiB) while reading 39.74 GiB of
payload, confirming the O_DIRECT path does not amplify into page cache on this
checkpoint. Two fixes came out of this run:

- The generic loader wrapper's tokenizer preflight fails closed on this
  checkpoint (`vocab_file=None`); weight-structure-only validation runs with
  `--skip-tokenizer-init` instead of a tokenizer fix, since tokenizer wiring is
  out of scope for this loader.
- The GPTQ-quantized checkpoint wraps `layer.mlp.experts` in a MoE runner
  module; the real `weight_loader` lives on `runner.routed_experts`, not the
  runner itself. `hunyuan_v1_uma.py` now resolves through the runner wrapper.
  Targeted `hunyuan_v1` tests: 3 passed.

This closes a major real-checkpoint gap for quantized routed-MoE structure:
GPTQ expert tensors (`qweight`/`qzeros`/`scales`/`g_idx`) are parsed, routed to
the MARLIN WNA16 MoE backend, and loaded through the model WeightSource path
with exact expected/actual byte accounting. It does **not** close the
`read_segments` real-checkpoint gap: this checkpoint used full tensor reads
(`full_reads=25634`, `read_into=0`) rather than segment reads. TeleChat2,
Llama4 fused experts, and any real Hunyuan fused-QKV checkpoint still need
their own segment-path smokes; OpenPangu remains a source-slice/stacked path
smoke target rather than a `read_segments` target.

### 2026-07-04 2node UMA O_DIRECT tensor streaming design

Attempting a second node before the next checkpoint surfaced a structural gap:
vLLM has no network-transfer weight loading path today, so a 2-node run of
`uma_odirect_safetensors` can only reach the second rank's model path over NFS
(`/data/shared`). That defeats the purpose of this loader. NFS-mounted reads on
the non-owning rank go through the kernel page cache like any other filesystem
read, so the safety property this whole IR exists to provide -- bounded,
accounted, O_DIRECT reads that do not balloon into page cache -- is lost on
every rank except whichever one happens to hold the local disk.

This is the same problem the sibling `tukisuwa/llama.cpp` UMA O_DIRECT loader
already solved for RPC nodes: see
`docs/uma-odirect-loader.md` in that fork. The design there fixes payload
ownership to the node with local disk access, reads with O_DIRECT only on that
node, and streams tensor bytes to RPC peers instead of letting peers open the
checkpoint file themselves. vLLM's 2-node case needs the same shape of fix.

**Problem statement**

- `ODirectSafetensorsWeightSource` assumes the safetensors files it opens are
  on local disk with O_DIRECT support. On a second rank whose local disk does
  not hold the checkpoint, the only path today is a shared/NFS mount, which
  reintroduces page-cache amplification on that rank and defeats the memory
  accounting this loader reports.
- Metadata (safetensors headers, i.e. `TensorCatalog`) is small enough that
  reading it over NFS or a shared mount is acceptable. Payload bytes are not.

**Goals**

- Exactly one rank per payload file ever opens that file for payload reads
  (the "payload owner"). All other ranks that need tensors from that file
  receive bytes over an explicit transport instead of opening the file.
- `execute_weight_plan()` and the rest of the executor stay unchanged. Only the
  `WeightSource` implementation differs between the owner rank and remote
  ranks, matching the existing duck-typed surface (`catalog`, `read_full_cpu`,
  `read_slice_cpu`, `read_into_cpu`, `read_segments_into_cpu`,
  `read_segment_group_into_cpu`, `empty_cpu`/`empty_cpu_shape`, `skip`,
  `set_expected_read_summary`, `stats_snapshot`).
- Accounting and fail-closed gates extend across the two ranks instead of
  being purely local: a remote rank that unexpectedly opens the payload file
  should be treated as a fail-closed violation, not a silent fallback path.

**Non-goals for the first design pass**

- No GPU-to-GPU RDMA transport. This is a CPU-side host-memory streaming
  design; GPU placement happens after `execute_weight_plan()` returns, as it
  does today.
- No attempt to unify the streaming transport with vLLM's NCCL process groups.
  NCCL groups are built for collective GPU tensor ops after weights are
  resident; this is a pre-placement, CPU-bytes, point-to-point transfer.
- No change to vLLM's tensor-parallel/pipeline-parallel sharding semantics.
  `WeightPlan` already knows which shard/expert each rank needs; this design
  only changes how the owning rank's bytes reach a non-owning rank's staging
  tensors.

**Proposed shape**

- `RemoteODirectSafetensorsWeightSource`: implements the same surface as
  `ODirectSafetensorsWeightSource` but has no local file handles for payload
  files. Read calls become requests sent to the payload-owner process; the
  owner reads locally with the existing O_DIRECT path and streams the bytes
  back.
- Payload ownership assignment: for this fork's target topology (small, fixed
  node count, checkpoint replicated or partitioned across known local disks),
  ownership can start as static configuration (which rank owns which file
  path) rather than a discovery protocol.
- Metadata distribution: either every rank builds its own `TensorCatalog` from
  a shared/NFS-visible header read (acceptable, KB-scale), or rank 0 builds it
  once and broadcasts the serialized catalog. Keep this pluggable; it is not
  the safety-critical path.
- Transport candidates:
  - A dedicated TCP stream server/client, mirroring the `llama.cpp` UMA
    O_DIRECT RPC design. Easiest to reason about in isolation from vLLM's own
    distributed init, easiest to fail closed on (refuse to serve anything but
    declared payload ranges).
  - `torch.distributed` with a Gloo subgroup for CPU-byte point-to-point
    transfer, kept separate from the NCCL groups vLLM already uses for GPU
    collectives. Reuses process-group bootstrap vLLM already has, but couples
    the streaming path's lifecycle to `torch.distributed` init ordering and
    needs care that it is not confused with NCCL-based paths.
  - Both need an explicit allowlist/path-prefix/capability gate before serving
    any read, matching the fail-closed posture the rest of this loader
    follows.
- Required accounting/safety fields (extending today's single-rank stats):
  `local_direct_bytes_read`, `remote_stream_bytes_sent`,
  `remote_stream_bytes_recv`, `remote_payload_loaded`,
  `local_buff_cache_peak`, `remote_buff_cache_peak`, `local_memory_psi`,
  `remote_memory_psi`. A remote rank opening the payload file directly should
  be detectable and treated as a fail-closed error, not silently tolerated.

**Open questions**

- How does per-rank payload ownership interact with pipeline-parallel layer
  distribution, where a rank's required layers may not align with which node
  physically holds which checkpoint shard?
- How does a remote-rank failure (transport drop, owner rank crash mid-stream)
  propagate as a fail-closed abort to the rest of the run, rather than hanging?
- Where does rank/ownership configuration live -- new `LoadConfig` fields,
  environment variables, or a small topology file -- given this fork's
  intentionally narrow deployment target?
- Should the transport be introduced behind the same `ExecutorCapability`
  mechanism already used to describe O_DIRECT alignment requirements, so a
  future executor can query "can this source stream to a remote rank" the same
  way it queries alignment needs today?

Detailed `RemoteODirectSafetensorsWeightSource` design (message shapes, error
handling, and phased implementation plan) is tracked separately in
`docs/uma-safe-2node-remote-weight-source-design.md`.
