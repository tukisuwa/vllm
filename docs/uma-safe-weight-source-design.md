# UMA-safe WeightSource / WeightPlan design

Date: 2026-07-02

This document describes a deeper loader design for UMA-safe model loading in
vLLM.  It is based on the current `uma_odirect_safetensors` experiments and on
the limitations found while testing Qwen3.6-35B-A3B on DGX Spark.

## Problem

vLLM's current weight loading contract is mostly:

```text
storage loader -> Iterable[(checkpoint_name, full CPU torch.Tensor)]
               -> model.load_weights()
               -> model-specific shard/fuse/quant/MoE placement
```

This is simple and works well for many dGPU deployments, but it is the wrong
abstraction for UMA safety.

The loader must often materialize a complete CPU tensor before the model code
decides whether the tensor is:

- needed at all
- only partly needed for TP/PP/EP
- one shard of a fused parameter such as QKV or gate/up
- a routed expert that is not local to this rank
- a quantized payload tensor, scale tensor, or auxiliary metadata tensor
- a tensor that should be skipped because embeddings are tied or the layer is
  absent on this pipeline rank

For UMA systems, that ordering is dangerous.  CPU RAM, page cache, staging
buffers, CUDA allocations, and device-visible memory all share the same physical
pool.  The loader needs to avoid reading and staging data that the model will
later discard or slice.

The current `direct_per_expert_moe` optimization proves the issue: it improves
Qwen MoE loading by bypassing the normal full-tensor iterator for per-expert
weights, but it does so by teaching the loader about a specific MoE checkpoint
layout.  That is a local optimization, not a sound general design.

## Current relevant code

Current loader entry points:

- `vllm/model_executor/model_loader/base_loader.py`
  - `BaseModelLoader.load_model()`
  - initializes the model and calls `self.load_weights(model, model_config)`
- `vllm/model_executor/model_loader/__init__.py`
  - maps `load_format` to a `BaseModelLoader`
- `vllm/model_executor/model_loader/uma_odirect_safetensors_loader.py`
  - current fail-closed local O_DIRECT safetensors loader
  - currently yields `(name, CPU tensor)` for the base path
  - optionally uses `direct_per_expert_moe` for Qwen-like routed expert tensors

Current model-side loading:

- `vllm/model_executor/models/utils.py`
  - `AutoWeightsLoader`
  - recursively routes `(name, tensor)` to parameters/modules
  - applies mapper, skip prefixes/substrings, and `param.weight_loader`
- Representative model implementations:
  - `vllm/model_executor/models/llama.py`
  - `vllm/model_executor/models/qwen3.py`
  - `vllm/model_executor/models/qwen3_moe.py`
  - `vllm/model_executor/models/deepseek_v2.py`
  - `vllm/model_executor/models/granitemoe*.py`

The key design issue is that model-specific placement decisions live after full
CPU tensor materialization.

## Goal

Move weight loading from a push iterator of full tensors to a pull-based plan:

```text
safetensors metadata
  -> TensorCatalog
  -> model-specific WeightPlan
  -> UMA-safe WeightExecutor
  -> only required byte ranges are read and placed
```

The loader should not know whether a tensor is Qwen, DeepSeek, Mixtral, Granite,
or Llama.  The model should describe what it needs; the loader should provide
safe I/O and execution primitives.

## Non-goals

- Do not replace all vLLM model `load_weights()` implementations at once.
- Do not require every model to implement the new API immediately.
- Do not weaken the existing fail-closed behavior of `uma_odirect_safetensors`.
- Do not make UMA-safe loading depend on Docker, Run:ai, InstantTensor, or
  model-specific external services.

## Proposed abstractions

### TensorCatalog

Metadata-only view of checkpoint tensors.

```python
@dataclass(frozen=True)
class TensorMeta:
    name: str
    file_path: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    data_offset: int
    nbytes: int

class TensorCatalog:
    def names(self) -> Iterable[str]: ...
    def get(self, name: str) -> TensorMeta: ...
    def has(self, name: str) -> bool: ...
```

Properties:

- reads safetensors metadata only
- validates duplicate names, ranges, dtype, shape, overlap
- performs no tensor payload read
- has no model-specific logic

The current `_read_records()` in `uma_odirect_safetensors_loader.py` is already
close to this and should be split out.

### WeightSource

I/O primitive provider for a catalog.

```python
class WeightSource:
    catalog: TensorCatalog

    def read_full_cpu(self, name: str) -> torch.Tensor: ...

    def read_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
        target_slices: tuple[slice | int, ...] | None = None,
    ) -> None: ...

    def read_slice_cpu(
        self,
        name: str,
        source_slices: tuple[slice | int, ...],
    ) -> torch.Tensor: ...

    def skip(self, name: str, reason: str) -> None: ...
```

Initial implementation can be conservative:

- only contiguous full-tensor reads are mandatory
- non-contiguous/sliced reads may initially read the minimal enclosing range or
  fall back to a CPU tensor slice if safety remains bounded
- O_DIRECT, PSI gate, swap gate, min-available gate, progress, and cancellation
  belong here

Longer-term implementation should support:

- direct contiguous byte-range reads
- row/column shard reads for TP
- local expert-only reads for EP
- destination-buffer reads when a parameter exposes a safe CPU/device staging
  target

### WeightPlan

Model-side declaration of required checkpoint reads and placements.

```python
@dataclass(frozen=True)
class WeightPlanEntry:
    checkpoint_name: str
    target_name: str
    required: bool = True
    source_slices: tuple[slice | int, ...] | None = None
    target_slices: tuple[slice | int, ...] | None = None
    shard_id: str | int | None = None
    expert_id: int | None = None
    transform: str | None = None
    loader_kind: str = "default"
```

The plan is built by the model, because only the model knows:

- tied embeddings
- pipeline-missing layers
- tensor-parallel shard boundaries
- fused parameter layout
- MoE expert mapping
- quantization auxiliary tensors
- which missing/unexpected tensors are acceptable

Suggested model hook:

```python
class SupportsWeightSource:
    def build_weight_plan(self, catalog: TensorCatalog) -> WeightPlan: ...

    def load_weights_from_source(
        self,
        source: WeightSource,
        plan: WeightPlan,
    ) -> set[str]: ...
```

The hook should be optional.  If a model does not implement it, the loader falls
back to the existing safe `(name, CPU tensor)` iterator path.

### WeightExecutor

Executes a model plan through a source.

Responsibilities:

- enforce fail-closed policy
- decide when full CPU tensor fallback is allowed
- run gates before allocation/read/placement
- route to `param.weight_loader`
- preserve existing vLLM post-load processing
- collect timing and byte counters

Conceptual execution:

```python
for entry in plan.entries:
    if not entry.required:
        source.skip(entry.checkpoint_name, "not required")
        continue

    meta = source.catalog.get(entry.checkpoint_name)
    param = resolve_target_param(model, entry.target_name)

    if entry.can_read_into_param_cpu_view:
        source.read_into_cpu(entry.checkpoint_name, cpu_view, ...)
        param.weight_loader(param, cpu_view, ...)
    elif entry.source_slices is not None:
        tensor = source.read_slice_cpu(entry.checkpoint_name, entry.source_slices)
        param.weight_loader(param, tensor, ...)
    else:
        tensor = source.read_full_cpu(entry.checkpoint_name)
        param.weight_loader(param, tensor, ...)
```

The first version can still call `weight_loader` with CPU tensors; the important
change is that the decision to skip/slice/fuse is made before payload read.

## Migration plan

### Phase 0: Keep current safe base path

Status: current `uma_odirect_safetensors`.

- O_DIRECT local safetensors reader
- metadata validation
- no mmap/eager/prefetch
- no buffered fallback
- memory/swap/PSI gates
- model-load-only tests on Qwen3.6-35B-A3B

### Phase 1: Extract TensorCatalog and WeightSource

Status: started on `uma-safe-weight-source`.

Implemented so far:

- `TensorMeta` / `TensorCatalog`
  - metadata-only safetensors catalog
  - duplicate name, dtype, shape, range, and overlap validation
  - lookup by name and stable file/offset ordering
- `ODirectSafetensorsWeightSource`
  - builds the catalog before payload reads
  - exposes `source.catalog`
  - keeps the compatibility full-tensor iterator
- optional model hook dispatch:
  - `build_weight_plan(catalog)`
  - `load_weights_from_source(source, plan)`
  - unsupported models still use `model.load_weights(iterator)`
- `source.read_full_cpu(name)`
  - uses the same O_DIRECT full-record read helper as the iterator path
  - keeps allocation/read memory gates on the pull path
- `source.read_slice_cpu(name, source_slices)`
  - supports only row-major contiguous slices
  - rejects stepped or non-contiguous slices instead of full-tensor fallback
  - records only the requested payload bytes in source stats
- source-level stats for model hook reads
  - files opened
  - tensors read/skipped
  - direct reads/window reads
  - bytes read/copied/payload
  - gate/allocation/read timing
- stable `source.stats_snapshot()` for tests and diagnostics
- minimal `WeightPlanEntry`, `WeightPlan`, and `execute_weight_plan(...)`
  - supports required/skipped entries
  - supports full CPU reads and contiguous source slices
  - routes tensors to a target parameter's `weight_loader`
- `build_auto_weight_plan_from_catalog(...)`
  - performs prefix/substr skips without reading payload bytes
  - applies `WeightsMapper`-style name and shard mapping before payload reads
  - can mark known ignorable missing suffixes such as `.bias`
  - keeps missing target detection fail-closed in the executor
- `build_auto_weight_plan_for_module(...)`
  - adds AutoWeightsLoader-compatible quant cache-scale mapper composition
  - adds module quant-config ignored suffixes
  - keeps model hooks small and consistent across dense model families
- `execute_weight_plan(...)`
  - resolves target parameters before payload reads
  - skips `ignore_missing` entries before payload reads
  - fails closed before payload reads for unexpected missing targets
- tests for catalog lookup, source construction, full CPU reads, contiguous
  slice reads, source stats, optional model hook dispatch, name-only plan
  building, and basic plan execution

Remaining refactor without behavioral change:

- move more `_ODirectFile` orchestration into `ODirectSafetensorsWeightSource`
- add `source.read_into_cpu(...)` for destination-buffer reads
- expand plan execution only where generic semantics are clear; model-specific
  transforms should stay in model-side plan code

Benefit:

- separates safe I/O from current iterator compatibility
- makes tests easier
- creates a stable API for plan experiments

### Phase 2: Add optional model hook

Status: initial hook and minimal generic executor implemented.

In `UmaODirectSafetensorsModelLoader.load_weights()`:

```python
if hasattr(model, "build_weight_plan") and hasattr(model, "load_weights_from_source"):
    catalog = source.catalog
    plan = model.build_weight_plan(catalog)
    model.load_weights_from_source(source, plan)
else:
    model.load_weights(source.iter_full_tensors())
```

Requirements:

- must be opt-in: implemented
- must log whether the model-source path or iterator path was used: implemented
- must fail closed if the model requested source loading but the source cannot
  provide a required operation: implemented for missing paired hooks,
  unsupported slices, missing targets, and missing `weight_loader`

Remaining:

- implement the first real model-side `build_weight_plan(...)`
- decide where generic helpers should live once more than one model uses them
- add broader tests around actual vLLM parameter loaders

### Phase 3: Dense model prototype

Status: started with `Qwen3ForCausalLM` and `LlamaForCausalLM`.

Start with Llama/Qwen dense, not MoE.

Why:

- fewer model-specific edge cases
- tests TP slicing and fused QKV/gate-up without expert mapping
- proves the plan interface without relying on Qwen35 MoE special cases

Target behavior:

- plan skips tied `lm_head`: implemented for Qwen3 and Llama
- plan skips rotary/cache tensors: implemented through shared auto-plan helper
- plan maps q/k/v into qkv placement before read: implemented through
  `hf_to_vllm_mapper` and `shard_id`
- plan maps gate/up into gate_up placement before read: implemented for Llama
- plan includes AutoWeightsLoader-compatible quant cache-scale mapper and
  ignored suffix handling for Qwen3 and Llama
- plan can read only local TP shard where possible: not implemented yet

Current limitations:

- The first Qwen3 hook still reads each required checkpoint tensor as a CPU
  tensor, then delegates to existing parameter `weight_loader`.
- It proves model-side planning and skip/map decisions before payload read, but
  does not yet perform TP-local source slicing for fused/parallel parameters.
- Qwen3 MoE now has a first model-side source hook, but other MoE families and
  full removal of the older compatibility optimization remain Phase 4/5 work.

### Phase 4: Qwen MoE prototype

Status: started with `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`, and
`Qwen3_5MoeForCausalLM`.

Replace `direct_per_expert_moe` with a model-side plan for Qwen MoE.

Target behavior:

- model plan declares local expert reads: implemented for Qwen-family MoE routed
  expert tensors
- non-local experts are skipped before payload read: implemented for Qwen-family
  MoE
- gate/up/down fused placements are described by plan entries: implemented for
  Qwen-family MoE per-expert gate/up/down naming
- quant payload and scale suffixes are handled by the model plan, not by loader:
  implemented for suffix-based Qwen-family MoE routed expert targets

This should preserve the observed Qwen35B performance while moving the
model-specific knowledge out of the loader.

Remaining work:

- remove or deprecate the older loader-side `direct_per_expert_moe`
  compatibility path once the model hook is validated on a real Qwen3 MoE load
- validate Qwen3.5/Qwen3Next MoE on real checkpoints; they share the model-side
  helper but have not been real-loaded yet
- add real-load verification that the model hook matches the previous
  Qwen35B performance and memory behavior

### Phase 5: Other MoE families

Add model-side plans only where needed:

- Mixtral
- DeepSeek V2/V3 style MoE
- Granite MoE variants

Each model family should implement its own plan builder instead of adding
loader-side conditionals.

## Compatibility matrix

Expected current behavior:

| Model type | Base `uma_odirect_safetensors` | Direct plan path |
| --- | --- | --- |
| Dense safetensors | Should work if normal vLLM load works | Phase 3 |
| Sharded dense safetensors | Should work if no duplicate names | Phase 3 |
| Qwen routed MoE | Works; current direct optimization exists | Phase 4 |
| Non-Qwen routed MoE | Base path should work if normal vLLM load works | Phase 5 |
| Non-safetensors | Not supported | Not supported |
| Remote HF path | Not supported by UMA-safe loader | Not supported |
| mmap/eager/prefetch | Rejected | Rejected |
| symlinked weights | Rejected | Rejected |

## Testing strategy

Static tests:

- malformed safetensors metadata
- duplicate names across shards
- symlink rejection
- overlapping ranges
- unsupported dtype
- zero-sized tensors
- `TensorCatalog` lookup and stable ordering

Model audit:

- `scripts/audit-vllm-uma-odirect-model.py <model-dir>`
- should be run before adding a model to the compatibility list

Runtime tests:

- small dense model, single shard
- small dense model, multiple shards
- Qwen35B current NVFP4 model
- one non-Qwen MoE with direct path disabled
- later: one non-Qwen MoE with a model-side plan

Runtime metrics to record:

- model load time
- peak used
- peak buff/cache
- peak used + buff/cache
- min available
- swap
- memory PSI
- IO PSI
- bytes read
- bytes copied
- skipped bytes
- full-tensor fallback bytes
- slice/direct-read bytes

## Design rules

- The loader must not contain model-family branches such as `if qwen` or
  `if deepseek`.
- The model must not perform raw file I/O.
- A requested safe source operation must fail closed if unsupported.
- Buffered fallback must be explicit and off by default.
- Skipped tensors should be skipped before payload read.
- Non-local expert tensors should be skipped before payload read where the model
  plan can determine locality.
- `read_full_cpu()` remains only for compatibility and should be measured.
- All source operations must honor the same memory/swap/PSI gates.

## Open questions

- How should transforms be represented?
  - string enum for common transforms
  - callable object
  - delegate entirely to `param.weight_loader`
- Can we expose destination CPU views for direct `read_into` without violating
  PyTorch storage assumptions?
- Which tensor slices are contiguous enough to read directly from safetensors
  without reading an enclosing large range?
- How should online quantization and `uses_meta_device` models integrate?
- Should this live only in UMA loader first, or become a generic vLLM loading
  interface after the prototype?

## Near-term recommendation

Do not add more model-specific direct paths to
`uma_odirect_safetensors_loader.py`.

Instead:

1. Refactor current loader into `TensorCatalog` + `WeightSource`.
2. Keep the existing iterator path for safety and compatibility.
3. Add an optional model hook for `WeightPlan`.
4. Prototype dense Qwen/Llama.
5. Move the current Qwen MoE direct optimization into a model-side plan.
   Qwen3 MoE is started; validation and cleanup remain.

This addresses the root issue: new models should add model-side placement
plans, not loader-side special cases.
