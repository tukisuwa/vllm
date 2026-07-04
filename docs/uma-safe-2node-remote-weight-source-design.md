# 2node UMA O_DIRECT tensor streaming: RemoteWeightSource design

Status: Phase 1 loopback implementation exists for the transport protocol and
`WeightSource` surface. See the "2node UMA O_DIRECT tensor streaming design"
entry in `docs/uma-safe-weight-plan-ir-roadmap.md` (2026-07-04) for the
problem statement and high-level shape this document expands on.

## Problem recap

`ODirectSafetensorsWeightSource` assumes every rank that needs a tensor can
open the safetensors file that holds it, locally, with O_DIRECT. On a 2-node
DGX Spark run this is false for whichever rank does not hold the checkpoint on
local disk: its only path to the file today is a shared/NFS mount, which reads
through the kernel page cache and defeats the memory-safety property this
loader exists to provide. This mirrors a problem the sibling
`tukisuwa/llama.cpp` UMA O_DIRECT loader already solved for RPC nodes by
pinning payload ownership to one node and streaming tensor bytes to peers
instead of letting peers open the file.

## Goal

Let a non-owning rank obtain the exact bytes an owning rank's
`ODirectSafetensorsWeightSource` would have read locally, without ever opening
the payload file itself, while keeping `execute_weight_plan()` and the rest of
the executor unmodified: `RemoteODirectSafetensorsWeightSource` only needs to
satisfy the same duck-typed `WeightSource` surface the executor already calls.

## Non-goals (first pass)

- No GPU-to-GPU RDMA. This is a CPU-host-memory streaming path; GPU placement
  happens after `execute_weight_plan()` returns, same as today.
- No reuse of vLLM's NCCL process groups. Those exist for collective GPU ops
  on resident weights; this transfer happens before weights are resident and
  moves plain CPU bytes point-to-point.
- No change to `WeightPlan` construction, TP/EP shard semantics, or the
  `*_uma.py` declarative specs. A remote rank still builds (or receives) the
  same `WeightPlan` a local rank would; only how bytes are fetched changes.
- No dynamic ownership discovery. Payload ownership is static configuration
  for this fork's fixed, small node topology.

## WeightSource surface to satisfy

This is the actual surface `execute_weight_plan()` and the plan-summary/
schedule helpers call today (verified against
`vllm/model_executor/model_loader/uma_odirect_safetensors_loader.py` and
`vllm/model_executor/model_loader/weight_plan.py`), not the older sketch in
`docs/uma-safe-weight-source-design.md`:

```python
class WeightSource(Protocol):
    catalog: TensorCatalog

    def read_full_cpu(self, name: str) -> torch.Tensor: ...

    def read_slice_cpu(
        self, name: str, source_slices: tuple[slice | int, ...]
    ) -> torch.Tensor: ...

    def read_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
        target_slices: tuple[slice | int, ...] | None = None,
    ) -> None: ...

    def read_segments_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        segments: tuple[WeightPlanReadSegment, ...],
    ) -> None: ...

    def read_segment_group_into_cpu(
        self,
        name: str,
        requests: tuple[tuple[torch.Tensor, tuple[WeightPlanReadSegment, ...]], ...],
    ) -> None: ...

    def empty_cpu(
        self, name: str, *, source_slices: tuple[slice | int, ...] | None = None
    ) -> torch.Tensor: ...

    def empty_cpu_shape(self, name: str, shape: tuple[int, ...]) -> torch.Tensor: ...

    def skip(self, name: str, reason: str) -> None: ...

    def set_expected_read_summary(self, summary: ReadScheduleSummary) -> None: ...

    def stats_snapshot(self) -> dict[str, int | float]: ...
```

`read_segment_group_into_cpu` is the one method `execute_weight_plan()`
probes with `getattr(..., None)` rather than requiring unconditionally (Stage
A coalescing degrades gracefully without it), so a first `RemoteWeightSource`
cut can omit it and fall back to per-entry `read_segments_into_cpu` calls
without breaking correctness -- only losing the coalescing optimization on
the remote path until it is implemented.

## Proposed shape

### Roles

- **Owner rank**: runs an unmodified `ODirectSafetensorsWeightSource` plus a
  request-serving loop. It is the only process that ever opens the payload
  safetensors files.
- **Remote rank**: runs `RemoteODirectSafetensorsWeightSource`, which holds a
  `TensorCatalog` (see Metadata below) and a connection to the owner, and
  turns every `WeightSource` call into a request/response over that
  connection. It never opens a payload file; this is the fail-closed
  invariant the whole design exists to enforce, and should be checked, not
  assumed (see Fail-closed checks below).

### Wire protocol sketch

A minimal request set that maps directly onto the `WeightSource` surface,
rather than a generic byte-range RPC, keeps the owner's fail-closed posture
intact -- it can validate each request against the catalog and the rank's
`WeightPlan` before ever touching the file, the same way
`validate_weight_plan_read_segments` validates shapes before
`empty_cpu_shape` allocates:

```
ReadFullRequest(name)
ReadSliceRequest(name, source_slices)
ReadIntoRequest(name, source_slices, target_slices, dst_shape, dst_dtype)
ReadSegmentsRequest(name, segments, staging_shape, dst_dtype)
ReadSegmentGroupRequest(name, [(segments, staging_shape, dst_dtype), ...])
SkipNotice(name, reason)  # informational, no response needed
```

Responses carry a status (`ok` / `error`) plus, on success, the raw tensor
bytes plus enough metadata (dtype, shape) for the remote side to reconstruct
a `torch.Tensor` without re-deriving it from the catalog, so that a catalog
drift between ranks fails the read instead of silently reshaping wrong bytes.
On `error`, the response carries the same message the owner's local
`ODirectSafetensorsWeightSource` would have raised with, so remote-rank
failures are exactly as legible as local ones -- this loader's whole review
history has been about surfacing errors before payload dispatch rather than
translating them into something vaguer at a process boundary.

### Metadata (`TensorCatalog`) distribution

Two acceptable options, deliberately left pluggable since this is not the
safety-critical path (headers are KB-scale, not the multi-GB payload path
O_DIRECT protects):

1. Every rank reads the safetensors headers itself over whatever shared
   filesystem is available (NFS is fine here -- only payload bytes are
   forbidden over NFS, not headers).
2. The owner rank builds `TensorCatalog` once and serializes it to remote
   ranks over the same connection used for tensor requests, before any
   `WeightPlan` construction. This avoids a shared-filesystem dependency
   entirely and is the better long-term default once the transport exists,
   but (1) is enough to unblock the first working version.

### Transport candidates

1. **Dedicated TCP stream server/client**, matching the `tukisuwa/llama.cpp`
   UMA O_DIRECT RPC design (see that fork's `docs/uma-odirect-loader.md`).
   Advantages: fully decoupled from vLLM's own distributed bootstrap and
   lifecycle, easiest to reason about for a fail-closed request allowlist
   (the server can refuse anything outside a declared path prefix / catalog
   before ever reading a byte), easiest to add a standalone integration test
   for. This is the recommended primary path.
2. **`torch.distributed` Gloo subgroup**, separate from the NCCL group(s)
   vLLM already creates for GPU collectives. Advantage: reuses process-group
   bootstrap machinery already present. Disadvantages: couples the streaming
   path's lifecycle to `torch.distributed` init ordering (which happens
   relatively late and is oriented around GPU collective setup, not
   pre-load CPU byte transfer), and needs explicit care so it is never
   mistaken for -- or accidentally shares state with -- the NCCL-based
   collective path. Worth prototyping only if the dedicated-server path turns
   out to fight vLLM's process launch model.

Either transport needs, before serving any read:

- an explicit allowlist of path prefixes the owner is willing to serve reads
  from (never "whatever the request says"),
- a capability/handshake step so a remote rank cannot silently connect to an
  unrelated owner process and get served arbitrary local files,
- the same fail-closed posture as the rest of this loader: malformed
  requests, catalog mismatches, or shape mismatches reject the request rather
  than reading a "close enough" range.

### Fail-closed checks specific to the 2-node path

- The owner should refuse to serve any path outside the checkpoint directory
  it was configured with.
- A remote rank's `RemoteODirectSafetensorsWeightSource` should assert (or a
  wrapping test harness should verify) that the process never opens the
  payload file descriptor -- for example by tracking `open()` calls against
  the known checkpoint path during a test run, the same spirit as the
  existing real-`_ODirectFile` tests that exercise the actual read path
  instead of a fake one.
- Transport-level failures (dropped connection, owner crash mid-stream)
  should raise on the remote rank rather than hang or silently retry into a
  degraded state; propagating that failure so the whole multi-rank run aborts
  is an open question (see below) but "hang forever" and "silently continue
  with partial data" are both unacceptable outcomes.

### Accounting fields

Extending `_SourceReadStats`/`stats_snapshot()` for the 2-rank case:

- `local_direct_bytes_read` -- bytes the owner rank read from disk via
  O_DIRECT (today's existing `bytes_read`, renamed/aliased for clarity once a
  remote counterpart exists).
- `remote_stream_bytes_sent` / `remote_stream_bytes_recv` -- bytes pushed
  onto and pulled off the transport, tracked on the owner and remote side
  respectively; these should match bytes_read on the owner side within the
  same 1.10x drift threshold `log_stats()` already applies locally.
- `remote_payload_loaded` -- set of tensor names the remote rank has
  successfully materialized, mirroring the `loaded: set[str]` return value
  `execute_weight_plan()` already produces locally.
- `local_buff_cache_peak` / `remote_buff_cache_peak` -- reuse the existing
  buff/cache sampling used in checkpoint smokes (see the
  Hunyuan-A13B-Instruct-GPTQ-Int4 smoke entry in the roadmap) on both ranks,
  since page-cache growth on the *remote* rank from an accidental NFS/open
  fallback is exactly the failure mode this design prevents.
- `local_memory_psi` / `remote_memory_psi` -- both ranks' PSI gate readings,
  since a stall on either rank should be visible, not just the owner's.

## Migration plan

1. **Static single-owner, TCP transport, per-entry requests.** No
   `read_segment_group_into_cpu` on the remote side yet (falls back to
   per-entry `read_segments_into_cpu`, matching the existing optional-method
   pattern in `execute_weight_plan()`). Metadata distributed via shared
   filesystem header reads (option 1 above) to avoid building catalog
   serialization on day one.
2. **Catalog broadcast**, replacing the shared-filesystem metadata read with
   owner-serialized `TensorCatalog` distribution, removing the NFS dependency
   entirely (including for headers).
3. **Segment-group requests**, adding `ReadSegmentGroupRequest` so Stage A
   coalescing benefits extend across the remote path instead of only the
   local one.
4. **Real 2-node checkpoint smoke**, following the same "small fixture first,
   then real checkpoint" progression this loader has used throughout: a
  loopback-transport unit test (owner and remote in the same process/host)
  before an actual 2-node DGX Spark run, mirroring how the segment fixtures
  validated boundary math before the Hunyuan real-checkpoint smoke.

## Phase 1 implementation note (2026-07-04)

The first implementation cut adds:

- `RemoteODirectSafetensorsWeightSourceServer`: a dedicated TCP owner process
  wrapper around an existing `ODirectSafetensorsWeightSource`. Requests are
  WeightSource-shaped operations (`read_full`, `read_slice`, `read_segments`,
  `skip`, `stats_snapshot`, `close_files`) rather than arbitrary byte ranges,
  so owner-side catalog/source validation remains in the existing local path.
- Handler threads serialize all calls into the wrapped
  `ODirectSafetensorsWeightSource` with a lock. The local source owns a single
  mutable O_DIRECT file/window cache and shared counters, so Phase 1 treats
  the owner as a correctness-first synchronous service rather than allowing
  concurrent reads to race that state.
- The wire frame is a length-prefixed JSON header plus an optional raw tensor
  payload. The server parses authentication and operation fields from JSON
  before touching payload bytes; it does not deserialize pickle or another
  executable object format from the peer. Request frames allow no payload;
  response payload size is bounded by the remote rank's catalog-derived
  expected tensor size, and headers have a fixed small upper bound.
- Auth tokens are compared with constant-time comparison before any owner read
  is attempted.
- `RemoteODirectSafetensorsWeightSource`: a remote-rank source with a local
  `TensorCatalog` and no payload file handles. It implements
  `read_full_cpu`, `read_slice_cpu`, `read_into_cpu`,
  `read_segments_into_cpu`, `empty_cpu`, `empty_cpu_shape`, `skip`,
  `set_expected_read_summary`, and `stats_snapshot`.
- No remote `read_segment_group_into_cpu` yet by design. The executor's
  optional-method probe therefore falls back to per-entry segment requests,
  matching Migration step 1.
- A loopback unit test using a real safetensors file and the actual
  `_ODirectFile` path on the owner side. The test validates full, sliced, and
  segmented reads; confirms the optional group method is absent; checks remote
  stream accounting; verifies auth failure is rejected before payload serving;
  rejects oversized frame headers/payloads before allocation; and confirms
  concurrent TCP handler threads serialize access to the shared owner source.
- Initial vLLM loader wiring via environment variables:
  - `VLLM_UMA_ODIRECT_REMOTE_ROLE=owner|remote`
  - `VLLM_UMA_ODIRECT_REMOTE_HOST`
  - `VLLM_UMA_ODIRECT_REMOTE_PORT`
  - `VLLM_UMA_ODIRECT_REMOTE_TOKEN`
  - `VLLM_UMA_ODIRECT_REMOTE_TIMEOUT_SECONDS` (optional, default `30`)

The owner role creates the normal local `ODirectSafetensorsWeightSource`,
starts the TCP owner server, and keeps the server/source alive on the loader
instance after local rank load returns so remote ranks can still fetch payload
bytes. The remote role builds a local `TensorCatalog` from safetensors headers
only, then uses `RemoteODirectSafetensorsWeightSource` for payload reads. This
preserves the Phase 1 assumption that shared/NFS metadata reads are acceptable
while payload bytes must come from the owner transport.

The env wiring is still a single-owner-process contract. A launch that starts
multiple owner-role worker processes on the same node with the same host/port
will fail to bind; rank-aware owner election/topology files are intentionally
left for the next harness step rather than hidden inside this first cut.

This implementation is intentionally not wired into distributed vLLM launch
automation yet. The next step is a small 2-process harness that starts one
rank with the owner env and one rank with the remote env, verifies that the
remote rank does not open the safetensors payload path, and records
owner/remote `buff/cache`, swap, and PSI before trying a full two-node vLLM
serve.

## Phase 1 two-node smoke (2026-07-04)

The first real 2-node check used the DGX Spark QSFP link with local
`dgx-spark2` as owner (`192.168.100.11`) and `dgx-spark1` as remote
(`192.168.100.10`):

1. Raw TCP reachability was verified without vLLM by accepting a remote
   connection on the owner address and echoing a payload.
2. A Python RemoteWeightSource harness streamed every tensor from
   `tiny-random-qwen3.5-moe` over TCP. The remote side received
   `9,805,008` tensor-payload bytes for `2034` tensors. Owner O_DIRECT stats
   reported `bytes_read=19,665,712`, `bytes_copied=9,805,008`,
   `direct_reads=2408`, `window_loads=274`, and `window_hits=2016`.
3. A real vLLM model-load smoke then loaded
   `tiny-random-qwen3-moe` on `dgx-spark1` with
   `VLLM_UMA_ODIRECT_REMOTE_ROLE=remote` while the owner TCP source served
   payload bytes from local `dgx-spark2`.

The vLLM smoke selected `Qwen3MoeForCausalLM`, built a 46-entry plan, received
`0.02 GiB` of remote tensor payload, and reported `1.00x` read amplification.
Owner direct-read stats reported `bytes_read=0.04 GiB`,
`bytes_copied=0.02 GiB`, `direct_reads=4873`, `window_loads=9`, and
`window_hits=14`.

Remote memory behavior stayed consistent with the design goal: `dgx-spark1`
`buff/cache` rose from `3.205 GiB` to `3.424 GiB` (`+0.220 GiB`), swap stayed
`0.00 GiB`, and memory PSI stayed `0.00/0.00`. Owner/local `buff/cache` rose
from `3.085 GiB` to `3.147 GiB` (`+0.062 GiB`) with swap and memory PSI also
at zero. This is not yet a large-checkpoint proof, but it confirms the
transport, env wiring, owner lifetime, and remote-source integration in a
real vLLM model-load path.

The remote env names are now registered in `vllm.envs` so vLLM's unknown-env
checker does not warn for this fork-specific transport configuration.

## Phase 1a persistent connection (2026-07-04)

The first Qwen35B 2-node smoke proved the remote path but also exposed the
Phase 1 performance limit: `124,306` full-read entries caused `124,306`
independent synchronous TCP connections.  The owner measured only `7.37s` of
O_DIRECT read time, while total model-load time was `201.25s`; the bottleneck
was request granularity/connection overhead, not disk or network bandwidth.

Phase 1a keeps the existing narrow WeightSource-shaped protocol and changes
only the connection lifetime:

- the owner handler now accepts multiple length-prefixed request frames on one
  TCP connection until the remote side closes it;
- owner source access remains protected by the same coarse `_source_lock`, so
  the mutable single-window O_DIRECT state is still serialized;
- the remote source keeps one socket and serializes request/response pairs
  with a lock, preserving correctness if multiple loader threads touch the
  same source;
- failed requests close the remote socket so a rejected/auth-failed connection
  is not reused;
- tests assert that repeated loopback reads and concurrent reads share one
  accepted owner connection while still keeping owner source calls serialized.

This is intentionally an isolated measurement step before adding batch-read
RPCs.  The next Qwen35B remote smoke should be compared against the
`201.25s` pre-persistent baseline to decide how much of the unexplained time
was TCP connect overhead versus JSON/tensor reconstruction/dispatch overhead.

Measured outcome on the same Qwen35B checkpoint:

- owner accepted connections: `1`
- model load: `143.982355s`
- pre-persistent baseline: `201.249922s`
- improvement: `57.27s` (`28.5%`)
- payload and owner read accounting stayed unchanged: remote stream
  `21.73 GiB`, owner `bytes_read=22.23 GiB`, `direct_reads=407`,
  `window_loads=163`, `window_hits=124304`
- remote `buff/cache` first-to-peak delta stayed small at `+0.200 GiB`
- swap and memory PSI stayed zero on both nodes

Persistent connection therefore removes a substantial TCP setup cost, but the
remaining gap is still large.  The next optimization should batch many
WeightPlan entries into one request, or otherwise pass the read schedule into
the transport, so the remote path does not pay JSON parsing, frame dispatch,
and tensor reconstruction overhead `124,306` times for Qwen35B.

## Open questions

- How does per-rank payload ownership interact with pipeline-parallel layer
  distribution, where a rank's required layers may not align with which node
  physically holds which checkpoint shard? A static ownership map may need to
  be keyed by (file, byte range) rather than by rank alone if a single
  checkpoint file's tensors end up needed by ranks on both nodes.
- Where should ownership/topology configuration live -- new `LoadConfig`
  fields, environment variables, or a small topology file? Given this fork's
  intentionally narrow deployment target, the simplest option that avoids
  inventing a discovery protocol is preferred.
- Should `ExecutorCapability` (already used to describe O_DIRECT alignment
  requirements) grow a capability describing "this source can stream to a
  remote rank," so a future executor can query it the same way it queries
  alignment needs today, instead of the remote-vs-local distinction being an
  implicit property of which class was constructed?
- How does a remote-rank or owner-rank failure abort the *other* rank's load
  rather than hanging -- does this need to hook into vLLM's existing
  multi-process launch/teardown, or can it stay fully local to this loader's
  transport?
