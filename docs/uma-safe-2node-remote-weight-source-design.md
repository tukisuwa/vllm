# 2node UMA O_DIRECT tensor streaming: RemoteWeightSource design

Status: Phase 1 is implemented and validated through real 2-node model-load
smokes. The path now has owner-broadcast `TensorCatalog` metadata, persistent
TCP connections, batched full/sliced reads, batched segmented reads, and an
eager owner-capability handshake for read-accounting correctness. See the
2026-07-04 entries in
`docs/uma-safe-weight-plan-ir-roadmap.md` for measurements and artifacts.

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
- No true distributed loading semantics. The remote source only changes where
  payload bytes are read from; it does not decide TP/PP placement, elect
  owners per rank, or integrate with vLLM's distributed process lifecycle.

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
A coalescing degrades gracefully without it). The Phase 1 remote source still
does not expose that method directly, but B2 preserves the same optimization
inside batched `read_many` requests: the owner groups segmented batch items by
`checkpoint_name` and calls the local source's group reader before returning
one tensor per request item.

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

The owner rank builds `TensorCatalog` once from local safetensors headers and
serializes it to remote ranks over the same authenticated TCP connection used
for tensor requests, before any remote `WeightPlan` construction.  This keeps
remote ranks from enumerating the model directory or reading safetensors
headers through NFS/shared storage.  Catalog responses are transported as a
bounded JSON payload rather than a large frame header, so large MoE catalogs
do not bypass the frame-size guard.

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

Completed Phase 1 work:

1. **Static single-owner TCP transport.** The owner wraps a local
   `ODirectSafetensorsWeightSource`; remote ranks fetch payload bytes through
   a WeightSource-shaped TCP protocol and do not open payload files.
2. **Catalog broadcast.** Remote ranks fetch owner-serialized `TensorCatalog`
   metadata over the same authenticated TCP transport and do not need a local
   or NFS-visible safetensors directory for headers.
3. **Persistent connection.** One remote source keeps one TCP connection open
   for repeated request/response frames, avoiding per-entry TCP setup.
4. **Bounded `read_many` batches for full/sliced entries.** Qwen35B remote
   request count drops from one request per tensor to bounded payload batches.
5. **Eager owner capability handshake.** Remote scheduling uses the owner's
   actual O_DIRECT chunk/window/alignment values, so expected and actual read
   stats match.
6. **Bounded segmented batches.** TeleChat2-style `read_segments` entries are
   batched, owner-validated, and grouped by source tensor so local Stage-A
   read-window reuse is preserved across the network path.
7. **Real 2-node checkpoint smokes.** Qwen35B and TeleChat2-35B have exercised
   large full-read and segmented-read payloads over the DGX Spark QSFP link.

Remaining migration steps:

1. **Rank-aware topology.** Add explicit owner election/port assignment for
   launches with more than one owner-capable worker on a node.
2. **Distributed lifecycle integration.** Ensure owner or remote failure
   aborts the peer ranks through vLLM's multiprocess launcher instead of
   relying only on socket timeouts.
3. **True TP/PP placement validation.** Exercise the remote source in a real
   distributed topology where ranks own different model shards/layers, not
   just a single remote payload consumer.

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
  `set_expected_read_summary`, and `stats_snapshot`. If no catalog is passed
  to the constructor, it requests the owner's serialized `TensorCatalog`
  before capability negotiation and read scheduling.
- No remote `read_segment_group_into_cpu` method is exposed directly by
  design. B2 handles grouped segmented reads inside `read_many` instead, so
  the executor still sees the same optional-method shape while owner-side
  window reuse is preserved for batched segment entries.
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
  - `VLLM_UMA_ODIRECT_REMOTE_PORT_OFFSET` (optional, added to the base port)
  - `VLLM_UMA_ODIRECT_REMOTE_TOKEN`
  - `VLLM_UMA_ODIRECT_REMOTE_TIMEOUT_SECONDS` (optional, default `30`)

The owner role creates the normal local `ODirectSafetensorsWeightSource`,
starts the TCP owner server, and keeps the server/source alive on the loader
instance after local rank load returns so remote ranks can still fetch catalog
metadata and payload bytes. The remote role no longer calls `_prepare_files()`
or reads safetensors headers from the configured model path; it fetches the
owner catalog first, then uses `RemoteODirectSafetensorsWeightSource` for
payload reads.

The env wiring is still a manually configured topology contract. A launch that
starts multiple owner-role worker processes on the same node must assign a
distinct resolved port per owner. `VLLM_UMA_ODIRECT_REMOTE_PORT_OFFSET` lets a
launcher use one base port and add a rank/local-rank-derived offset, but owner
election and topology-file generation remain outside this loader.

This implementation is intentionally not wired into distributed vLLM launch
automation yet. Manual owner/remote env wiring has been validated, but
rank-aware launch integration, port assignment, and peer-failure propagation
remain outside Phase 1.

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

## Phase 1b batch/schedule-aware transport design (2026-07-04)

The persistent Qwen35B result narrows the remaining bottleneck: connection
setup is no longer dominant, but the remote path still performs `124,306`
serialized WeightSource requests.  The next step should reduce request count
without changing model-side WeightPlan semantics.

### Goals

- Keep the remote rank from opening safetensors payload files.
- Preserve fail-closed owner-side catalog validation; remote requests must
  still name tensors and declared slices/segments, not arbitrary file ranges.
- Keep executor dispatch order unchanged.  Weight loaders should still see
  tensors in the scheduled plan order.
- Bound owner and remote staging memory with an explicit batch payload limit.
- Make expected read accounting use the owner source's actual O_DIRECT
  `chunk_size`, `window_size`, and `alignment`, closing the Qwen35B
  `22.86 GiB expected` vs `22.23 GiB actual` mismatch.

### Non-goals

- Do not implement true tensor/pipeline parallel distributed loading in this
  transport step.  TP/PP rank topology remains a separate track.
- Do not expose generic byte-range reads over the network.
- Do not batch all model tensors at once.  That would reduce request count but
  would create large temporary payload buffers and defeat UMA peak-memory
  goals.

### Stage B1: bounded `read_many` for full/sliced entries

Add an optional source method used only by `execute_weight_plan`:

```text
read_many_cpu(requests: tuple[WeightPlanRemoteReadRequest, ...])
    -> tuple[torch.Tensor, ...]
```

`WeightPlanRemoteReadRequest` is a small executor-local data shape, not a new
model semantic primitive:

- `checkpoint_name`
- optional `source_slices`
- optional `read_segments`
- optional `staging_shape`

B1 can initially support only entries without `read_segments`; this covers
Qwen35B's `124,306` full reads and gives a clean measurement before segment
batching.  If an entry is unsupported, the executor keeps the existing
per-entry fallback.

The executor forms batches from consecutive scheduled prepared entries:

- preserve schedule order;
- stop a batch before required/skipped boundaries that already need special
  handling;
- stop when the sum of expected response payload bytes exceeds a configurable
  limit, for example `VLLM_UMA_ODIRECT_REMOTE_BATCH_BYTES` or loader extra
  config, initially conservative (`64-256 MiB`);
- after receiving tensors, call each weight loader in the original batch
  order and release tensors as the loop advances.

This keeps peak remote staging near:

```text
batch response payload bytes
+ reconstructed tensor bytes for that batch
+ normal model parameter allocations
```

instead of holding the whole model payload at once.

Wire protocol:

- request op: `read_many`
- request payload: JSON only, containing an ordered `items` array of
  tensor names and slice descriptors
- response: one JSON header plus one raw concatenated payload
- response header contains an ordered `tensors` array with dtype, shape, and
  payload sizes/offsets
- remote reconstructs tensors in order and verifies each payload size against
  the local catalog-derived expectation

Owner-side validation:

- authenticate once per request as today;
- reject batches whose item count or total expected payload exceeds the
  configured limit;
- for each item, resolve the tensor through the owner catalog and call the
  same local `read_full_cpu` / `read_slice_cpu` methods used today;
- return tensors in request order.

The owner still serializes source access through `_source_lock`.  B1 is about
reducing request/frame overhead, not introducing concurrent O_DIRECT reads.

Expected impact:

- Qwen35B request count drops from `124,306` to roughly
  `ceil(21.73 GiB / batch_limit)`.  At `128 MiB`, this is about `174`
  requests.
- If the remaining `~115s` after persistent connection is mostly per-request
  JSON/frame/tensor reconstruction overhead, B1 should be a large step toward
  single-node O_DIRECT time plus network transfer and bounded reconstruction
  cost.

### Stage B2: segments and fused-source coalescing

After B1 measurement, extend the same batch request shape to
`read_segments` entries:

- each item carries `read_segments` and `staging_shape`;
- owner validates each item with `validate_weight_plan_read_segments`;
- owner materializes each staging tensor using the existing local segmented
  read path;
- remote receives one tensor per item and executor dispatch stays unchanged.

This is sufficient for TeleChat2/HunYuan/Llama4 segment-family remote smokes,
but it does not yet coalesce multiple segment entries into a single owner-side
staging tensor.  That can remain a later optimization if B2 still leaves a
measurable segment-family gap.

Implementation note: the initial B2 implementation does preserve the local
Stage-A read-window reuse for the common fused-source case.  The owner-side
`read_many` handler groups segmented items by `checkpoint_name` and calls the
existing `read_segment_group_into_cpu` path for each group.  This keeps
HunYuan-style Q/K/V entries that share one fused checkpoint tensor from
devolving into independent owner sweeps.  Response tensors are still returned
in the original request order, so executor dispatch order remains unchanged.

The batch payload cap is configurable through the loader extra config
`remote_batch_payload_mib` (default `128`).  TeleChat2's K/V staging tensors
are about `72 MiB` each, so the B2 real-checkpoint smoke used `256 MiB` to let
K/V pairs that share one `key_value.weight` source tensor enter the same remote
batch while still keeping peak transport staging bounded.

### Stage B3: owner capability handshake

Add a small owner op, for example `source_capability`, returning:

- `chunk_size`
- `window_size`
- `alignment`
- `supports_strided_read`
- optional maximum owner response payload recommendation

`RemoteODirectSafetensorsWeightSource` stores these values as attributes that
`execute_weight_plan` already probes (`_chunk_size`, `_window_size`,
`_alignment`).  This makes remote schedule simulation use the owner's real
O_DIRECT settings and should make Qwen35B expected bytes match owner actual
bytes (`22.23 GiB` with the 128 MiB window) instead of the conservative
`22.86 GiB` computed with local defaults.

The handshake runs eagerly when constructing the remote source.  This avoids a
timing hazard where `schedule_weight_plan_reads` could run before the remote
source has learned the owner source's O_DIRECT settings.  If the owner does
not support the handshake, fail closed; silently falling back to local defaults
would reintroduce hidden expected/actual accounting drift.

### Tests

Unit/fixture tests should cover:

- repeated full reads use one connection and one `read_many` request per
  batch;
- batch response order is preserved even when tensor names are not sorted;
- owner rejects unknown names, malformed slices, oversized item counts, and
  oversized total payload before reading;
- remote rejects payload-size mismatches and malformed tensor descriptors;
- fallback remains correct for unsupported entries;
- owner capability handshake updates remote scheduling parameters;
- a small real O_DIRECT safetensors fixture verifies values, not only shapes.

The first real measurement should repeat the Qwen35B remote smoke with the
same settings as the `143.98s` persistent baseline and record:

- request count / batch count;
- model load time;
- owner direct read stats;
- remote stream bytes;
- local/remote `buff/cache`, swap, memory PSI, and IO PSI.

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
