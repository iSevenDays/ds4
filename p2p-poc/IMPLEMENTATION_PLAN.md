# Implementation plan

Four patches, in order. Each builds on the previous. All four apply
cleanly on `85512d5` (the current HEAD of `feat/tp-ssd-experiment`).

## Patch 0001 — p2p detection (`patches/0001-p2p-detection.patch`)

**Goal:** capture a one-bit "is P2P usable between tier A and tier B"
flag at startup, so downstream code can gate on it.

**Why:** the existing `g_gpu_peer_ok[][]` matrix already exists, but
it's set to `0` for BOUNCE even on hardware where P2P could work but
hasn't been explicitly enabled via `cudaDeviceEnablePeerAccess`. This
patch calls `cudaDeviceEnablePeerAccess` (which is the actual enabling
call, not just the probe) for every pair, and updates the matrix based
on the return value.

**File:** `ds4_cuda.cu:2280–2360` (the probe block).

**Diff size:** ~30 lines. No behaviour change for 4090 D (the call
returns `cudaErrorPeerAccessUnsupported`, we keep the matrix at 0,
everything else is unchanged).

## Patch 0002 — l2 cache core (`patches/0002-l2-cache-core.patch`)

**Goal:** add the L2 cache data structures and primitive helpers, all
gated on `g_gpu_peer_ok[primary][l2] && g_gpu_peer_ok[l2][primary]`.

**What it adds:**
- `cuda_stream_l2_tier_for(primary_tier)` — picks the L2 tier (lowest
  tier ≠ primary). Returns `-1` if P2P is unavailable or env
  override is set.
- `cuda_stream_l2_ensure(primary_tier, gate_bytes, down_bytes)` —
  lazily allocates the L2 cache slab on the L2 tier via the existing
  `cuda_stream_expert_cache_prepare`. Caches the pointer in
  `g_load_pool.l2_cache`.
- `cuda_stream_l2_fetch_into_primary(...)` — lookup-in-L2 +
  peer-copy L2 → primary slot. Uses `cuda_xdev_copy_raw` which
  dispatches to `cudaMemcpyPeer` (P2P path on capable hardware,
  transparent BOUNCE fallback otherwise).
- `cuda_stream_l2_write_through(...)` — peer-copy primary → L2 slot
  after a fresh SSD read, with claim-flag serialisation so concurrent
  write_throughs don't pick the same L2 slot.
- New observability counters on `g_load_pool`: `l2_hits`, `l2_misses`,
  `l2_writes` — printed in the periodic summary.

**Why inert without P2P:** every entry point checks
`g_gpu_peer_ok[primary][l2]`. On 4090 D that bit is `0`, so the L2
helpers return immediately without doing work.

**File:** `ds4_cuda.cu`, inserted between the parallel pool code
(ends ≈ line 24341 in committed state) and the producer function
`cuda_stream_selected_cache_begin_load`.

**Diff size:** ~250 lines.

## Patch 0003 — l2 integrate into workers (`patches/0003-l2-integrate.patch`)

**Goal:** wire the L2 into the parallel pool's per-expert loop.

**Where:** in `cuda_stream_load_worker_run`, between Phase 1 (primary
LRU find/claim) and Phase 2 (SSD read). New phase:

```
Phase 2a (new): if cache_slot < 0 AND l2 enabled AND l2 has expert:
                peer-copy L2 → claimed primary slot, mark primary
                slot valid + claimed=0, set cache_slot = load_slot.
                Increment l2_hits.
Phase 2b (existing SSD read, now also does Phase 2c when L2 enabled):
Phase 2c (new): peer-copy primary → L2 slot (best-effort, async on
                P2P-capable hardware), increment l2_writes.
```

Phase 2a's lookup happens under `g_load_pool.l2_mutex`. The peer-copy
itself happens *outside* the mutex so multiple workers can pull from
L2 in parallel.

**File:** `ds4_cuda.cu`, inside the worker function body (around line
24200 in the committed state).

**Diff size:** ~80 lines.

## Patch 0004 — l2 async writer (`patches/0004-l2-async-writer.patch`)

**Goal:** on P2P-capable hardware, push L2 writes to a dedicated
background thread so they never block the producer's workers.

**Why:** even at P2P bandwidth (24 GB/s), each write is 0.28 ms.
Across ~31 misses/token that's ~9 ms/token of L2-write wall time. If
the writes run synchronously in worker threads, they contend with
worker's SSD reads (which need the same PCIe root complex). A
dedicated writer thread decouples the two.

**What it adds:**
- A persistent `std::thread g_l2_writer` with a
  `std::queue<l2_write_job>` and a cond-var.
- `cuda_stream_l2_write_through_async(slot, expert, table, primary_cache,
    primary_slot)` — pushes a job, signals the cond-var, returns
    immediately.
- The writer pops jobs one at a time, does
  `cudaMemcpyPeerAsync(primary → l2, g_l2_writer_stream)`, records
  a per-slot event, marks the slot valid.
- Phase 2c in the worker now calls `_async` instead of the sync
  version (gated on `g_gpu_peer_ok[primary][l2]`).

**Correctness:** Phase 2a (L2 lookup) must wait for any pending write
to that slot before reading. Tracked via per-L2-slot `cudaEvent_t`
stored alongside the slot metadata. Phase 2a calls
`cudaEventSynchronize(slot.pending_write_event)` before the
peer-copy-out.

**Fallback:** when P2P is unavailable (4090 D today), the async writer
is never started and Phase 2c remains a no-op — the producer runs
exactly as it does at HEAD `85512d5`.

**File:** `ds4_cuda.cu`, additive on top of patch 0003.

**Diff size:** ~150 lines.

## Total impact

| file           | +/- lines | net-change to baseline commit |
|----------------|-----------|-------------------------------|
| `ds4_cuda.cu`  | +520 / -8 | +512                          |

All additive. No existing behaviour changes when P2P is disabled.

## Order of operations to deploy

1. Apply patch 0001, rebuild, run server, confirm `peer access matrix`
   log line. On 4090 D: `0->1 BOUNCE 1->0 BOUNCE` (unchanged). On
   P2P-capable hw: `0->1 P2P 1->0 P2P`.
2. Apply patch 0002, rebuild, run server. No behaviour change yet
   (L2 helpers exist but aren't called).
3. Apply patch 0003, rebuild, run server.
   - On 4090 D: identical tok/s to baseline (Phase 2a/2c are guarded
     and skipped).
   - On P2P hw: tok/s should climb; check the L2-ready banner and the
     `summary` line for hit-rate.
4. Apply patch 0004, rebuild, run server. Should be the same or better
   than 0003 (async writes should reduce worker contention on P2P hw).

## Rollback

Each patch is independent in the sense that you can `git revert` them
in reverse order. Patch 0004 first, then 0003, then 0002, then 0001.

## What I am *not* proposing in this PoC

- **Direct NVLink plumbing.** If you have NVLink, the right answer is
  `cudaMallocAsync` with `cudaMemAdvise`-managed buffers and letting
  the driver migrate. That's a separate, larger piece of work.
- **Moving the producer itself to GPU1.** Producer runs on the same
  tier as the consumer; splitting experts half-and-half across tiers
  is the existing `cuda_tp_ep` path, which is currently force-disabled
  under SSD streaming (ds4.c:54377 `engine_cuda_tp_ep_requested`). That
  is Direction B in the upstream task and is significantly more
  invasive than this PoC.
- **`cudaMallocManaged`.** Unified memory has high overhead on
  consumer GPUs (slow page-fault handler) and is uniformly slower than
  explicit BOUNCE for these access patterns. Not recommended.
