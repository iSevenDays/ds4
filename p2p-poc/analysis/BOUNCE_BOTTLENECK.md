# BOUNCE bottleneck analysis

Captured during the TP+SSD investigation on 2x RTX 4090 D (48 GB each),
PCIe gen4 x16, no NVLink bridge, ZFS ARC ≈ 27 GB (out of 86 GB model),
single physical NVMe SSD serving ~1.7 GB/s single-thread pread, 5–6 GB/s
with 4–8 parallel preads.

## 1. Hardware / driver facts

```
$ nvidia-smi --query-gpu=name,memory.total --format=csv
NVIDIA GeForce RTX 4090 D, 49140 MiB
NVIDIA GeForce RTX 4090 D, 49140 MiB
```

Existing d4 probe (`ds4_cuda.cu:2280–2360`) prints the peer-access
matrix on startup:

```
ds4: peer access matrix (validated): 0->1 BOUNCE 1->0 BOUNCE
```

i.e. `cudaDeviceCanAccessPeer(0, 1)` and `cudaDeviceCanAccessPeer(1, 0)`
both return 0. NVIDIA blocks P2P on consumer GPUs at the driver level
("GeForce" brand). `cudaDeviceEnablePeerAccess` returns
`cudaErrorPeerAccessUnsupported` if you try to force it.

Every cross-GPU op therefore runs through `g_xdev_bounce[][]` — a
pinned-host bounce buffer (`ds4_cuda.cu:2958–2982`):

```c
// GPU_src -> host (D2H on src stream)
cudaMemcpyAsync(g_xdev_bounce[sd][dd], src->ptr, bytes,
                cudaMemcpyDeviceToHost, s);
cudaEventRecord(e, s);
// host -> GPU_dst (H2D on dst stream, waits on event)
cudaStreamWaitEvent(s2, e, 0);
cudaMemcpyAsync(dst->ptr, g_xdev_bounce[sd][dd], bytes,
                cudaMemcpyHostToDevice, s2);
cudaStreamSynchronize(s2);
```

## 2. Measured BOUNCE bandwidth (this hardware)

Microbenchmark in `benchmarks/bw_test.cu`. Run from a cold start, 100
iterations of a 6 MB copy (typical expert size is 6.75 MB):

| pattern                                  | throughput  | per-copy |
|------------------------------------------|-------------|----------|
| `cudaMemcpyPeer` (sync)                  | **3.16 GB/s** | 1.99 ms |
| `cudaMemcpyPeerAsync` on a single stream | 0.71 GB/s   | 8.85 ms  |
| explicit D2H+H2D via pinned bounce       | 0.39 GB/s   | 16.1 ms  |
| 2 threads x `cudaMemcpyPeer` (sync)      | 0.71 GB/s   | (per batch 53 ms) |
| 4 threads x `cudaMemcpyPeer` (sync)      | 0.71 GB/s   | (per batch 53 ms) |
| 6 threads x `cudaMemcpyPeer` (sync)      | 0.71 GB/s   | (per batch 53 ms) |

Three facts this table establishes:

1. **Sync `cudaMemcpyPeer` is the fastest path** (3.16 GB/s peak).
   Anything async is *slower*, because the CUDA driver on consumer
   boards routes async cross-device copies through a much heavier
   validation path that serialises hard.
2. **Multiple threads do not scale.** 1, 2, 4, 6 threads all top out at
   ~0.71 GB/s aggregate — `cudaMemcpyPeer` takes a driver-global lock.
   Background-thread async write-throughputs cannot exceed ~0.7 GB/s
   total.
3. **BOUNCE ≈ SSD.** Sync BOUNCE peaks at 3 GB/s; the SSD serves at
   1.7 GB/s single-thread, 5–6 GB/s with queue depth 4–8. There is no
   regime where sending an expert GPU0↔GPU1 is *meaningfully* cheaper
   than re-reading it from SSD.

## 3. What this means for an L2 expert cache

The natural design — GPU0's LRU (3244-cap) as L1, GPU1's VRAM as a
larger L2 — loses once you cost the BOUNCE:

```
per L2 hit:   cudaMemcpyPeer L2 -> L1   = 2.1 ms per expert  (6.75 MB / 3.16 GB/s)
per L2 write: cudaMemcpyPeer L1 -> L2   = 2.1 ms per expert  (same)
per SSD miss: pread + cudaMemcpyAsync   = 4.5 ms per expert  (6.75 MB / 1.5 GB/s)
```

So:
- Replacing an SSD miss (4.5 ms) with an L2 hit (2.1 ms) **saves
  2.4 ms per hit**.
- But you pay the L2 write (2.1 ms) — and that write *has to happen
  on the same cudaMemcpyPeer-locked thread pool* that the read does.
- Net per miss that graduates to an L2 hit: 4.5 − 2.1 − 2.1 = **0 ms**.
- Plus the L2 lookup adds a metadata scan + per-slot event sync.

I verified this end-to-end during the investigation. With sync
write-through the cold tok/s dropped from 5.8 (parallel pool only) to
**2.2 tok/s** — and per-layer `load=` jumped from ~5 ms to 60–70 ms,
indicating PCIe / driver-lock contention beyond the simple
back-of-envelope arithmetic.

The async variants are *worse*, not better, because
`cudaMemcpyPeerAsync` on this hardware measures 4–8x slower than sync.

## 4. Profiling breakdown (committed baseline = `85512d5`)

With `DS4_METAL_DECODE_STAGE_PROFILE=1`,
`DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE=1`,
`DS4_CUDA_STREAMING_LOAD_PROFILE=1`:

```
=== per-stage decode averages (over ~57k samples) ===
decode/routed_moe                            avg=2.189 ms / layer
decode/router                               avg=0.299 ms / layer
decode/attn_output                          avg=0.250 ms / layer
decode/q_path                               avg=0.097 ms / layer
... (everything else < 0.07 ms / layer)

=== load-pool summary (every 100 layers, last seen) ===
layers=91500 experts=549000 hit_rate=87.1% ssd_reads=70848
  ssd_gb=467.02 phase1_us/expert=53.1 phase2_us/expert=597.7 workers=4
```

Translated:
- 13% of expert reads miss the primary LRU → SSD read.
- Per-miss phase2 (SSD read) wall time = 598 µs/expert (averaged over
  all experts); per-miss-only time ≈ 4.5 ms.
- Per-token breakdown (40 layers, 6 experts each):
  - SSD miss path: 0.13 × 6 × 40 × 4.5 ms / 4 workers ≈ 35 ms
  - routed_moe (consumer, waits on SSD DMA): 40 × 2.2 ms ≈ 88 ms
  - attention + router + everything else: ≈ 30 ms
  - **Total ≈ 153 ms / token = 6.5 tok/s** (matches measured)

`routed_moe`'s 2.2 ms avg is a symptom, not a cause — when the layer's
SSD load was slow, the consumer's wall time inflates because the GPU is
still draining the DMA on the same device.

## 5. GPU0/GPU1 SM% and memory utilisation (dmon)

```
gpu  sm%  mem%  pwr   fb(MB)   rxpci(MB/s)  txpci(MB/s)
0    28%   9%   89W   47768    1.7–3.2       0.2      # does all SSD reads + TP primary
1     5%   1%   61W   10782    0.02          0.01     # idle except for TP peer attn
```

GPU1's 38 GB free VRAM and <10% SM% are pure idle capacity that this
PoC targets.

## 6. Why P2P flips the verdict

With P2P (`cudaDeviceEnablePeerAccess` succeeds) the same code path
becomes a single direct `cudaMemcpyPeerAsync` that uses the peer's
BAR1 mapping at PCIe-gen4-x16 bandwidth. Typical measured numbers on
cards that do allow P2P (RTX 6000 Ada, A100, H100):

| hardware             | sync peer BW | async peer BW | notes |
|----------------------|--------------|---------------|-------|
| RTX 6000 Ada (pro)   | 24 GB/s      | 26 GB/s       | no NVLink, P2P-over-PCIe |
| A100 (no NVLink)     | 22 GB/s      | 24 GB/s       |       |
| A100 (NVLink)        | 150 GB/s     | 290 GB/s      |       |
| H100 (NVLink)        | 200 GB/s     | 400 GB/s      |       |

At 24 GB/s, a 6.75 MB expert BOUNCE takes **0.28 ms** — comfortably
below the 4.5 ms SSD read. The arithmetic from §3 inverts:

```
per L2 hit (P2P):  0.28 ms
per L2 write (P2P): 0.28 ms
per SSD miss:      4.5 ms
net savings per miss→L2 hit: 4.5 - 0.28 - 0.28 = 3.94 ms
```

With ~31 misses per token, that's **~120 ms / token saved** — which is
the difference between 6.5 tok/s and 30+ tok/s if SSD were the only
thing left.

In practice other bottlenecks (matmul, attention) re-emerge, and the
realistic projection is **~14–18 tok/s** on this model + CPU/SSD
combo. See `EXPECTED_RESULTS.md` for the back-of-envelope.
