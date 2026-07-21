# Expected results with P2P enabled

All numbers projected from the measured baseline (HEAD `85512d5` of
`feat/tp-ssd-experiment`, parallel pread pool + race fix). Back-of-
envelope; real numbers will vary ±20%.

## 1. The arithmetic

Captured constants on the current hardware:

```
N_LAYERS                  = 40
N_EXPERTS_PER_LAYER       = 6
N_TOTAL_EXPERTS           = 5556
PRIMARY_LRU_CAPACITY      = 3244 experts   (GPU0 22 GB free)
L2_CAPACITY               = 4986 experts   (GPU1 38 GB free)
EXPERT_BYTES              = 6.75 MB        (gate + up + down)

# per-miss latencies (no P2P / current):
T_SSD_MISS                = 4.5 ms
T_BOUNCE_READ             = 2.1 ms         (cudaMemcpyPeer sync, 3.16 GB/s)
T_BOUNCE_WRITE            = 2.1 ms

# per-miss latencies (with P2P, RTX 6000 Ada / A100 PCIe class):
T_P2P_READ                = 0.28 ms        (cudaMemcpyPeerAsync, ~24 GB/s)
T_P2P_WRITE               = 0.28 ms

# non-SSD decode cost (constant across configurations):
T_OTHER_PER_TOKEN         = 60 ms          (router + attn + matmul + epilogue)
```

## 2. Per-token budget

### Current (committed baseline, no L2)

```
primary hit rate           = 0.87
misses per token           = (1 - 0.87) * 6 * 40          = 31.2
SSD read time              = 31.2 * T_SSD_MISS / 4 workers = 35.1 ms
routed_moe wait            = 88 ms                         (measured)
T_other                    = 30 ms                         (measured)
TOTAL                      = 153 ms / token  →  6.5 tok/s
```

### With L2 + sync BOUNCE writes (P2P OFF — what we tested and rejected)

```
L2 hit rate (after warmup) = 1.0   (3244 + 4986 > 5556)
SSD read time              = 0     (every miss served from L2)
L2 read time               = 31 * T_BOUNCE_READ            = 65.2 ms
L2 write time (sync)       = 31 * T_BOUNCE_WRITE           = 65.2 ms
T_other                    = 30 ms
TOTAL                      = 160 ms / token  →  6.3 tok/s

# Worse, because BOUNCE writes serialise PCIe with SSD reads.
```

### With L2 + P2P (sync peer, projected, 24 GB/s)

```
L2 hit rate (after warmup) = 1.0
L2 read time               = 31 * 0.28 ms                  = 8.7 ms
L2 write time (sync)       = 31 * 0.28 ms                  = 8.7 ms
T_other                    = 30 ms
TOTAL                      = 47 ms / token   →  21 tok/s
```

### With L2 + P2P + async writer (patch 0004, projected)

```
L2 read time               = 8.7 ms        (still sync — needs the data)
L2 write time              = 0   ms        (hidden behind SSD reads on a bg thread)
T_other                    = 30 ms
TOTAL                      = 39 ms / token   →  26 tok/s
```

### Headroom with NVLink (A100 SXM / H100 SXM, 150 GB/s+)

```
T_P2P_READ  ≈ 0.05 ms; T_P2P_WRITE ≈ 0.05 ms
TOTAL                      = 31 ms           →  32 tok/s
```

## 3. Realistic projections per hardware class

The back-of-envelope above hits hard ceilings that don't show up in
the simple arithmetic — `routed_moe` has its own compute floor, the
SSD itself becomes the bottleneck again at high hit-rate because the
13% of unique-expert cold-loads still need to come from SSD the first
time, etc. Calibrated against the current measurement:

| hardware                              | sync P2P BW | projected warm tok/s |备注 |
|---------------------------------------|-------------|----------------------|---|
| RTX 4090 D (current, P2P disabled)    | n/a         | **6.5**              | measured |
| RTX 4090 D + community P2P unlock     | ~3 GB/s     | 6–7                  | near-zero win, BOUNCE-bound |
| RTX 6000 Ada (pro, no NVLink)         | 24 GB/s     | **14–16**            | patches here pay off cleanly |
| RTX A100 PCIe                         | 22 GB/s     | **14–16**            | similar |
| RTX A100 SXM4 (NVLink)                | 150 GB/s    | **20–24**            | matmul-bound |
| H100 SXM5                             | 200 GB/s    | **22–26**            | matmul-bound |
| H100 SXM5 + faster SSD (7 GB/s)       | 200 GB/s    | **28–32**            | approaches compute floor |

## 4. What "matmul-bound" means here

At ~22+ tok/s the per-token budget is ~45 ms. Of that:
- ~30 ms is `T_other` (attention + router + matmul + epilogue, all
  the small kernels).
- ~8 ms is the L2 read peer-copy (still 31 misses × 0.28 ms).

To go faster you need to attack `T_other`. Concretely:
- `decode/attn_output` is 0.25 ms/layer × 40 = 10 ms. TP attention
  already halves this; further gains need kernel work.
- `decode/router` is 0.30 ms/layer × 40 = 12 ms. Already small.
- `decode/routed_moe` (matmul, cache-hit case) is 0.30 ms/layer × 40 =
  12 ms. Re-enabling `cuda_tp_ep` with sharded half-experts would
  halve this by running the matmul on both GPUs in parallel (the
  Direction B mentioned in the upstream task).

## 5. Sanity check: theoretical minimum

The model's matmul FLOPs per token at decode: roughly 240 expert
matmuls × (2048² × 2) × 2 FLOPs ≈ 4 GFLOP. RTX 4090 has ~80 TFLOPs
of q4/q8 throughput. Theoretical min ≈ 50 µs/token = 20 000 tok/s.
We are nowhere near that — the bottleneck is SSD I/O, and after this
PoC lands, then matmul-TP.

## 6. Risk register

| risk | likelihood | mitigation |
|------|------------|------------|
| Async writer queue grows unbounded if P2P BW < SSD read rate | low (P2P is much faster than SSD even on weak pro cards) | cap queue at 64, drop new writes once full (logged) |
| Per-slot event pool leak | medium | reuse events cyclically; pre-allocate sized to L2 capacity |
| `cudaDeviceEnablePeerAccess` requires BAR1 ≥ slab size | high on default-config pro cards | document the `nvidia-smi --bar1-size=1024` step in `P2P_ENABLEMENT.md` |
| Multi-GPU topology change (new GPU added) at runtime | low | re-probe P2P on every L2 ensure; release old L2 if peer matrix changed |
| Producer evicts a primary slot still being written to L2 | medium | write_through holds the primary slot's `claimed` flag across the peer-copy, primary slot can't be re-claimed until publish |

## 7. What success looks like (acceptance criteria for the PoC)

Once P2P is enabled and all four patches are applied, on this exact
model + CPU/SSD combo:

1. `ds4[load-pool] L2 cache ready on tier 1 capacity=... experts (p2p=1)`
   appears in startup log.
2. After 200 tokens of generation, the periodic summary shows:
   ```
   hit_rate≈87% (unchanged primary rate)
   l2_hits / l2_misses ≈ N (where N grows over time, eventually ≈ ssd_reads)
   ```
3. Warm tok/s (after 500+ tokens of generation) is ≥ 2× the committed
   baseline (≥ 13 tok/s on RTX 6000 Ada class).
4. Cold tok/s (first 100 tokens) is ≥ the committed baseline (no
   regression during cache fill).
5. Correctness: `8+8=16`, counting 1..100, longer essay generation.

If criteria 1–5 hold, the L2 cache is shippable.
