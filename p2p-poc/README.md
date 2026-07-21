# P2P PoC: GPU1 VRAM as L2 expert cache for TP+SSD streaming

## Status: DESIGN + READY-TO-APPLY PATCHES

This folder contains the design, patched code, and validation harness for
turning GPU1's idle VRAM (~38 GB free during TP+SSD decode) into an L2
expert cache, gated on **P2P peer access being enabled between GPU0 and
GPU1**.

On the current hardware (2x RTX 4090 D) P2P is blocked by NVIDIA at the
driver level (consumer GPUs). All cross-GPU ops go through a host
bounce-buffer path that the d4_cuda BENCHMARKS show runs at 1–3 GB/s
sync, 0.4–0.7 GB/s async — comparable to or slower than the SSD itself
(1.7–3 GB/s). That is why a naive L2 implementation made tok/s *worse*
(5.8 → 2.2 cold) in the experiment logged in
`analysis/BOUNCE_BOTTLENECK.md`.

The patches here are inert (compile in, hidden behind `g_gpu_peer_ok[]`
and runtime detection) until P2P is enabled at the driver / hardware
level. Once P2P is on, expected bandwidth jumps to 20–30 GB/s (PCIe
gen4 x16 peer-direct) and the L2 cache becomes a clear net win — see
`EXPECTED_RESULTS.md` for the projection (~10–12 tok/s warm vs 6.6 tok/s
today on this hardware, ~14+ tok/s on RTX 6000 Ada / H100).

## Layout

```
p2p-poc/
├── README.md                      this file
├── analysis/
│   ├── BOUNCE_BOTTLENECK.md       measured numbers + dmon + profile breakdown
│   ├── P2P_ENABLEMENT.md          how to turn P2P on (HW/SW/driver)
│   └── PROFILING_BASELINE.md      the captured baseline this PoC starts from
├── IMPLEMENTATION_PLAN.md         step-by-step plan with file:line targets
├── EXPECTED_RESULTS.md            projected tok/s with P2P at various BW
├── patches/
│   ├── 0001-p2p-detection.patch   NOTE: already implemented in tree at
│   │                               ds4_cuda.cu:2326-2400 (with bonus
│   │                               corruption-validation for RTX 6000 Ada).
│   │                               This patch is kept only as a reference
│   │                               for what the existing code already does.
│   ├── 0002-l2-cache-core.patch   L2 cache struct + helpers (inert if no P2P)
│   ├── 0003-l2-integrate.patch    hook L2 lookup + write-through into workers
│   └── 0004-l2-async-writer.patch background thread for non-blocking writes
├── benchmarks/
│   ├── bw_test.cu                 current BOUNCE bandwidth microbench
│   ├── bw_test_p2p.cu             P2P bandwidth microbench (run after enabling)
│   └── run-shader-trace.sh        nsys/ncu recipe for cross-checking
└── ROLL_OUT_CHECKLIST.md          go/no-go gates for taking this to production
```

## Important: P2P detection is already in the tree

The existing code at `ds4_cuda.cu:2326-2400` already probes
`cudaDeviceCanAccessPeer`, calls `cudaDeviceEnablePeerAccess`, and runs
a multi-size / multi-iteration validation pass before setting
`g_gpu_peer_ok[i][j] = 1`. The validation is required because **RTX
6000 Ada under recent drivers silently corrupts `cudaMemcpyPeer` even
when both API calls report success** — see the comment block at
`ds4_cuda.cu:2311-2325`. This means even on P2P-capable pro cards,
whether `g_gpu_peer_ok` ends up 1 depends on the driver version
passing the validation. On RTX 4090 D the probe returns 0 in the
first place, so the matrix is all-zero and the L2 stays inert.

## One-paragraph summary

`ds4_cuda.cu`'s SSD streaming producer loads routed-expert weights from
the SSD into a per-tier LRU cache on GPU0 (`g_ssd[0].expert_cache`,
capacity 3244 experts ≈ 22 GB). The model has 5556 routed experts total,
so the LRU can only hold ~58% of them — the producer's observed
hit-rate is 87% (the working set is concentrated) and the 13% misses
cost ~4.5 ms each (single-thread pread at 1.7 GB/s per expert of
6.75 MB). GPU1 sits idle with 38 GB VRAM free; putting the 13% miss
working set on GPU1 would push the effective hit-rate to ~100%.
**The only thing stopping that is the BOUNCE copy overhead between the
two GPUs. P2P peer access removes that overhead.**

## How to use this PoC

```bash
# 1. Apply all four patches on top of feat/tp-ssd-experiment.
cd /root/antirez/ds4
for p in p2p-poc/patches/0*.patch; do git apply --check $p && git apply $p; done

# 2. Build.
PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
  make -j$(nproc) ds4-server CUDA_ARCH=sm_89

# 3. Enable P2P (see analysis/P2P_ENABLEMENT.md).
sudo nvidia-smi --conf-compute=1   # or whatever your enabling path is
modprobe nvidia NVreg_OpenRmEnableUnsupportedGpus=1   # if needed

# 4. Verify P2P bandwidth.
nvcc -arch=sm_89 -o /tmp/bw_p2p p2p-poc/benchmarks/bw_test_p2p.cu
/tmp/bw_p2p         # expect >20 GB/s if P2P is truly enabled

# 5. Run ds4-server with default settings (L2 auto-enables when P2P is detected).
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=4 ./ds4-server \
  --cuda --cuda-tensor-parallel --ssd-streaming \
  --ssd-streaming-cache-experts 40GB --gpu-vram 45,45 --ctx 262144 \
  -m .../DeepSeek-V4-Flash-*.gguf --mtp .../DSpark-support.gguf --dspark \
  --host 0.0.0.0 --port 8018

# 6. Confirm L2 is active (look for these lines in stderr):
#    ds4[load-pool] L2 cache ready on tier 1 capacity=XXXX experts (p2p=1)
#    ds4[load-pool] summary ... l2_hits=... l2_misses=... l2_writes=...
```

## Pointers into the codebase (all in `/root/antirez/ds4`)

| Component | File | Key lines |
|---|---|---|
| P2P peer-access probe (existing) | `ds4_cuda.cu` | 2280–2360 |
| BOUNCE fallback impl (existing) | `ds4_cuda.cu` | 2935–2982 |
| Per-tier SSD context | `ds4_cuda.cu` | 221–234 (`struct ds4_ssd_ctx`, `g_ssd[DS4_MAX_GPUS]`) |
| SSD producer (parallel pool) | `ds4_cuda.cu` | 23825–24341 (committed in 905ac92) |
| LRU slot + lru_slot logic | `ds4_cuda.cu` | 188–207, 23694–23710 |
| Selected-cache consumer | `ds4_cuda.cu` | routed_moe_launch ≈ 21040+ |
| `g_gpu_peer_ok[][]` matrix | `ds4_cuda.cu` | 332 area, set at probe time |
| Layer-phase dispatch (ds4.c) | `ds4.c` | 22905–22920 (cuda_tp_moe / cuda_tp_ep gating) |

## Commits this PoC builds on

```
85512d5 feat(ssd): per-expert race fix + load-pool observability
905ac92 feat(ssd): parallel pread worker pool for SSD streaming expert loads
dcddd6b fix(ssd/tp): apply support-map offset bias in q8 f16/f32 strict lookup   (starting HEAD)
```

The patches in `patches/` are designed to apply cleanly on `85512d5`.
