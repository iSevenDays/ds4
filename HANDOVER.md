# DS4 DeepSeek-V4-Flash Production Handover

**Last updated:** 2026-07-21  
**Hardware:** 2× NVIDIA RTX 4090 D 48 GB (Ada Lovelace, sm_89, **no NVLink/P2P**)  
**Model:** DeepSeek-V4-Flash IQ2XXS (2-bit, 81 GB GGUF)  
**Repo:** `/root/antirez/ds4` → `git@github.com:iSevenDays/ds4.git`

---

## 1. Motivation

Serve the 81 GB DeepSeek-V4-Flash 2-bit model at **262k context** across **two 48 GB GPUs** that can't hold it resident (81 GB > 48 GB/card). The solution: **SSD weight streaming** (experts paged from disk into a VRAM LRU cache on demand) + **multi-GPU pipeline parallelism** (layer-split) + **DSpark speculative decoding** (drafter proposes tokens, main model verifies).

---

## 2. What Was Built (the Journey)

### Phase 1: Multi-GPU SSD Streaming (cchuter fork → antirez main)
1. Discovered multi-GPU support lives only in `cchuter/ds4` fork (`mgpu-v0.1.0` tag), not antirez upstream.
2. Built a Docker image (cchuter mgpu-v0.1.0, sm_89) → 2-GPU resident at 16k (~46 tok/s gen).
3. Ported antirez's CUDA SSD streaming into the mgpu build (per-tier `g_ssd[DS4_MAX_GPUS]`, `ssd_current()` via `cudaGetDevice`, embed-device fix, packer/balance).
4. Achieved 2-GPU SSD at 131k (~11 tok/s warm, ~44 GiB VRAM/card).

### Phase 2: Merge antirez main + Bug Fixes
5. Merged antirez `origin/main` (169 commits, including **native CUDA tensor parallelism** + **DSpark speculative decoding** + **session batching**).
6. Fixed merge bugs:
   - `sort_expert_count` reading a zero-init shim global instead of `ssd_current()->selected_cache` (`3c1f869`).
   - `metal_graph_encode_layer_batch` calling `encode_layer_ffn_batch()` **twice** (double-applied FFN — merge artifact).
   - `cuda_ok()` was passive (logged + returned 0) — sticky CUDA errors bricked the server silently. Fixed: fail-fast `_exit(1)` on sticky errors → systemd auto-restart (`2c5c235`).
   - `active_tier=-1` (init sentinel) read out-of-bounds in `metal_graph_eval_token_raw_swa_streaming` → silent `cuda prefill failed`. Fixed: repoint `active_tier = emb_tier` at entry (`9522bac`).
   - `g_current_logical_tier` was process-global but `cudaSetDevice` is per-thread → async staging worker on wrong device. Fixed: `thread_local`.

### Phase 3: Expert LRU Cache Port (the efficiency piece)
7. antirez `origin/main` SSD has **no expert LRU cache** (0 `cuda_stream_expert_cache` refs) — every token re-reads experts from SSD → 0.45 tok/s. Ported the expert LRU from `feat/mgpu-ssd` (79 refs) → per-tier in `g_ssd[t].expert_cache` (`f836788`).
8. Result: VRAM fills to ~44 GiB/card, gen rises to ~10 tok/s warm.

### Phase 4: DSpark + 262k + Full VRAM
9. Removed the `--ssd-streaming` + `--mtp` incompatibility gate (antirez refused them together).
10. Fixed DSpark scratch tensors hardcoded to tier 0 but drafter runs on tier 1 → cross-device illegal memory access. Fixed: allocate on `exec_tier` (`e26e2f0`).
11. Tuned VRAM: `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=2` → 46.5 GiB/card (~96%).
12. Result: **262k context + DSpark + SSD + ~22 tok/s warm + 46.5 GiB/card VRAM**.

### Phase 5: TP+SSD Experiment (exploratory)
13. Made `--cuda-tensor-parallel` coexist with `--ssd-streaming` (`feat/tp-ssd-experiment`).
14. Result: both GPUs active simultaneously but **slower** (6.6 tok/s vs 22 tok/s) — expert streaming overhead under TP negates the attention parallelism.
15. Added **parallel pread worker pool** (4 workers vs 1 → +23% cold, +14% warm) + **race fix** (two workers claiming same LRU slot → silent corruption).
16. Produced **P2P L2 cache PoC** (patches + docs) for P2P-capable hardware (RTX Pro/Tesla/A100) where the verdict flips at ~15 GB/s peer bandwidth.

---

## 3. Current Production State

### Service
```
Name:     ds4-2gpu-ssd.service (systemd, enabled, auto-restart)
Port:     8002 (OpenAI-compatible API)
Binary:   /root/antirez/ds4/ds4-server
Branch:   feat/mtp-native @ e26e2f0
```

### Command (from systemd unit)
```bash
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=2 \
/root/antirez/ds4/ds4-server \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --gpu-vram 45,45 --ctx 262144 \
  -m /root/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --mtp /root/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf \
  --dspark \
  --host 0.0.0.0 --port 8002
```

### Key Environment Variables
| Variable | Value | Purpose |
|---|---|---|
| `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB` | `2` | Leave 2 GiB/card headroom (lower = more expert cache = faster, but tighter) |
| `LD_LIBRARY_PATH` | `/usr/local/cuda/lib64` | CUDA runtime libs |

### Management Commands
```bash
systemctl status ds4-2gpu-ssd          # health
journalctl -u ds4-2gpu-ssd -f          # live logs
tail -f /var/log/ds4-2gpu-ssd.log      # raw log (gen tps, DSpark, cache)
curl http://localhost:8002/v1/models   # endpoint check
nvidia-smi                             # GPU state
```

---

## 4. Actual Numbers (Measured)

### Throughput
| Metric | Value | Notes |
|---|---|---|
| **Warm gen (with DSpark)** | **~22 tok/s** | Steady-state after cache warms (~6 requests) |
| Warm gen (without DSpark) | ~10 tok/s | DSpark roughly doubles throughput |
| Cold gen (first request) | ~0.45 tok/s | Expert cache empty — every token reads from SSD |
| Prefill (2-GPU) | ~15 tok/s | Layer-parallel pipeline |
| DSpark speedup | ~2× | Speculative decoding on predictable continuations |

### VRAM Utilization
| Component | Per Card | Notes |
|---|---|---|
| Resident weights (dense) | ~5 GiB | Attention, embedding, shared expert |
| Graph scratch | ~6 GiB | Per-tier, pre-subtracted from budget |
| KV cache (262k) | ~6 GiB | MLA-compressed |
| **Expert LRU cache** | **~30 GiB** | ~4000 experts/card, 87% hit rate |
| Reserve (free) | ~2 GiB | For KV growth + q8 cache + margin |
| **Total used** | **~46.5 GiB (96%)** | |

### SSD / Cache
| Metric | Value |
|---|---|
| Expert LRU hit rate | 87% |
| SSD read speed (single-thread) | 1.7 GB/s |
| SSD read speed (4-worker parallel) | 5.6 GB/s |
| Expert miss cost | ~4.5 ms/expert |
| LRU cap (reserve=2) | ~4000 experts/card (5556 total in model) |

### P2P / Cross-GPU (RTX 4090 D)
| Path | Speed |
|---|---|
| BOUNCE sync (`cudaMemcpyPeer`) | 3.16 GB/s |
| BOUNCE async | 0.71 GB/s |
| Explicit D2H+H2D | 0.39 GB/s |
| **Conclusion** | BOUNCE ≈ SSD speed → GPU1-as-L2-cache is **net negative** on this hardware |

---

## 5. Verifications Performed

| Check | Result |
|---|---|
| `/v1/models` context_length | 262144 ✅ |
| Correctness (8+8=16, 9+4=13, capital=Paris) | ✅ |
| Both GPUs active during gen | ✅ (pipeline: GPU0+GPU1 sequential) |
| VRAM ~46 GiB/card | ✅ |
| DSpark engaged (target-hidden capture layers 40-42) | ✅ |
| Zero CUDA errors (post-fix) | ✅ |
| NRestarts=0 (stable, no crash loop) | ✅ |
| Fail-fast on sticky CUDA errors | ✅ (`cuda_ok` _exit(1) → systemd restart) |
| Service survives reboot | ✅ (enabled, WantedBy=multi-user.target) |

---

## 6. Branches on `iSevenDays/ds4`

| Branch | Commit | Description |
|---|---|---|
| `main` | `e26e2f0` | **Production**: antirez main + multi-tier SSD + expert LRU + DSpark + all fixes |
| `feat/mtp-native` | `e26e2f0` | Same as main (the development branch) |
| `feat/mgpu-ssd` | `e26e2f0` | Same (force-pushed to unify) |
| `mgpu-ssd-merge-main` | `e26e2f0` | Same |
| `feat/tp-ssd-experiment` | `eb7d594` | **Experimental**: TP+SSD coexistence + parallel pread pool + P2P PoC |

---

## 7. Common Pitfalls

### Operational
1. **Background process teardown**: processes started with `nohup &` inside a Bash tool call get reaped when the shell exits. Use systemd or `run_in_background` for persistent servers.
2. **Port conflicts**: always check `ss -tlnp | grep <port>` before launching. The single-instance lock means a stale process blocks startup with "Address already in use."
3. **Silent graph build**: startup is ~4-5 min of SILENCE (multi-GPU graph compilation) before "listening" appears. Don't assume it's hung.
4. **Cold vs warm**: the first few requests are very slow (~0.45 tok/s) because the expert LRU is empty. After ~6 varied requests it warms to ~22 tok/s.

### Configuration
5. **The 16 GiB reserve** (`DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB`, default 16): silently caps the expert LRU to ~20 GiB/card. Set to 2-4 for full VRAM utilization. The cache only fills ON-DEMAND (warms with requests).
6. **Debug probes** (`DS4_SSD_DEBUG=1`): adds `cudaDeviceSynchronize` per probe call — massive overhead. Keep OFF in production (default).
7. **`--cuda-tensor-parallel` + `--ssd-streaming`**: technically coexists on the experimental branch but is **slower** than pipeline mode on 4090D (BOUNCE ≈ SSD). Use pipeline mode (default, no `--cuda-tensor-parallel`).

### Code
8. **antirez main SSD lacks the expert LRU** (0 `cuda_stream_expert_cache` refs). If you rebuild from a fresh antirez main, you MUST port the LRU from `feat/mgpu-ssd` or gen will be 0.45 tok/s.
9. **Device-mismatch bugs** are the #1 class of issue. Every time a tensor allocated on device A is accessed by a kernel running on device B → `cudaErrorIllegalAddress` (sticky, bricks the context). The fixes: `active_tier = emb_tier` before embed; `WITH_DEVICE(target_dev)` on kernels; `thread_local` for per-thread device tracking.
10. **DSpark scratch tensors**: must be allocated on `exec_tier` (the tier running the drafter), not hardcoded to tier 0.

---

## 8. Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    ds4-server (port 8002)                         │
├──────────────────────┬──────────────────────────────────────────┤
│      GPU0 (tier 0)   │            GPU1 (tier 1)                  │
│  Layers 0-20 + emb   │       Layers 21-42 + output head          │
│                      │       DSpark support model                 │
│  Resident: ~5 GiB    │       Resident: ~5 GiB                    │
│  KV cache: ~3 GiB    │       KV cache: ~3 GiB                    │
│  Scratch:  ~6 GiB    │       Scratch:  ~6 GiB                    │
│  Expert LRU: ~30 GiB │       Expert LRU: ~30 GiB                 │
│  (4000 experts)      │       (4000 experts)                      │
│  Free: ~2 GiB        │       Free: ~2 GiB                        │
├──────────────────────┴──────────────────────────────────────────┤
│  Pipeline: token flows GPU0 → GPU1 (boundary copy via BOUNCE)    │
│  SSD: 81 GB model on ZFS pool, experts stream on cache miss      │
│  DSpark: support model on tier 1 proposes 5 tokens, verified     │
│  Context: 262144 tokens (262k)                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Decode flow per token:**
1. Embed on GPU0 (emb_tier).
2. Layers 0-20 on GPU0 (pipeline stage 1).
3. Hidden state copied to GPU1 (BOUNCE, ~2 ms).
4. Layers 21-42 on GPU1 (pipeline stage 2 + DSpark capture at 40-42).
5. Output head on GPU1.
6. DSpark proposes up to 5 draft tokens (on tier 1).
7. Main model verifies (accepts prefix).

**Expert access (SSD streaming):**
1. Matmul needs 6 experts for this token's layer.
2. Check per-tier LRU (`ssd_current()->expert_cache`).
3. **Hit (87%)**: copy from LRU slab to per-request `selected_cache` (D2D, fast).
4. **Miss (13%)**: stream from SSD (1.7 GB/s) into LRU slot → then to `selected_cache`.

---

## 9. Possible Next Steps

### Short-term (low effort, high value)
1. **Backport parallel pread pool** (`feat/tp-ssd-experiment` commits `905ac92` + `85512d5`) to `feat/mtp-native` → +14-23% throughput (22 → ~25-27 tok/s). The pool parallelizes SSD reads across 4 workers (1.7 → 5.6 GB/s aggregate).
2. **Switch working tree to `feat/mtp-native`** for production stability (currently on `feat/tp-ssd-experiment`).
3. **Add `--dspark-timing`** to log DSpark acceptance rate → confirm DSpark is helping on your workloads.

### Medium-term (moderate effort)
4. **Direction B: sharded half-experts under TP** — each GPU streams its HALF of each expert (half the data → 2× effective SSD bandwidth) + both compute simultaneously. The profiling shows `routed_moe` (2.19 ms/layer, SSD-bound) is the bottleneck. This would attack it directly. Requires extending `cuda_stream_selected_cache` for sharded slabs.
5. **Tune DSpark confidence** (`--dspark-confidence`): default 0.9. Lower = more aggressive drafting (more tokens proposed but lower acceptance). Benchmark on your workloads.

### Long-term (significant effort, hardware-dependent)
6. **P2P L2 expert cache** (apply the PoC patches from `feat/tp-ssd-experiment/p2p-poc/`): on **P2P-capable hardware** (RTX Pro 6000, A100, H100 — anything with NVLink or PCIe P2P), GPU1's VRAM becomes an L2 cache for GPU0's expert misses. The BOUNCE bottleneck (~3 GB/s) disappears (P2P: ~20-50 GB/s). Expected: ~2× tok/s. The PoC includes patches 0002-0004 + benchmarks + go/no-go checklist.
7. **Upgrade hardware**: 2× RTX Pro 6000 96 GB (Blackwell, sm_120) would hold the model nearly resident (81 GB < 96 GB/card) → skip SSD streaming entirely → ~40+ tok/s.

---

## 10. Key Commits (Chronological)

| Commit | Branch | Description |
|---|---|---|
| `b6aa80d` | feat/mgpu-ssd | Port CUDA SSD streaming into mgpu build (single-tier) |
| `779c51b` | feat/mgpu-ssd | Fix SSD streaming stubs + port expert staging |
| `23245bb` | feat/mgpu-ssd | Per-tier SSD weight-streaming across multiple GPUs |
| `5dce0ac` | feat/mgpu-ssd | Merge antirez/main (169 commits) into feat/mgpu-ssd |
| `3c1f869` | feat/mgpu-ssd | Fix sort_expert_count shim bug (merge artifact) |
| `9522bac` | feat/mgpu-ssd | Fix multi-GPU prefill illegal-memory-access (embed device) |
| `2c5c235` | feat/mgpu-ssd | Fail-fast on sticky CUDA errors → systemd auto-restart |
| `0f8fa24` | feat/mgpu-ssd | Add deployment artifacts (Dockerfile, entrypoint, systemd) |
| `da49460` | feat/tp-ssd-experiment | Allow --cuda-tensor-parallel to coexist with --ssd-streaming |
| `f908bfc` | feat/tp-ssd-experiment | Multi-tier init for TP+SSD |
| `dcddd6b` | feat/tp-ssd-experiment | Fix cache-miss offset bias in q8 lookup |
| `905ac92` | feat/tp-ssd-experiment | Parallel pread worker pool (+23% cold) |
| `85512d5` | feat/tp-ssd-experiment | Race fix + load-pool observability |
| `66df47a` | feat/mtp-native | Port multi-tier SSD onto antirez main base |
| `d2479ca` | feat/mtp-native | Fix prefill device-mismatch (active_tier=-1) |
| `f836788` | feat/mtp-native | Port expert LRU cache (0.45 → ~10 tok/s) |
| `e26e2f0` | feat/mtp-native | Fix DSpark scratch device-mismatch + remove SSD+MTP gate |
| `eb7d594` | feat/tp-ssd-experiment | P2P L2 cache PoC package |

---

## 11. P2P Enablement (Attempt on RTX 4090 D)

**Status:** RTX 4090 D is a GeForce card — NVIDIA **blocks P2P access** on GeForce by default. The BOUNCE path (host-memory bounce buffer) gives ~3 GB/s, comparable to SSD. On P2P-capable hardware (RTX Pro, Tesla, A100), direct peer-to-peer gives ~20-50 GB/s, which would make GPU1 an effective L2 expert cache (~2× tok/s). The user plans to attempt P2P enablement via BIOS/driver tweaks.

### How to check if P2P is working
```bash
# 1. Topology matrix — look for "PIX" (peer) or "SYS" (cross-socket).
#    "PHB" or "NODE" = BOUNCE (no P2P).
nvidia-smi topo -m

# 2. Direct bandwidth test (run the PoC benchmark)
cd /root/antirez/ds4 && PATH=/usr/local/cuda/bin:$PATH \
  nvcc -run p2p-poc/benchmarks/bw_test_p2p.cu -o /tmp/bw_p2p -lcudart
# Expected: ≥15 GB/s for P2P; ~3 GB/s for BOUNCE

# 3. ds4's internal P2P detection (runs at startup):
#    Look for "peer access matrix" in the log:
grep "peer access" /var/log/ds4-2gpu-ssd.log
#    "0->1 BOUNCE" = no P2P; "0->1 PIX/SYS" = P2P active
```

### Steps to attempt P2P enablement (reboot required)
1. **BIOS settings** (may require multiple reboots):
   - Enable **Above 4G Decoding** (Required — allows 64-bit BAR).
   - Enable **Resizable BAR** (ReBAR / Smart Access Memory) — critical for P2P.
   - Increase **BAR1 size** to 1024 MB if configurable.
   - Disable **IOMMU** (if enabled — can block peer access on some platforms).

2. **Driver settings** (after reboot):
   ```bash
   sudo nvidia-smi -pm 1                    # persistence mode
   sudo nvidia-smi --query-gpu=pci.domain --format=csv,noheader  # note domains
   # Check topology again:
   nvidia-smi topo -m
   ```

3. **If P2P appears** ("PIX" or "SYS" in topo, ≥15 GB/s in bandwidth test):
   - Apply the L2 cache patches from `p2p-poc/patches/0002-l2-cache-core.patch` + `0003-l2-integrate.patch` + `0004-l2-async-writer.patch` onto `feat/mtp-native`.
   - Rebuild + restart. Expected: ~2× tok/s (GPU1 VRAM becomes L2 cache for GPU0's expert misses).
   - The PoC includes `p2p-poc/ROLL_OUT_CHECKLIST.md` with G0–G7 go/no-go gates.

4. **If P2P still doesn't work** (GeForce driver block):
   - This is expected — NVIDIA enforces P2P as a Quadro/Tesla/Pro feature.
   - No workaround on stock GeForce drivers.
   - The pipeline mode (current production) remains the best option.
   - Consider upgrading to RTX Pro 6000 96 GB (P2P + larger VRAM → near-resident).

### P2P PoC Reference (in-repo)
All PoC materials are at `/root/antirez/ds4/p2p-poc/` (commit `eb7d594` on `feat/tp-ssd-experiment`):
- `README.md` — 6-step usage recipe.
- `IMPLEMENTATION_PLAN.md` — 4-patch plan with file:line targets.
- `EXPECTED_RESULTS.md` — per-hardware tok/s projections (L2 verdict flips at ~15 GB/s).
- `analysis/BOUNCE_BOTTLENECK.md` — the 3.16 GB/s BOUNCE numbers + profiling.
- `analysis/P2P_ENABLEMENT.md` — driver/BIOS/HW matrix.
- `benchmarks/bw_test_p2p.cu` — P2P bandwidth test.
- `patches/0002-l2-cache-core.patch` through `0004-l2-async-writer.patch` — the actual patches.

---

## 12. Quick-Start (New Session)

```bash
# Verify production
systemctl status ds4-2gpu-ssd
curl http://localhost:8002/v1/models | python3 -m json.tool

# Check performance
tail -5 /var/log/ds4-2gpu-ssd.log   # gen tps
nvidia-smi                           # VRAM util

# Send a test request
curl http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# Branch: feat/mtp-native @ e26e2f0 (production)
# Experimental: feat/tp-ssd-experiment @ eb7d594 (TP+SSD + pread pool + P2P PoC)
```
