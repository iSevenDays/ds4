# Roll-out checklist

Use this when promoting the L2 cache from PoC to a real merge.

## Go / no-go gates

Each must pass before moving to the next.

### G0 — prerequisite (hardware / driver)
- [ ] GPUs are a P2P-capable class (pro cards: RTX 6000 Ada / A100 / H100 / Tesla V100).
- [ ] `nvidia-smi --query-gpu=persistence_mode` returns `Enabled` on all tiers.
- [ ] `nvidia-smi -i 0,1 --bar1-size=1024` (or larger) is in effect.
- [ ] BIOS: Above 4G Decoding + Resizable BAR are on.
- [ ] `p2p-poc/benchmarks/bw_test_p2p.cu` reports sync P2P ≥ 15 GB/s.

### G1 — patches apply and build cleanly
- [ ] `git -C /root/antirez/ds4 status` shows a clean tree on `feat/tp-ssd-experiment` at HEAD `85512d5`.
- [ ] `for p in p2p-poc/patches/0*.patch; do git apply --check $p || echo FAIL:$p; done` prints no FAIL.
- [ ] `for p in p2p-poc/patches/0*.patch; do git apply $p; done`
- [ ] `PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda make -j$(nproc) ds4-server CUDA_ARCH=sm_89` succeeds.
- [ ] `PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda make -j$(nproc) ds4 CUDA_ARCH=sm_89` also succeeds (CLI client).

### G2 — startup banner
- [ ] On startup, stderr contains:
      `ds4: peer access matrix (validated): 0->1 P2P 1->0 P2P`
- [ ] On first decode-layer load, stderr contains:
      `ds4[load-pool] L2 cache ready on tier 1 capacity=NNNN experts (p2p=1)`

### G3 — correctness, no P2P regression
Run on the no-P2P hardware (4090 D) with all patches applied and
`DS4_CUDA_STREAMING_NO_L2=1` to confirm the inert-by-default path still
matches the committed baseline:
- [ ] `8+8=16` (max_tokens=80).
- [ ] Count `1,2,...,30` (max_tokens=200).
- [ ] Essay generation looks coherent over 248 tokens.
- [ ] tok/s is within ±5% of the HEAD baseline (~6.5 tok/s).

### G4 — correctness, P2P on
Same correctness suite on P2P-capable hardware with L2 enabled (default):
- [ ] `8+8=16`.
- [ ] Count `1,2,...,30`.
- [ ] Long-form generation is coherent (no degenerate repetition).
- [ ] **Critical:** no gibberish output across 1000+ generated tokens
      (this would indicate a race condition in slot claiming — see
      commit 85512d5 for the prior history).

### G5 — performance
On P2P-capable hardware, 248-token generations, warm after 5+ runs:
- [ ] Warm tok/s ≥ 2× the no-L2 baseline on the same hardware.
- [ ] `ds4[load-pool] summary` shows `l2_hits / l2_misses` ratio growing
      over time; after 500 tokens `l2_hits` should dominate `ssd_reads`.
- [ ] GPU1 SM% rises from baseline 5–10% to ≥ 25% (indicating the L2
      peer-copies are actually happening).
- [ ] GPU0 PCIe rx roughly unchanged (it's still doing SSD reads for the
      first-time misses); GPU1 PCIe rx rises by ~expert-bytes ×
      l2_writes / window.

### G6 — stability
- [ ] 30-minute continuous generation: no stalls, no crashes, no OOM.
- [ ] Peak host pinned memory growth ≤ 1 GB (async writer queue cap).
- [ ] `dmesg` after the run shows no Xid errors.

### G7 — rollback
- [ ] `git revert` of all four patches applies cleanly back to `85512d5`.
- [ ] Rebuild succeeds.
- [ ] Server runs without the L2 banner.
- [ ] tok/s matches the pre-PoC baseline.

## Sign-off matrix

| gate | who | notes |
|------|-----|-------|
| G0   | ops / hardware owner | driver version + BIOS settings |
| G1   | dev                  | pure build check |
| G2   | dev                  | runtime probe |
| G3   | dev + QA             | correctness regression (no-P2P) |
| G4   | dev + QA             | correctness (P2P) |
| G5   | perf engineer        | tok/s + summary counters |
| G6   | SRE                  | long-run stability |
| G7   | release manager      | rollback drill |

## Known failure modes and where to look

| symptom | probable cause | where to look |
|---------|----------------|---------------|
| `L2 cache alloc failed on tier 1` | GPU1 free VRAM too small | `nvidia-smi`, reduce `--ssd-streaming-cache-experts` |
| Tok/s unchanged with P2P ON | async writer never started | check `l2_writer_started` in OBS, queue cap |
| Tok/s got *worse* with L2 ON | PCIe peer BW < expected | re-run `bw_test_p2p`, check BAR1 size |
| Gibberish output | race in primary slot publish | revert patch 0003, re-verify race fix from 85512d5 |
| `Xid 43` in dmesg | peer access fault | P2P topology changed mid-run; check `lspci -vv` |
| Memory growth | async writer queue leak | `g_load_pool.l2_writer_queue.size()` ≥ 64 sustained |
