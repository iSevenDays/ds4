# Profiling baseline (captured before this PoC)

Snapshot of the system state that produced the 6.5 tok/s baseline
referenced throughout this folder. Captured on `feat/tp-ssd-experiment`
at commit `85512d5` ("feat(ssd): per-expert race fix + load-pool
observability").

## Launch line

```bash
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=4 ./ds4-server \
  --cuda --cuda-tensor-parallel --ssd-streaming \
  --ssd-streaming-cache-experts 40GB --gpu-vram 45,45 --ctx 262144 \
  -m /root/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --mtp /root/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf --dspark \
  --host 0.0.0.0 --port 8018
```

## Hardware

```
2x NVIDIA GeForce RTX 4090 D, 49140 MiB each
PCIe gen4 x16 (no NVLink bridge)
ZFS ARC 27 GB (out of 86 GB model file)
CPU: enough threads, not bottleneck
SSD: 1.7 GB/s single-thread pread, 5-6 GB/s with 4-8 parallel preads
Host RAM: 34 GB
```

## Steady-state tok/s (warm)

```
Run 1: 6.14 tok/s
Run 2: 6.63 tok/s
Run 3: 6.27 tok/s
Run 4: 6.56 tok/s
Run 5: 6.59 tok/s
Run 6: 7.00 tok/s
Run 7: 6.83 tok/s
Run 8: 6.60 tok/s
-----------
median: 6.6 tok/s
```

## Per-stage decode averages (DS4_METAL_DECODE_STAGE_PROFILE=1)

```
decode/routed_moe                            avg=2.189 ms / layer    DOMINANT
decode/router                               avg=0.299 ms / layer
decode/attn_output                          avg=0.250 ms / layer
decode/q_path                               avg=0.097 ms / layer
decode/attn_hc_pre                          avg=0.072 ms / layer
decode/compressor_proj                      avg=0.067 ms / layer
decode/ffn_hc_pre                           avg=0.061 ms / layer
decode/shared_gate_up                       avg=0.053 ms / layer
decode/shared_down                          avg=0.040 ms / layer
decode/attn_inv_rope                        avg=0.040 ms / layer
decode/kv_path                              avg=0.033 ms / layer
decode/attn_hc_post                         avg=0.026 ms / layer
decode/compressor_update                    avg=0.023 ms / layer
decode/indexer_compressor_proj              avg=0.044 ms / layer
decode/attn_norm                            avg=0.013 ms / layer
decode/ffn_norm                             avg=0.012 ms / layer
decode/indexer_compressor_update            avg=0.022 ms / layer
decode/ffn_hc_post                          avg=0.010 ms / layer
decode/compressor_indexer                   avg=0.009 ms / layer
```

Sum of avg per layer ≈ 3.6 ms. × 40 layers = 144 ms / token = 6.9 tok/s.
Matches measured median.

## Load-pool summary line

```
ds4[load-pool] summary layers=91500 experts=549000 hit_rate=87.1%
  ssd_reads=70848 ssd_gb=467.02 phase1_us/expert=53.1
  phase2_us/expert=597.7 workers=4
```

Translation:
- Primary LRU hit rate: 87.1%
- SSD reads: 70848 over 91500 layer-loads = 0.77 SSD reads / layer
- Per-miss SSD read time: 598 / 0.13 ≈ 4.5 ms
- Phase 1 (cache lookup under lock): 53 µs / expert
- Workers: 4

## GPU utilisation (dmon)

```
gpu  sm%  mem%  pwr   fb(MB)   rxpci(MB/s)  txpci(MB/s)
0    28%   9%   89W   47768    1.7–3.2       0.2
1     5%   1%   61W   10782    0.02          0.01
```

GPU1 has ~38 GB free VRAM and 5–10% SM utilisation. This is the
resource the L2 cache targets.

## Memory snapshot

```
GPU0: 44650 / 49140 MiB used    (3888 MiB free)
GPU1: 10782 / 49140 MiB used    (37756 MiB free)
Host: 1.8 GB used / 34 GB total  (ZFS ARC ≈ 27 GB)
```

## P2P matrix (from startup log)

```
ds4: peer access matrix (validated): 0->1 BOUNCE 1->0 BOUNCE
```

This is what every projection in `EXPECTED_RESULTS.md` is gated to
*flip* to `0->1 P2P 1->0 P2P`.
