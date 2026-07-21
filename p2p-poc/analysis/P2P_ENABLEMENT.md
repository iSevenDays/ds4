# How to enable P2P peer access

This document is the concrete checklist for taking the patches in this
folder from "inert" to "active". The software side is easy; everything
else depends on what kind of GPU you can swap in.

## A. Hardware matrix

| GPU class                          | P2P-over-PCIe | NVLink bridge | Notes |
|------------------------------------|---------------|---------------|-------|
| **RTX 4090 / 4090 D (consumer)**   | blocked       | n/a           | NVIDIA driver refuses; the matrix you'll see is `0->1 BOUNCE 1->0 BOUNCE`. **This is the current state.** |
| RTX 3090 / 3090 Ti (consumer)      | blocked       | partially     | NVLink bridge *was* allowed on Turing/Ampere consumer; recent drivers lock it down too. |
| **RTX 6000 Ada / RTX A6000 (pro)** | allowed       | n/a (Ada)     | Single-slot cards, no NVLink, but P2P-over-PCIe works. ~24 GB/s measured. |
| RTX A100 (PCIe)                    | allowed       | optional      | NVLink bridge optional; P2P-over-PCIe still works at ~22 GB/s. |
| RTX A100 (SXM4)                    | allowed       | yes (NVSwitch)| 600 GB/s aggregate, ~150 GB/s per pair. |
| H100 (PCIe)                        | allowed       | n/a           | ~24 GB/s. |
| H100 (SXM5)                        | allowed       | yes            | 900 GB/s aggregate. |

**Rule of thumb:** GeForce = no P2P; RTX pro / Tesla / Quadro = yes P2P.
Driver strings: GeForce driver (`PCI_DeviceName~GeForce`) returns
`cudaErrorPeerAccessUnsupported` from `cudaDeviceEnablePeerAccess`
regardless of physical capability.

## B. Driver / OS side

Assuming a pro GPU where P2P is actually allowed, do once at boot:

```bash
# 1. Make sure both GPUs are in TCC mode (Windows) or just visible (Linux).
nvidia-smi

# 2. Reserve enough BAR1 for peer mappings. The default 256 MB is too small
#    for a 40 GB L2 cache. Set to the max (1 GB on most consumer/pro cards):
sudo nvidia-smi -i 0,1 --bar1-size=1024

# 3. Persist the driver state:
sudo nvidia-smi --persistence-mode=ENABLED -i 0,1

# 4. (Optional, but recommended for big caches) Allow 64-bit BAR via BIOS:
#    - reboot into BIOS
#    - enable "Above 4G Decoding" + "Resizable BAR"
#    - on Linux, also `kernel ... pci=realloc`
```

Verify P2P is enabled:

```bash
nvcc -arch=sm_89 -o /tmp/bw_p2p p2p-poc/benchmarks/bw_test_p2p.cu
/tmp/bw_p2p
# Expect:
#   cudaDeviceCanAccessPeer(0->1) = 1
#   cudaDeviceCanAccessPeer(1->0) = 1
#   sync P2P cudaMemcpy:           24 GB/s   (not 3 GB/s)
#   async P2P cudaMemcpyPeerAsync: 26 GB/s   (not 0.7 GB/s)
```

If you see BOUNCE numbers from the P2P test, P2P is still disabled — go
back to step 2/3.

## C. On 4090 D specifically

If you genuinely cannot swap the GPUs, your realistic options are:

1. **Driver-patch route (third-party).** Some community projects (e.g.
   `nvidia-patch`, `nvml-unlock`) claim to flip the P2P-veto bit in the
   driver. The patches here are still useful in that scenario, but I
   cannot vouch for stability, and you would definitely void any
   support contract. Buyer beware.
2. **Move SSD bandwidth, not VRAM.** Use GPU1's SSD-port peer-DMA
   capability to read directly into GPU1's HBM (so each GPU has its own
   producer). That's `feat/mgpu-ssd` territory and is the
   pipeline-parallel path, not TP+SSD. Different branch, different
   design.
3. **Accept the L2 idea is gated on hardware swap.** That is what
   this PoC assumes.

## D. What the d4 codebase detects automatically

The startup probe at `ds4_cuda.cu:2280–2360` already records the
peer-access matrix into `g_gpu_peer_ok[][]`. With the patches in this
folder, the L2 cache auto-enables whenever:

```c
g_n_gpus >= 2
&& g_gpu_peer_ok[primary_tier][l2_tier]
&& g_gpu_peer_ok[l2_tier][primary_tier]
&& getenv("DS4_CUDA_STREAMING_NO_L2") == NULL;
```

No env var required. If P2P is not enabled at the driver level,
`g_gpu_peer_ok[0][1] == 0`, the L2 is silently skipped, and behaviour
matches the committed baseline.

## E. Smoke test

After applying the patches and rebuilding, run the ds4 server normally
and grep stderr for the L2-ready banner:

```bash
./ds4-server --cuda --cuda-tensor-parallel --ssd-streaming ... 2>&1 | \
  grep "L2 cache ready"
# P2P ON:  ds4[load-pool] L2 cache ready on tier 1 capacity=4986 experts (p2p=1)
# P2P OFF: (no line — L2 disabled silently)
```

The `p2p=1` suffix is the explicit confirmation that the runtime saw
the peer-access matrix say P2P and proceeded to allocate.
