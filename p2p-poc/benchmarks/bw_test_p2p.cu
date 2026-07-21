// P2P bandwidth microbenchmark — run AFTER enabling P2P
//
// Build:
//   nvcc -arch=sm_89 -o bw_test_p2p bw_test_p2p.cu
// Run:
//   ./bw_test_p2p
//
// Expected output on P2P-capable hardware (RTX 6000 Ada, A100, H100):
//   cudaDeviceCanAccessPeer(0->1) = 1
//   cudaDeviceCanAccessPeer(1->0) = 1
//   cudaDeviceEnablePeerAccess(1 on 0) = cudaSuccess
//   cudaDeviceEnablePeerAccess(0 on 1) = cudaSuccess
//   sync P2P cudaMemcpyPeer:           24 GB/s   (0.25 ms per 6 MB copy)
//   async P2P cudaMemcpyPeerAsync:     26 GB/s   (0.23 ms per copy)
//   multi-thread sync (4 threads):     24 GB/s   (saturates at PCIe peer BW)
//
// If you see numbers close to those in bw_test.cu (3 GB/s or lower), P2P
// is NOT enabled. Go back to analysis/P2P_ENABLEMENT.md and re-check.

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <vector>

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev < 2) { printf("need 2 gpus\n"); return 1; }

    int can01 = 0, can10 = 0;
    cudaDeviceCanAccessPeer(&can01, 0, 1);
    cudaDeviceCanAccessPeer(&can10, 1, 0);
    printf("cudaDeviceCanAccessPeer(0->1) = %d\n", can01);
    printf("cudaDeviceCanAccessPeer(1->0) = %d\n", can10);
    if (!can01 || !can10) {
        printf("P2P is NOT supported by the driver on this hardware.\n");
        printf("On GeForce consumer cards this is permanent; on pro cards check\n");
        printf("that persistence-mode is on and BAR1 is large enough (>= slab size).\n");
        return 2;
    }

    // Explicitly enable P2P in both directions.
    cudaError_t e1 = cudaDeviceEnablePeerAccess(1, 0);
    printf("cudaDeviceEnablePeerAccess(1 on 0) = %s\n",
           cudaGetErrorString(e1));
    cudaSetDevice(1);
    cudaError_t e2 = cudaDeviceEnablePeerAccess(0, 0);
    printf("cudaDeviceEnablePeerAccess(0 on 1) = %s\n",
           cudaGetErrorString(e2));
    cudaSetDevice(0);

    const size_t bytes = 6 * 1024 * 1024;  // 6 MB - one expert
    const int N_EXP = 6;                    // experts per layer
    const int N_ITER = 1000;

    char *srcs[N_EXP], *dsts[N_EXP];
    cudaSetDevice(0);
    for (int i = 0; i < N_EXP; i++) {
        cudaMalloc(&srcs[i], bytes);
        cudaMemset(srcs[i], i, bytes);
    }
    cudaSetDevice(1);
    for (int i = 0; i < N_EXP; i++) {
        cudaMalloc(&dsts[i], bytes);
    }
    cudaSetDevice(0);

    // Warmup
    for (int i = 0; i < N_EXP; i++) {
        cudaMemcpyPeer(dsts[i], 1, srcs[i], 0, bytes);
    }

    // Test 1: sync cudaMemcpyPeer with P2P enabled.
    double t0 = now_sec();
    for (int iter = 0; iter < N_ITER; iter++) {
        for (int i = 0; i < N_EXP; i++) {
            cudaMemcpyPeer(dsts[i], 1, srcs[i], 0, bytes);
        }
    }
    double t1 = now_sec();
    printf("sync P2P cudaMemcpyPeer (%d experts x %d iter): %.2f GB/s (per-expert %.3f ms)\n",
           N_EXP, N_ITER,
           N_ITER * N_EXP * bytes / (t1 - t0) / 1e9,
           (t1 - t0) / (N_ITER * N_EXP) * 1000);

    // Test 2: async via cudaMemcpyPeerAsync on a single stream.
    cudaStream_t s0;
    cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
    t0 = now_sec();
    for (int iter = 0; iter < N_ITER; iter++) {
        for (int i = 0; i < N_EXP; i++) {
            cudaMemcpyPeerAsync(dsts[i], 1, srcs[i], 0, bytes, s0);
        }
    }
    cudaStreamSynchronize(s0);
    t1 = now_sec();
    printf("async P2P cudaMemcpyPeerAsync: %.2f GB/s (per-expert %.3f ms)\n",
           N_ITER * N_EXP * bytes / (t1 - t0) / 1e9,
           (t1 - t0) / (N_ITER * N_EXP) * 1000);

    // Test 3: multi-threaded sync, like the worker pool would do.
    auto worker = [&](int tid, int n_threads) {
        for (int iter = 0; iter < N_ITER; iter++) {
            for (int i = tid; i < N_EXP; i += n_threads) {
                cudaMemcpyPeer(dsts[i], 1, srcs[i], 0, bytes);
            }
        }
    };
    for (int n_threads : {1, 2, 4, 6}) {
        t0 = now_sec();
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back(worker, t, n_threads);
        }
        for (auto& th : threads) th.join();
        t1 = now_sec();
        printf("%d-thread sync P2P: %.2f GB/s (per-batch-of-%d %.3f ms)\n",
               n_threads,
               N_ITER * N_EXP * bytes / (t1 - t0) / 1e9,
               N_EXP,
               (t1 - t0) / N_ITER * 1000);
    }

    printf("\n");
    printf("Verdict:\n");
    printf("  If sync P2P >= 15 GB/s -> L2 cache patches will pay off.\n");
    printf("  If sync P2P 4-15 GB/s -> L2 helps, but watch the per-miss critical path.\n");
    printf("  If sync P2P < 4 GB/s  -> P2P is not actually enabled. L2 will regress.\n");
    return 0;
}
