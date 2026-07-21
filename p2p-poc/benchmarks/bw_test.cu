// BOUNCE bandwidth microbenchmark — current state on RTX 4090 D
//
// Build:
//   nvcc -arch=sm_89 -o bw_test bw_test.cu
// Run:
//   ./bw_test
//
// Expected output on hardware where P2P is BLOCKED (RTX 4090 D, GeForce driver):
//   cudaDeviceCanAccessPeer(0->1) = 0
//   cudaMemcpyPeer sync:           3.16 GB/s   (1.99 ms per 6 MB copy)
//   BOUNCE async (per-stage stream): 0.39 GB/s (16.1 ms per copy)
//   cudaMemcpyPeerAsync:            0.71 GB/s  (8.85 ms per copy)
//   1 thread, 6 sequential:         0.94 GB/s  (40 ms per batch of 6)
//   4 threads, 6 experts in parallel: 0.71 GB/s (53 ms per batch)
//
// See analysis/BOUNCE_BOTTLENECK.md for the interpretation.

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

    const size_t bytes = 6 * 1024 * 1024;  // 6 MB - typical expert size
    char *src, *dst;

    cudaSetDevice(0);
    cudaMalloc(&src, bytes);
    cudaMemset(src, 0xAA, bytes);

    cudaSetDevice(1);
    cudaMalloc(&dst, bytes);

    cudaSetDevice(0);

    // Warmup
    cudaMemcpyPeer(dst, 1, src, 0, bytes);

    // Test 1: sync cudaMemcpyPeer
    double t0 = now_sec();
    int N = 100;
    for (int i = 0; i < N; i++) {
        cudaMemcpyPeer(dst, 1, src, 0, bytes);
    }
    double t1 = now_sec();
    printf("cudaMemcpyPeer sync: %.2f GB/s (per-copy %.2f ms)\n",
           N * bytes / (t1 - t0) / 1e9, (t1 - t0) / N * 1000);

    // Test 2: BOUNCE async (explicit D2H+H2D via pinned host)
    cudaStream_t s0, s1;
    cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

    char *bounce;
    cudaMallocHost(&bounce, bytes);

    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);

    t0 = now_sec();
    for (int i = 0; i < N; i++) {
        cudaSetDevice(0);
        cudaMemcpyAsync(bounce, src, bytes, cudaMemcpyDeviceToHost, s0);
        cudaEventRecord(ev, s0);
        cudaSetDevice(1);
        cudaStreamWaitEvent(s1, ev, 0);
        cudaMemcpyAsync(dst, bounce, bytes, cudaMemcpyHostToDevice, s1);
        cudaEventRecord(ev, s1);
        cudaSetDevice(0);
        cudaStreamWaitEvent(s0, ev, 0);
    }
    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);
    t1 = now_sec();
    printf("BOUNCE async (per-stage stream): %.2f GB/s (per-copy %.2f ms)\n",
           N * bytes / (t1 - t0) / 1e9, (t1 - t0) / N * 1000);

    // Test 3: cudaMemcpyPeerAsync on a single stream
    t0 = now_sec();
    for (int i = 0; i < N; i++) {
        cudaMemcpyPeerAsync(dst, 1, src, 0, bytes, s0);
    }
    cudaStreamSynchronize(s0);
    t1 = now_sec();
    printf("cudaMemcpyPeerAsync: %.2f GB/s (per-copy %.2f ms)\n",
           N * bytes / (t1 - t0) / 1e9, (t1 - t0) / N * 1000);

    // Test 4: multi-threaded sync
    const int N_EXP = 6;
    char *srcs[N_EXP], *dsts[N_EXP];
    for (int i = 0; i < N_EXP; i++) {
        cudaSetDevice(0); cudaMalloc(&srcs[i], bytes); cudaMemset(srcs[i], i, bytes);
        cudaSetDevice(1); cudaMalloc(&dsts[i], bytes);
    }
    cudaSetDevice(0);

    auto worker = [&](int tid, int n_threads) {
        for (int iter = 0; iter < N; iter++) {
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
        printf("%d thread(s), %d sequential cudaMemcpyPeer: %.2f GB/s (per-batch %.2f ms)\n",
               n_threads, N_EXP,
               N * N_EXP * bytes / (t1 - t0) / 1e9, (t1 - t0) / N * 1000);
    }

    return 0;
}
