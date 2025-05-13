// 🧮 Compute Throughput Benchmark for RTX A5500
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to measure FP32 compute throughput using fused multiply-add operations
__global__ void computeThroughputFP32(float *d_data, size_t n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = d_data[idx];
        float y = x + 0.1f;
        float z = x - 0.1f;
        
        // Perform many arithmetic operations to measure compute throughput
        // Using fused multiply-add (FMA) operations that execute in a single cycle
        #pragma unroll 32
        for (int i = 0; i < iterations; i++) {
            // Each line is 4 FMA operations (8 FLOPS)
            x = __fmaf_rn(x, y, z); z = __fmaf_rn(z, x, y);
            y = __fmaf_rn(y, z, x); x = __fmaf_rn(x, y, z);
            z = __fmaf_rn(z, x, y); y = __fmaf_rn(y, z, x);
            x = __fmaf_rn(x, y, z); z = __fmaf_rn(z, x, y);
        }
        
        // Write result back to prevent compiler from optimizing away the computation
        d_data[idx] = x + y + z;
    }
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("🖥️  Device: %s\n", prop.name);
    printf("💻 CUDA Capability: %d.%d\n", prop.major, prop.minor);
    printf("⚙️  CUDA Cores: %d\n", prop.multiProcessorCount * 128); // Approximation for Ampere
    printf("⏱️  Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    
    // Allocate memory
    size_t n = 4 * 1024 * 1024; // 4M elements
    size_t size = n * sizeof(float);
    float *h_data, *d_data;
    h_data = (float*)malloc(size);
    cudaMalloc(&d_data, size);
    
    // Initialize data
    for (size_t i = 0; i < n; i++) {
        h_data[i] = 1.0f + (float)i / n;
    }
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // Number of iterations inside the kernel
    int iterations = 100;
    // FLOPS per iteration per thread: 8 FMA ops * 2 FLOPS per FMA = 16 FLOPS
    long long flops_per_thread = iterations * 8 * 2;
    long long total_flops = flops_per_thread * n;
    
    // Warmup
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    computeThroughputFP32<<<gridSize, blockSize>>>(d_data, n, iterations);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    computeThroughputFP32<<<gridSize, blockSize>>>(d_data, n, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate results
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    float tflops = total_flops / (seconds * 1e12);
    
    printf("⚡ FP32 Compute Throughput: %.2f TFLOPS\n", tflops);
    printf("🕒 Execution Time: %.2f ms\n", milliseconds);
    
    // Theoretical peak throughput

    float peak_tflops = (prop.multiProcessorCount * 128) * 2 * (prop.clockRate / 1e6) / 1000.0f;

    printf("📈 Theoretical Peak: %.2f TFLOPS\n", peak_tflops);
    printf("🔍 Efficiency: %.2f%%\n", (tflops / peak_tflops) * 100.0f);
    
    // Copy result back to verify
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    printf("✅ Result verification: %f\n", h_data[0]);
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}