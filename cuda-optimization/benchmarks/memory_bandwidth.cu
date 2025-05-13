// 🚀 Memory Bandwidth Benchmark for RTX A5500 🚀
// This script measures the effective memory transfer speed of your GPU
// by performing read/write operations to global memory.

#include <stdio.h>
#include <cuda_runtime.h>

// 📊 Simple kernel to measure memory bandwidth
// This kernel performs a load and store operation on each element
// which allows us to measure the real-world memory throughput
__global__ void bandwidthTest(float *d_data, size_t size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // 📝 Simple load/store operations to measure bandwidth
        // 1. Read value from global memory (memory read)
        float val = d_data[idx];
        // 2. Perform minimal computation
        // 3. Write back to global memory (memory write)
        d_data[idx] = val + 1.0f;
    }
}

int main() {
    // 📋 Print device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("🖥️  Device: %s\n", prop.name);
    printf("💻 CUDA Capability: %d.%d\n", prop.major, prop.minor);
    printf("🧠 Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("🔄 Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("⏱️  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    
    // 🧮 Allocate memory for benchmark
    // Using a large buffer to ensure we measure sustained bandwidth
    size_t size = 256 * 1024 * 1024; // 256 MB
    size_t num_elements = size / sizeof(float);
    float *h_data, *d_data;
    
    // Allocate host memory
    h_data = (float*)malloc(size);
    
    // Allocate device memory
    cudaMalloc(&d_data, size);
    
    // 🔢 Initialize data with sequential values
    for (size_t i = 0; i < num_elements; i++) {
        h_data[i] = (float)i;
    }
    
    // 📤 Copy data to device
    // Initial transfer not included in the benchmark
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // 🔥 Warmup run to ensure GPU is at full performance state
    // 256 threads per block is a common choice for good occupancy
    bandwidthTest<<<(num_elements + 255) / 256, 256>>>(d_data, num_elements);
    cudaDeviceSynchronize();
    
    // ⏱️ Setup CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 📏 Start measurement
    cudaEventRecord(start);
    
    // Run multiple iterations for better measurement accuracy
    // This helps average out any timing variations
    for (int i = 0; i < 100; i++) {
        bandwidthTest<<<(num_elements + 255) / 256, 256>>>(d_data, num_elements);
    }
    
    // 📏 Stop measurement
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 📊 Calculate bandwidth
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    
    // Each element requires one read and one write (2 operations)
    // size * 100 iterations * 2 operations
    float gb_per_sec = (size * 100 * 2) / (seconds * 1e9); // *2 for read+write
    
    // 📢 Output results
    printf("⚡ Memory Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("🕒 Execution Time: %.2f ms\n", milliseconds);
    
    // Compare to theoretical bandwidth
    float theoretical_bw = (prop.memoryClockRate * 1e6) * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("📈 Theoretical Bandwidth: %.2f GB/s\n", theoretical_bw);
    printf("🔍 Efficiency: %.2f%%\n", (gb_per_sec / theoretical_bw) * 100.0f);
    
    // 🧹 Cleanup
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}