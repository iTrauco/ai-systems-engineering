// 🚀 Memory Bandwidth Benchmark for RTX A5500 🚀
// This script measures the effective memory transfer speed of your GPU

#include <stdio.h>
#include <cuda_runtime.h>

// 📊 Simple kernel to measure memory bandwidth
__global__ void bandwidthTest(float *d_data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = d_data[idx];
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
    
    // Debug output for memoryClockRate
    printf("DEBUG: Raw memoryClockRate = %d kHz\n", prop.memoryClockRate);
    
    // 🧮 Allocate memory for benchmark
    size_t size = 256 * 1024 * 1024; // 256 MB
    size_t num_elements = size / sizeof(float);
    float *h_data, *d_data;
    
    h_data = (float*)malloc(size);
    cudaMalloc(&d_data, size);
    
    // 🔢 Initialize data
    for (size_t i = 0; i < num_elements; i++) {
        h_data[i] = (float)i;
    }
    
    // 📤 Copy data to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // 🔥 Warmup
    bandwidthTest<<<(num_elements + 255) / 256, 256>>>(d_data, num_elements);
    cudaDeviceSynchronize();
    
    // ⏱️ Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        bandwidthTest<<<(num_elements + 255) / 256, 256>>>(d_data, num_elements);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 📊 Calculate bandwidth
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    float gb_per_sec = (size * 100 * 2) / (seconds * 1e9); // *2 for read+write
    
    printf("⚡ Memory Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("🕒 Execution Time: %.2f ms\n", milliseconds);
    
    // Try multiple different theoretical bandwidth calculations to diagnose the issue
    
    // Approach 1: Using memory clock in kHz directly
    float theoretical_bw1 = (prop.memoryClockRate / 1000.0f) * (prop.memoryBusWidth / 8.0f) * 2.0f / 1000.0f;
    printf("📈 Theoretical Bandwidth (Approach 1): %.2f GB/s\n", theoretical_bw1);
    printf("🔍 Efficiency (Approach 1): %.2f%%\n", (gb_per_sec / theoretical_bw1) * 100.0f);
    
    // Approach 2: Using memory clock in MHz (assuming prop.memoryClockRate is in kHz)
    float clock_mhz = prop.memoryClockRate / 1000.0f;
    float bytes_per_second = clock_mhz * 1000000.0f * (prop.memoryBusWidth / 8.0f) * 2.0f;
    float theoretical_bw2 = bytes_per_second / 1000000000.0f;
    printf("📈 Theoretical Bandwidth (Approach 2): %.2f GB/s\n", theoretical_bw2);
    printf("🔍 Efficiency (Approach 2): %.2f%%\n", (gb_per_sec / theoretical_bw2) * 100.0f);
    
    // Approach 3: Using hardcoded values for RTX A5500
    float theoretical_bw3 = 768.0f; // 8.0 GHz * 48 bytes * 2
    printf("📈 Theoretical Bandwidth (Approach 3 - Hardcoded): %.2f GB/s\n", theoretical_bw3);
    printf("🔍 Efficiency (Approach 3): %.2f%%\n", (gb_per_sec / theoretical_bw3) * 100.0f);
    
    // Approach 4: Original calculation
    float theoretical_bw4 = (prop.memoryClockRate * 1e6) * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("📈 Theoretical Bandwidth (Original): %.2f GB/s\n", theoretical_bw4);
    
    // Approach 5: Manual calculation with explicit conversions
    float clock_ghz = prop.memoryClockRate / 1000000.0f;  // kHz to GHz
    float bus_bytes = prop.memoryBusWidth / 8.0f;         // bits to bytes
    float transfers = 2.0f;                               // DDR
    float theoretical_bw5 = clock_ghz * bus_bytes * transfers;
    printf("📈 Theoretical Bandwidth (Manual): %.2f GB/s (%.2f GHz * %.1f bytes * %.1f)\n", 
           theoretical_bw5, clock_ghz, bus_bytes, transfers);
    
    // 🧹 Cleanup
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}