// 🚀 Memory Bandwidth Benchmark for RTX A5500
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void bandwidthTest(float *d_data, size_t size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < size) {
       float val = d_data[idx];
       d_data[idx] = val + 1.0f;
   }
}

int main() {
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0);
   printf("🖥️  Device: %s\n", prop.name);
   printf("💻 CUDA Capability: %d.%d\n", prop.major, prop.minor);
   printf("🧠 Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
   printf("🔄 Memory Bus Width: %d bits\n", prop.memoryBusWidth);
   printf("⏱️  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
   
   size_t size = 256 * 1024 * 1024; // 256 MB
   size_t num_elements = size / sizeof(float);
   float *h_data, *d_data;
   
   h_data = (float*)malloc(size);
   cudaMalloc(&d_data, size);
   
   for (size_t i = 0; i < num_elements; i++) {
       h_data[i] = (float)i;
   }
   
   cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
   
   // Warmup
   bandwidthTest<<<(num_elements + 255) / 256, 256>>>(d_data, num_elements);
   cudaDeviceSynchronize();
   
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   for (int i = 0; i < 100; i++) {
       bandwidthTest<<<(num_elements + 255) / 256, 256>>>(d_data, num_elements);
   }
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   float seconds = milliseconds / 1000.0f;
   float gb_per_sec = (size * 100 * 2) / (seconds * 1e9); // *2 for read+write
   
   printf("⚡ Memory Bandwidth: %.2f GB/s\n", gb_per_sec);
   printf("🕒 Execution Time: %.2f ms\n", milliseconds);
   
   float clock_ghz = prop.memoryClockRate / 1000000.0f;  // kHz to GHz
   float bus_bytes = prop.memoryBusWidth / 8.0f;         // bits to bytes
   float transfers = 2.0f;                               // DDR
   float theoretical_bw = clock_ghz * bus_bytes * transfers;
   
   printf("📈 Theoretical Bandwidth: %.2f GB/s\n", theoretical_bw);
   printf("🔍 Efficiency: %.2f%%\n", (gb_per_sec / theoretical_bw) * 100.0f);
   
   cudaFree(d_data);
   free(h_data);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   
   return 0;
}