#!/bin/bash
# Base System Configuration for AI/HPC Environment
# System: Debian XFCE with NVIDIA RTX A5500

# Exit on error
set -e

# Log all output
exec > >(tee -i base_system_setup.log)
exec 2>&1

echo "Starting base system configuration for AI/HPC environment"
echo "System: Debian XFCE with NVIDIA RTX A5500"
echo "Timestamp: $(date)"

# Create backup of bashrc before modification
cp ~/.bashrc ~/.bashrc.bak
echo "Created backup of ~/.bashrc to ~/.bashrc.bak"

# Update system and install essential packages
echo "Updating system packages..."
apt-get update && apt-get upgrade -y

echo "Installing development tools..."
apt-get install -y build-essential git cmake autoconf automake gfortran
if [ $? -ne 0 ]; then
    echo "Failed to install development tools. Exiting."
    exit 1
fi

echo "Installing HPC libraries..."
apt-get install -y libopenmpi-dev libopenblas-dev liblapack-dev
if [ $? -ne 0 ]; then
    echo "Failed to install HPC libraries. Exiting."
    exit 1
fi

echo "Installing Python development tools..."
apt-get install -y python3-dev python3-pip python3-venv
if [ $? -ne 0 ]; then
    echo "Failed to install Python tools. Exiting."
    exit 1
fi

# NVIDIA driver verification
echo "Checking NVIDIA driver installation"
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers found:"
    nvidia-smi
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    echo "Driver version: $DRIVER_VERSION"
else
    echo "NVIDIA drivers not found. Installing recommended drivers..."
    apt-get install -y nvidia-driver
    if [ $? -ne 0 ]; then
        echo "Failed to install NVIDIA drivers. Exiting."
        exit 1
    fi
    
    # Verify driver installation was successful
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Driver installation failed. Please install manually and retry."
        exit 1
    fi
    
    echo "NVIDIA drivers installed successfully:"
    nvidia-smi
fi

# CUDA toolkit installation
echo "Installing CUDA toolkit"
apt-get install -y nvidia-cuda-toolkit
if [ $? -ne 0 ]; then
    echo "Failed to install CUDA toolkit. Exiting."
    exit 1
fi

# Verify CUDA installation
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
    echo "CUDA version: $CUDA_VERSION"
else
    echo "CUDA installation failed. Please check your installation."
    exit 1
fi

# Configure environment variables
echo "Configuring environment variables"
if ! grep -q "export PATH=/usr/local/cuda/bin:\$PATH" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
fi

if ! grep -q "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Run a simple CUDA test to verify functionality
echo "Running simple CUDA test..."
mkdir -p ~/cuda_test
cat > ~/cuda_test/vector_add.cu << 'EOF'
#include <stdio.h>

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Allocate and initialize host memory
    const int N = 1000000;
    size_t size = N * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_out, size);
    
    // Copy from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_out, d_a, d_b, N);
    
    // Copy from device to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < 5; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_out[i]);
    }
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);
    
    printf("CUDA test completed successfully\n");
    return 0;
}
EOF

cd ~/cuda_test
nvcc -o vector_add vector_add.cu
if [ $? -ne 0 ]; then
    echo "CUDA compilation failed. Please check your CUDA installation."
    exit 1
fi

./vector_add
if [ $? -ne 0 ]; then
    echo "CUDA test execution failed. Please check your GPU configuration."
    exit 1
fi

echo "Base system configuration completed successfully"
echo "Timestamp: $(date)"
echo "To roll back changes, run the rollback_environment.sh script"