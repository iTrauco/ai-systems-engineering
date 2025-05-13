#!/bin/bash
# Validation script for base system configuration

echo "Validating system environment..."

# Check NVIDIA drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA drivers not found"
    exit 1
fi

# Check CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA toolkit not found"
    exit 1
fi

# Check OpenMPI
if ! command -v mpirun &> /dev/null; then
    echo "ERROR: OpenMPI not found"
    exit 1
fi

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Check library availability
echo "Checking libraries..."
python3 -c "import numpy; print('NumPy version:', numpy.__version__)" || { echo "ERROR: NumPy not found"; exit 1; }

# Compile and run a simple CUDA program
echo "Testing CUDA compilation..."
cd "$(mktemp -d)"
cat > test.cu << 'CUDA'
#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
CUDA

nvcc -o test test.cu
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA compilation failed"
    exit 1
fi

./test
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA execution failed"
    exit 1
fi

# Test OpenMPI
echo "Testing OpenMPI..."
cat > mpi_test.c << 'MPI'
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Hello from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    MPI_Finalize();
    return 0;
}
MPI

mpicc -o mpi_test mpi_test.c
if [ $? -ne 0 ]; then
    echo "ERROR: MPI compilation failed"
    exit 1
fi

mpirun -np 2 ./mpi_test
if [ $? -ne 0 ]; then
    echo "ERROR: MPI execution failed"
    exit 1
fi

echo "All validation tests passed!"
exit 0