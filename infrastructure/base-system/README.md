# Base System Configuration

This component establishes the foundational environment for AI/HPC workloads on Debian XFCE with NVIDIA GPU support.

## Objective
Establish a stable development environment with full GPU functionality for AI/ML workloads.

## Component Details

### Hardware Layer
- **Lenovo Thinkstation P5**: High-performance workstation with Xeon W5-2445 processor, 32GB ECC RAM, and NVIDIA RTX A5500 GPU providing the physical computing foundation.

### Operating System
- **Debian XFCE**: Lightweight Linux distribution optimized for HPC workloads, providing stability and performance without overhead.

### Driver Layer
- **NVIDIA Drivers**: Provides low-level hardware access to the GPU, enabling CUDA and compute workloads.
- **CUDA Toolkit**: Development kit for GPU-accelerated applications, including compiler tools, runtime libraries, and development APIs.

### Libraries
- **OpenMPI**: Implementation of the Message Passing Interface (MPI) standard for parallel computing across distributed systems.
- **OpenBLAS/LAPACK**: Optimized linear algebra packages for high-performance numerical computing.
- **Python Environment**: Runtime environment for high-level scripting and AI framework access.

### Development Tools
- **Core Build Environment**: Essential compilation and build tools (gcc, cmake, git) for development workflows.

### Validation Testing
- **CUDA Vector Addition Test**: Simple CUDA program that verifies proper GPU functionality by performing vector calculations on the device.

## Usage
```bash
# Run as root or with sudo
sudo ./setup_environment.sh
```

## Status: Implemented

## Resources
- [Debian Science Project](https://wiki.debian.org/DebianScience)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [OpenMPI Documentation](https://www.open-mpi.org/doc/)
- [OpenBLAS GitHub](https://github.com/xianyi/OpenBLAS)
