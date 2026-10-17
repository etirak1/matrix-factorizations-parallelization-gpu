# matrix-factorizations-parallelization-gpu
Showcasing different ways of how we can use parallelization on GPU (CUDA) in order to speed up crucial algorithms in modern computing - matrix factorizations.

# GPU-Accelerated Cholesky Matrix Factorization with CUDA

This repository contains the GPU-accelerated implementation of the **Cholesky Factorization** algorithm ($A = L \cdot L^T$) for symmetric positive-definite matrices, implemented in **C/C++ and NVIDIA CUDA**.

The goal of this project is to leverage massive parallel thread concurrency and the memory hierarchy of modern GPUs to achieve substantial speedups over traditional sequential and multi-threaded CPU implementations, especially for large-scale matrices.

---

## 📌 Algorithmic Overview

Cholesky decomposition factorizes a symmetric, positive-definite matrix $A \in \mathbb{R}^{n \times n}$ into:

$$A = L \cdot L^T$$

where $L$ is a lower triangular matrix with strictly positive diagonal elements.

### GPU Parallelization Strategy:
While the diagonal elements impose a serial dependency step, the column updates and trailing submatrix trailing updates exhibit high data independence. The GPU implementation optimizes this through:
* **Kernel Parallelism:** Mapping independent element/column operations across 1D and 2D CUDA thread blocks (`<<<grid, block>>>`).
* **Memory Optimization:** Maximizing coalesced global memory access patterns and minimizing branch divergence across warps.
* **Host-Device Transfers:** Profiling and minimizing PCIe overhead during `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost` stages.

---

## 🚀 Key Features

* **CUDA Kernel Implementation:** Customized CUDA kernels for parallel column scaling and submatrix reduction.
* **CPU vs. GPU Benchmarking:** Comparative performance analysis measuring kernel execution time versus host sequential/OpenMP baseline.
* **Scalability Analysis:** Evaluates throughput and speedup across varying matrix dimensions ($N = 500, 1000, 2000, 4000+$).
* **Numerical Verification:** Validates matrix reconstruction error ($\|A - L \cdot L^T\|$) to ensure numerical stability and floating-point accuracy.

---

## 🛠️ Tech Stack & Requirements

* **Language:** C / C++
* **GPU Computing Platform:** NVIDIA CUDA
* **Compiler:** NVIDIA CUDA Compiler (`nvcc`)
* **Hardware:** CUDA-capable NVIDIA GPU
* **OS:** Linux / Windows (with CUDA Toolkit installed)

---

## ⚙️ Build & Run

### 1. Clone the repository and navigate to the branch:
```bash
git clone https://github.com/etirak1/matrix-factorizations-parallelization-gpu.git
cd matrix-factorizations-parallelization-gpu
# If Cholesky is on a specific branch:
git checkout Cholesky
