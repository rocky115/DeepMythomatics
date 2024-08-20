/*

*Copyright (c) 2018 Radhamadhab Dalai
*Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above
copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

 */


#include "batch_rand.h"
#include <curand_kernel.h>
#include <iostream>

// Kernel for generating random matrices
__global__ void batchRandKernel(float* d_matrices, int n, int batchSize, unsigned long long seed) {
    int batchIdx = blockIdx.x;
    int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (rowIdx < n && colIdx < n) {
        int index = batchIdx * n * n + rowIdx * n + colIdx;

        // Initialize random number generator
        curandState state;
        curand_init(seed, batchIdx * n * n + rowIdx * n + colIdx, 0, &state);

        // Generate a random float value
        d_matrices[index] = curand_uniform(&state);
    }
}

// Host function to handle memory allocation, kernel launch, and memory deallocation
void batchRand(float* h_matrices, int n, int batchSize, unsigned long long seed) {
    float* d_matrices;
    size_t size = batchSize * n * n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_matrices, size);

    // Define kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(batchSize, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    batchRandKernel<<<numBlocks, threadsPerBlock>>>(d_matrices, n, batchSize, seed);

    // Copy results from device to host
    cudaMemcpy(h_matrices, d_matrices, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrices);
}

int main() {
    int n = 3; // Size of the matrix
    int batchSize = 2; // Number of matrices in the batch
    unsigned long long seed = 1234; // Random seed

    size_t size = batchSize * n * n;
    float* h_matrices = new float[size];

    // Generate random matrices
    batchRand(h_matrices, n, batchSize, seed);

    // Print a portion of the result for verification
    for (int b = 0; b < batchSize; ++b) {
        std::cout << "Matrix " << b << ":" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << h_matrices[b * n * n + i * n + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Clean up
    delete[] h_matrices;

    return 0;
}
