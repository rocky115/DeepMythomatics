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

#include "batch_transpose.h"
#include <iostream>

// Kernel for transposing matrices in a batch
__global__ void batchTransposeKernel(float* d_matrices, float* d_transposed_matrices, int n, int batchSize) {
    int batchIdx = blockIdx.z;
    int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (rowIdx < n && colIdx < n) {
        int srcIndex = batchIdx * n * n + rowIdx * n + colIdx;
        int dstIndex = batchIdx * n * n + colIdx * n + rowIdx;

        d_transposed_matrices[dstIndex] = d_matrices[srcIndex];
    }
}

// Host function to handle memory allocation, kernel launch, and memory deallocation
void batchTranspose(float* h_matrices, float* h_transposed_matrices, int n, int batchSize) {
    float* d_matrices;
    float* d_transposed_matrices;
    size_t size = batchSize * n * n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_matrices, size);
    cudaMalloc(&d_transposed_matrices, size);

    // Copy data from host to device
    cudaMemcpy(d_matrices, h_matrices, size, cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   batchSize);

    // Launch the kernel
    batchTransposeKernel<<<numBlocks, threadsPerBlock>>>(d_matrices, d_transposed_matrices, n, batchSize);

    // Copy results from device to host
    cudaMemcpy(h_transposed_matrices, d_transposed_matrices, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrices);
    cudaFree(d_transposed_matrices);
}

int main() {
    int n = 3; // Size of the matrix
    int batchSize = 2; // Number of matrices in the batch

    size_t size = batchSize * n * n;
    float* h_matrices = new float[size];
    float* h_transposed_matrices = new float[size];

    // Initialize matrices with some values (for example purposes)
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                h_matrices[b * n * n + i * n + j] = static_cast<float>(i * n + j);
            }
        }
    }

    // Perform matrix transposition
    batchTranspose(h_matrices, h_transposed_matrices, n, batchSize);

    // Print original and transposed matrices for verification
    for (int b = 0; b < batchSize; ++b) {
        std::cout << "Original Matrix " << b << ":" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << h_matrices[b * n * n + i * n + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Transposed Matrix " << b << ":" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << h_transposed_matrices[b * n * n + i * n + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Clean up
    delete[] h_matrices;
    delete[] h_transposed_matrices;

    return 0;
}
