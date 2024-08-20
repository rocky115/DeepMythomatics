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


#include "batch__qep/batch_geqp.h"
#include <iostream>

// Kernel for performing batched matrix operations
__global__ void batchGeqpKernel(float* d_A, float* d_R, int* d_pivot, int m, int n, int batchSize) {
    int batchIdx = blockIdx.x;
    int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (rowIdx < m && colIdx < n) {
        int index = batchIdx * m * n + rowIdx * n + colIdx;
        // Perform matrix operation here, e.g., copy matrix or other transformations
        d_R[index] = d_A[index];
        // Further operations like QR decomposition could be added here
    }
}

// Host function to handle memory allocation, kernel launch, and memory deallocation
void batchGeqp(float* h_A, float* h_R, int* h_pivot, int m, int n, int batchSize) {
    float* d_A;
    float* d_R;
    int* d_pivot;
    size_t sizeA = batchSize * m * n * sizeof(float);
    size_t sizeR = batchSize * m * n * sizeof(float);
    size_t sizePivot = batchSize * std::min(m, n) * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_R, sizeR);
    cudaMalloc(&d_pivot, sizePivot);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y, batchSize);

    // Launch the kernel
    batchGeqpKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_R, d_pivot, m, n, batchSize);

    // Copy results from device to host
    cudaMemcpy(h_R, d_R, sizeR, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pivot, d_pivot, sizePivot, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_R);
    cudaFree(d_pivot);
}

int main() {
    int m = 3; // Number of rows
    int n = 3; // Number of columns
    int batchSize = 1; // Number of matrices in the batch

    size_t size = batchSize * m * n;

    float* h_A = new float[size];
    float* h_R = new float[size];
    int* h_pivot = new int[batchSize * std::min(m, n)];

    // Initialize source data
    for (size_t i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    // Perform batch GEQP
    batchGeqp(h_A, h_R, h_pivot, m, n, batchSize);

    // Print a portion of the result for verification
    for (size_t i = 0; i < 9; ++i) {
        std::cout << h_R[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_R;
    delete[] h_pivot;

    return 0;
}

