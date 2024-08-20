
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


#include "batch_pstrf.h"
#include <iostream>

// Kernel for performing batched pivoted Cholesky factorization
__global__ void batchPstrfKernel(float* d_A, float* d_L, int* d_pivot, int* d_info, int n, int batchSize) {
    int batchIdx = blockIdx.x;
    int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (rowIdx < n && colIdx < n) {
        int index = batchIdx * n * n + rowIdx * n + colIdx;

        // Example: Copy matrix A to L (Placeholder for actual factorization logic)
        if (rowIdx == colIdx) {
            d_L[index] = d_A[index]; // This would be replaced by actual Cholesky logic
        } else {
            d_L[index] = 0.0f;
        }

        // Set pivot and info values (Placeholder logic)
        if (rowIdx == 0 && colIdx == 0) {
            d_pivot[batchIdx] = 0; // Example pivot
            d_info[batchIdx] = 0;  // Example info status
        }
    }
}

// Host function to handle memory allocation, kernel launch, and memory deallocation
void batchPstrf(float* h_A, float* h_L, int* h_pivot, int* h_info, int n, int batchSize) {
    float* d_A;
    float* d_L;
    int* d_pivot;
    int* d_info;
    size_t sizeA = batchSize * n * n * sizeof(float);
    size_t sizeL = batchSize * n * n * sizeof(float);
    size_t sizePivot = batchSize * sizeof(int);
    size_t sizeInfo = batchSize * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_L, sizeL);
    cudaMalloc(&d_pivot, sizePivot);
    cudaMalloc(&d_info, sizeInfo);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   batchSize);

    // Launch the kernel
    batchPstrfKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_L, d_pivot, d_info, n, batchSize);

    // Copy results from device to host
    cudaMemcpy(h_L, d_L, sizeL, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pivot, d_pivot, sizePivot, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_info, d_info, sizeInfo, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_pivot);
    cudaFree(d_info);
}

int main() {
    int n = 3; // Size of the matrix
    int batchSize = 1; // Number of matrices in the batch

    size_t size = batchSize * n * n;

    float* h_A = new float[size];
    float* h_L = new float[size];
    int* h_pivot = new int[batchSize];
    int* h_info = new int[batchSize];

    // Initialize source data (example data)
    for (size_t i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(i + 1);
    }

    // Perform batch pivoted Cholesky factorization
    batchPstrf(h_A, h_L, h_pivot, h_info, n, batchSize);

    // Print a portion of the result for verification
    for (size_t i = 0; i < 9; ++i) {
        std::cout << h_L[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_L;
    delete[] h_pivot;
    delete[] h_info;

    return 0;
}
