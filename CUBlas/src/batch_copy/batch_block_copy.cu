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
#include "batch_copy/batch_block_copy.h"
#include <iostream>

// Kernel to perform block-wise data copy
__global__ void batchBlockCopyKernel(float* dst, const float* src, size_t batchSize, size_t blockSize) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * blockSize) {
        size_t batchIdx = idx / blockSize;
        size_t offset = (batchIdx * blockSize) + (idx % blockSize);
        dst[offset] = src[offset];
    }
}

// Host function to handle memory allocation, kernel launch, and memory deallocation
void batchBlockCopy(float* h_dst, const float* h_src, size_t batchSize, size_t blockSize) {
    float* d_src;
    float* d_dst;
    size_t size = batchSize * blockSize * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);

    // Copy data from host to device
    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

    // Launch the kernel with appropriate configuration
    size_t numThreadsPerBlock = 256;
    size_t numBlocks = (batchSize * blockSize + numThreadsPerBlock - 1) / numThreadsPerBlock;
    batchBlockCopyKernel<<<numBlocks, numThreadsPerBlock>>>(d_dst, d_src, batchSize, blockSize);

    // Copy results from device to host
    cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    size_t batchSize = 10;
    size_t blockSize = 1024;
    size_t size = batchSize * blockSize;

    float* h_src = new float[size];
    float* h_dst = new float[size];

    // Initialize source data
    for (size_t i = 0; i < size; ++i) {
        h_src[i] = static_cast<float>(i);
    }

    // Perform block-wise copy
    batchBlockCopy(h_dst, h_src, batchSize, blockSize);

    // Print a portion of the result for verification
    for (size_t i = 0; i < 10; ++i) {
        std::cout << h_dst[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_src;
    delete[] h_dst;

    return 0;
}
