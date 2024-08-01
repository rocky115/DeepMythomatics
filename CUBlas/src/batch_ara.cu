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


#include <cuda_runtime.h>
#include <cstdio>

// Kernel to perform some batch operation (example)
__global__ void batchOperationKernel(float* d_data, int batchSize, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * dataSize) {
        int batchIdx = idx / dataSize;
        int dataIdx = idx % dataSize;

        // Example operation: simply increment each element
        d_data[idx] += 1.0f;
    }
}

// Function to launch the kernel
void batchOperation(float* h_data, int batchSize, int dataSize) {
    float* d_data;
    size_t size = batchSize * dataSize * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256; // Choose an appropriate block size
    int numBlocks = (batchSize * dataSize + blockSize - 1) / blockSize;

    // Launch the kernel
    batchOperationKernel<<<numBlocks, blockSize>>>(d_data, batchSize, dataSize);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Copy results back to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}

int main() {
    int batchSize = 10;
    int dataSize = 1000;
    size_t size = batchSize * dataSize * sizeof(float);

    // Allocate host memory
    float* h_data = (float*)malloc(size);
    // Initialize host data
    for (int i = 0; i < batchSize * dataSize; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Perform batch operation
    batchOperation(h_data, batchSize, dataSize);

    // Print a few results
    for (int i = 0; i < 10; ++i) {
        printf("h_data[%d] = %f\n", i, h_data[i]);
    }

    // Free host memory
    free(h_data);

    return 0;
}
