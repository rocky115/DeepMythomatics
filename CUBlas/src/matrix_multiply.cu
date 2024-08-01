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

#include "matrix_helpers.cuh"
#include <iostream>

__global__ void matrix_multiply_kernel(const float* d_A, const float* d_B, float* d_C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float value = 0.0f;
        for (int e = 0; e < n; ++e) {
            value += d_A[row * n + e] * d_B[e * k + col];
        }
        d_C[row * k + col] = value;
    }
}

__global__ void matrix_transpose_kernel(const float* d_matrix, float* d_transposed, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        d_transposed[col * m + row] = d_matrix[row * n + col];
    }
}

void matrix_multiply(const float* h_A, const float* h_B, float* h_C, int m, int n, int k) {
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * k * sizeof(float));
    cudaMalloc((void**)&d_C, m * k * sizeof(float));

    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(h_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrix_transpose(const float* h_matrix, float* h_transposed, int m, int n) {
    float *d_matrix, *d_transposed;

    cudaMalloc((void**)&d_matrix, m * n * sizeof(float));
    cudaMalloc((void**)&d_transposed, m * n * sizeof(float));

    cudaMemcpy(d_matrix, h_matrix, m * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_transposed, m, n);

    cudaMemcpy(h_transposed, d_transposed, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_transposed);
}
