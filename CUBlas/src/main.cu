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


#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "my_cublas.h"

void perform_symv() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int n = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;
    float *d_A, *d_x, *d_y;
    float *h_A, *h_x, *h_y;

    h_A = (float*)malloc(n * n * sizeof(float));
    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(n * sizeof(float));

    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    for (int i = 0; i < n * n; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX;
        h_y[i] = 0.0f;
    }

    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    cublasSsymv(handle, CUBLAS_FILL_MODE_LOWER, n, &alpha, d_A, n, d_x, 1, &beta, d_y, 1);

    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result vector y (SYMV):\n";
    for (int i = 0; i < n; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);
    cublasDestroy(handle);
}

void perform_gemv() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int m = 1024, n = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;
    float *d_A, *d_x, *d_y;
    float *h_A, *h_x, *h_y;

    h_A = (float*)malloc(m * n * sizeof(float));
    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(m * sizeof(float));

    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, m * sizeof(float));

    for (int i = 0; i < m * n; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < m; i++) {
        h_y[i] = 0.0f;
    }

    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice);

    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1);

    cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result vector y (GEMV):\n";
    for (int i = 0; i < m; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);
    cublasDestroy(handle);
}

void perform_hemv() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int n = 1024;
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta = make_cuComplex(0.0f, 0.0f);
    cuComplex *d_A, *d_x, *d_y;
    cuComplex *h_A, *h_x, *h_y;

    h_A = (cuComplex*)malloc(n * n * sizeof(cuComplex));
    h_x = (cuComplex*)malloc(n * sizeof(cuComplex));
    h_y = (cuComplex*)malloc(n * sizeof(cuComplex));

    cudaMalloc((void**)&d_A, n * n * sizeof(cuComplex));
    cudaMalloc((void**)&d_x, n * sizeof(cuComplex));
    cudaMalloc((void**)&d_y, n * sizeof(cuComplex));

    for (int i = 0; i < n * n; i++) {
        h_A[i] = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
    }
    for (int i = 0; i < n; i++) {
        h_x[i] = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
        h_y[i] = make_cuComplex(0.0f, 0.0f);
    }

    cudaMemcpy(d_A, h_A, n * n * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cublasChemv(handle, CUBLAS_FILL_MODE_LOWER, n, &alpha, d_A, n, d_x, 1, &beta, d_y, 1);

    cudaMemcpy(h_y, d_y, n * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    std::cout << "Result vector y (HEMV):\n";
    for (int i = 0; i < n; i++) {
        std::cout << h_y[i].x << " + " << h_y[i].y << "i ";
    }
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);
    cublasDestroy(handle);
}

int main() {
    perform_symv();
    perform_gemv();
    perform_hemv();
    return 0;
}
