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
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "my_cublas.h"

void perform_batch_qr() {
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    int batch_size = 10;
    int m = 5, n = 3;
    int lda = m;

    float **d_Aarray = (float **)malloc(batch_size * sizeof(float *));
    float **d_Tauarray = (float **)malloc(batch_size * sizeof(float *));

    float *h_A = (float *)malloc(m * n * batch_size * sizeof(float));
    float *h_Tau = (float *)malloc(n * batch_size * sizeof(float));

    for (int i = 0; i < m * n * batch_size; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < batch_size; ++i) {
        cudaMalloc((void**)&d_Aarray[i], m * n * sizeof(float));
        cudaMalloc((void**)&d_Tauarray[i], n * sizeof(float));

        cudaMemcpy(d_Aarray[i], h_A + i * m * n, m * n * sizeof(float), cudaMemcpyHostToDevice);
    }

    float **d_Aarray_dev, **d_Tauarray_dev;
    cudaMalloc((void**)&d_Aarray_dev, batch_size * sizeof(float *));
    cudaMalloc((void**)&d_Tauarray_dev, batch_size * sizeof(float *));

    cudaMemcpy(d_Aarray_dev, d_Aarray, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Tauarray_dev, d_Tauarray, batch_size * sizeof(float *), cudaMemcpyHostToDevice);

    int *info_array;
    cudaMalloc((void**)&info_array, batch_size * sizeof(int));

    int lwork;
    cusolverDnSgeqrf_bufferSize(cusolverH, m, n, d_Aarray[0], lda, &lwork);

    float *d_work;
    cudaMalloc((void**)&d_work, lwork * batch_size * sizeof(float));

    cusolverDnSgeqrfBatched(
        cusolverH,
        m, n,
        d_Aarray_dev, lda,
        d_Tauarray_dev,
        info_array,
        batch_size
    );

    for (int i = 0; i < batch_size; ++i) {
        cudaMemcpy(h_Tau + i * n, d_Tauarray[i], n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    std::cout << "Tau values of the first matrix:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << h_Tau[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_Aarray_dev);
    cudaFree(d_Tauarray_dev);
    cudaFree(d_work);
    cudaFree(info_array);

    for (int i = 0; i < batch_size; ++i) {
        cudaFree(d_Aarray[i]);
        cudaFree(d_Tauarray[i]);
    }

    free(d_Aarray);
    free(d_Tauarray);
    free(h_A);
    free(h_Tau);

    cusolverDnDestroy(cusolverH);
}
