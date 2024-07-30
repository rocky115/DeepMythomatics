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

void perform_batch_svd() {
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    int batch_size = 10;
    int m = 5, n = 3;
    int lda = m, ldu = m, ldv = n;

    float **d_Aarray = (float **)malloc(batch_size * sizeof(float *));
    float **d_Uarray = (float **)malloc(batch_size * sizeof(float *));
    float **d_Sarray = (float **)malloc(batch_size * sizeof(float *));
    float **d_VTarray = (float **)malloc(batch_size * sizeof(float *));

    float *h_A = (float *)malloc(m * n * batch_size * sizeof(float));
    float *h_U = (float *)malloc(m * m * batch_size * sizeof(float));
    float *h_S = (float *)malloc(n * batch_size * sizeof(float));
    float *h_VT = (float *)malloc(n * n * batch_size * sizeof(float));

    for (int i = 0; i < m * n * batch_size; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < batch_size; ++i) {
        cudaMalloc((void**)&d_Aarray[i], m * n * sizeof(float));
        cudaMalloc((void**)&d_Uarray[i], m * m * sizeof(float));
        cudaMalloc((void**)&d_Sarray[i], n * sizeof(float));
        cudaMalloc((void**)&d_VTarray[i], n * n * sizeof(float));

        cudaMemcpy(d_Aarray[i], h_A + i * m * n, m * n * sizeof(float), cudaMemcpyHostToDevice);
    }

    float **d_Aarray_dev, **d_Uarray_dev, **d_Sarray_dev, **d_VTarray_dev;
    cudaMalloc((void**)&d_Aarray_dev, batch_size * sizeof(float *));
    cudaMalloc((void**)&d_Uarray_dev, batch_size * sizeof(float *));
    cudaMalloc((void**)&d_Sarray_dev, batch_size * sizeof(float *));
    cudaMalloc((void**)&d_VTarray_dev, batch_size * sizeof(float *));

    cudaMemcpy(d_Aarray_dev, d_Aarray, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Uarray_dev, d_Uarray, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sarray_dev, d_Sarray, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_VTarray_dev, d_VTarray, batch_size * sizeof(float *), cudaMemcpyHostToDevice);

    int *info_array;
    cudaMalloc((void**)&info_array, batch_size * sizeof(int));

    int lwork;
    cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);

    float *d_work;
    cudaMalloc((void**)&d_work, lwork * batch_size * sizeof(float));

    cusolverDnSgesvdjBatched(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR,
        0, // compute left & right singular vectors
        m, n,
        d_Aarray_dev, lda,
        d_Sarray_dev,
        d_Uarray_dev, ldu,
        d_VTarray_dev, ldv,
        d_work,
        lwork,
        info_array,
        batch_size
    );

    for (int i = 0; i < batch_size; ++i) {
        cudaMemcpy(h_U + i * m * m, d_Uarray[i], m * m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_S + i * n, d_Sarray[i], n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_VT + i * n * n, d_VTarray[i], n * n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    std::cout << "Singular values of the first matrix:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << h_S[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_Aarray_dev);
    cudaFree(d_Uarray_dev);
    cudaFree(d_Sarray_dev);
    cudaFree(d_VTarray_dev);
    cudaFree(d_work);
    cudaFree(info_array);

    for (int i = 0; i < batch_size; ++i) {
        cudaFree(d_Aarray[i]);
        cudaFree(d_Uarray[i]);
        cudaFree(d_Sarray[i]);
        cudaFree(d_VTarray[i]);
    }

    free(d_Aarray);
    free(d_Uarray);
    free(d_Sarray);
    free(d_VTarray);
    free(h_A);
    free(h_U);
    free(h_S);
    free(h_VT);

    cusolverDnDestroy(cusolverH);
}

int main() {
    perform_batch_svd();
    return 0;
}
