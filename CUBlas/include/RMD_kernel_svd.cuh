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


/*Creating a header file kernel_svd.cuh involves defining the CUDA kernel functions used for Singular Value Decomposition
 * (SVD). Typically, in a CUDA project, you might want to implement some of the matrix operations required for SVD in
 * custom kernels if the standard cuBLAS functions are not enough for your needs.
 * Here’s an example of what the kernel_svd.cuh header file might look like. This file declares the CUDA kernels and
 * related utility functions for performing matrix operations relevant to SVD.


 */

#ifndef CUBLAS_RMD_KERNEL_SVD_H
#define CUBLAS_RMD_KERNEL_SVD_H

#include <cuda_runtime.h>

__global__ void svd_decomposition_kernel(float* d_matrix, float* d_U, float* d_S, float* d_VT, int m, int n);

__global__ void matrix_multiply_kernel(const float* d_A, const float* d_B, float* d_C, int m, int n, int k);

void perform_svd(float* h_matrix, float* h_U, float* h_S, float* h_VT, int m, int n);
#endif //CUBLAS_RMD_KERNEL_SVD_H


/*
 * Explanation
Header Guards: Prevent multiple inclusions of the header file.

Kernel Declarations:

svd_decomposition_kernel: A placeholder kernel function for performing SVD operations.
matrix_multiply_kernel: A kernel function for matrix multiplication, which is often used in SVD computations.
Function Declarations:

perform_svd: A utility function that will manage the SVD operation by invoking CUDA kernels and handling memory transfers.

 * */