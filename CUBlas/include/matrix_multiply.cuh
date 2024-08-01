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

#ifndef CUBLAS_MATRIX_MULTIPLY_HPP
#define CUBLAS_MATRIX_MULTIPLY_HPP

#include <cuda_runtime.h>

// Kernel function for matrix multiplication
__global__ void matrix_multiply_kernel(const float* d_A, const float* d_B, float* d_C, int m, int n, int k);

// Kernel function for matrix transpose
__global__ void matrix_transpose_kernel(const float* d_matrix, float* d_transposed, int m, int n);

void matrix_multiply(const float* h_A, const float* h_B, float* h_C, int m, int n, int k);
void matrix_transpose(const float* h_matrix, float* h_transposed, int m, int n);

#endif //CUBLAS_MATRIX_MULTIPLY_HPP
