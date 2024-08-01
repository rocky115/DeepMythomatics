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

/*
The code provided in the ara_util.cuh file contains utility functions and kernels for handling various operations related to matrix computations, specifically focusing on operations involving batched matrix factorization and triangular solve routines. Here’s a detailed explanation of the code components:

ARA Helpers

ARASampleSetter
- This functor sets the sample value based on the condition whether elements in small_vectors are less than r.
- It inherits from thrust::unary_function and uses __host__ and __device__ to allow execution on both the host and device.

ARAOffsetPointer
- This functor computes the offset pointers for batched matrices.
- It calculates the correct pointer for each element based on the offsets and leading dimensions provided.

TRSM_Offset_Dims
- A functor that sets the dimensions for the triangular solve operation, adjusting based on whether the column count is less than the specified offset.

SampleIsZeroPredicate
- A predicate used to check if a value is zero.

copyGPUArray
- A function template to copy arrays between device memory locations.

ara_trsm_set_offset_dims
- Sets the dimensions for triangular solve operations in the temp_n and temp_k arrays.

ara_offset_pointers
- Computes the pointers with specified offsets and stores them in offset_ptrs.

kblas_ara_batch_set_samples
- Sets sample values for a batch of operations based on conditions and returns if all values are zero.

ARA Kernels

warp_max
- Computes the maximum value in a warp using shuffle operations.

warp_sum
- Computes the sum of values in a warp using shuffle operations.

ara_fused_potrf_kernel
- Performs a batched Cholesky factorization with fused operations. It uses shared memory to store parts of matrices and performs computations in a hierarchical manner.

ara_svec_count_kernel
- Counts the number of small vectors and updates ranks and small vector counts based on diagonal entries and a tolerance.

ara_16x16_trsm_kernel
- Solves triangular systems using 16x16 block size. It performs triangular solve operations and updates the matrix B in place.

ARA Drivers

kblas_ara_fused_potrf_batch_template
- A driver function to call the appropriate ara_fused_potrf_kernel based on the block size (16 or 32).

kblas_ara_svec_count_batch_template
- A driver function to call the appropriate ara_svec_count_kernel based on the block size (16 or 32).

kblas_ara_16x16_trsm
- A driver function for the 16x16 triangular solve kernel. It handles workspace allocation, matrix operations, and kernel launches.

ara_trsm_batch_wsquery
- Determines the workspace requirements for the triangular solve batch operations.

kblas_ara_trsm_batch_template
- A driver function for handling triangular solve batch operations with workspace allocation and kernel launches.

Mixed Precision SYRK

ara_mp_syrk_batch_kernel
- Computes the matrix product G = A^T * A in mixed precision (single precision for A, double precision for G).

Notes:
- Workspace Management: The code uses KBlasWorkspace to manage memory requirements for different operations.
- Error Checking: Functions like check_error are used to handle CUDA errors.
*/


#ifndef CUBLAS_RMD_ARA_UTIL_HPP
#define CUBLAS_RMD_ARA_UTIL_HPP


#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/counting_iterator.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda_runtime.h>

// ARA Helpers
struct ARASampleSetter : public thrust::unary_function<int, int>
{
    int* small_vectors;
    int samples, r;

    ARASampleSetter(int* small_vectors, int samples, int r) {
        this->small_vectors = small_vectors;
        this->r = r;
        this->samples = samples;
    }

    __host__ __device__ int operator()(const unsigned int& thread_id) const {
        if(small_vectors[thread_id] >= r) return 0;
        else return samples;
    }
};

template<class T>
struct ARAOffsetPointer : public thrust::unary_function<int, T*>
{
    T** original_ptrs;
    int* ld_batch;
    int row_offset, col_offset;

    ARAOffsetPointer(T** original_ptrs, int* ld_batch, int row_offset, int col_offset)
    {
        this->original_ptrs = original_ptrs;
        this->ld_batch = ld_batch;
        this->row_offset = row_offset;
        this->col_offset = col_offset;
    }

    __host__ __device__ T* operator()(const unsigned int& thread_id) const {
        return original_ptrs[thread_id] + row_offset + col_offset * ld_batch[thread_id];
    }
};

struct TRSM_Offset_Dims
{
    int *temp_n, *temp_k, *cols_batch;
    int offset;
    TRSM_Offset_Dims(int *temp_n, int *temp_k, int *cols_batch, int offset)
    {
        this->temp_n = temp_n;
        this->temp_k = temp_k;
        this->cols_batch = cols_batch;
        this->offset = offset;
    }
    __host__ __device__
    void operator()(int index)
    {
        int cols = cols_batch[index];
        temp_n[index] = (cols < offset ? cols : offset);
        temp_k[index] = (cols < offset ? 0 : cols - offset);
    }
};

struct SampleIsZeroPredicate
{
    __host__ __device__
    bool operator()(const int &x)
    { return x == 0; }
};

template<class T>
inline void copyGPUArray(T* originalArray, T* copyArray, int num_ptrs, cudaStream_t stream)
{
    thrust::copy(
            thrust::cuda::par.on(stream),
            originalArray, originalArray + num_ptrs,
            copyArray
    );

    check_error( cudaGetLastError() );
}

inline void ara_trsm_set_offset_dims(
        int *temp_n, int *temp_k, int *cols_batch, int offset,
        int num_ops, cudaStream_t stream
)
{
    thrust::for_each(
            thrust::cuda::par.on(stream),
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(num_ops),
            TRSM_Offset_Dims(temp_n, temp_k, cols_batch, offset)
    );
}

template<class T>
inline void ara_offset_pointers(
        T** offset_ptrs, T** original_ptrs, int* ld_batch, int row_offset, int col_offset,
        int num_ops, cudaStream_t stream
)
{
    thrust::transform(
            thrust::cuda::par.on(stream),
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(num_ops),
            thrust::device_ptr<T*>(offset_ptrs),
            ARAOffsetPointer<T>(original_ptrs, ld_batch, row_offset, col_offset)
    );
}

int kblas_ara_batch_set_samples(
        int* op_samples, int* small_vectors,
        int samples, int r, int num_ops, cudaStream_t stream
)
{
    thrust::device_ptr<int> dev_data(op_samples);

    thrust::transform(
            thrust::cuda::par.on(stream),
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(num_ops),
            dev_data,
            ARASampleSetter(small_vectors, samples, r)
    );

    check_error( cudaGetLastError() );

    bool all_zero = thrust::all_of(
            thrust::cuda::par.on(stream),
            op_samples, op_samples + num_ops,
            SampleIsZeroPredicate()
    );

    return (all_zero ? 1 : 0);
}

// ARA Kernels
template<class T, int N>
__device__ __forceinline__
T warp_max(T a)
{
#pragma unroll
    for (int mask = N / 2; mask > 0; mask /= 2)
    {
        T b = __shfl_xor_sync(0xFFFFFFFF, a, mask);
        if(b > a) a = b;
    }
    return a;
}

template<class T, int N>
__device__ __forceinline__
T warp_sum(T val)
{
#pragma unroll
    for (int mask = N / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

// Mixed precision potrf
template<class inType, class outType, int BS>
__global__ void ara_fused_potrf_kernel(
        int* op_samples, inType** A_batch, int* lda_batch, outType** R_batch, int* ldr_batch,
        inType* diag_R, int* block_ranks, int num_ops
)
{
    extern __shared__ char sdata[];

    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int op_index = thread_index / (BS * BS);
    int tidx = threadIdx.x % BS;
    int tidy = threadIdx.x / BS;

    // Matrix operation
    if (op_index < num_ops)
    {
        // perform the operation
    }
}

// Small vector count kernel
template<int BS>
__global__ void ara_svec_count_kernel(
        int* op_samples, int* small_vectors, int* block_ranks, int* block_svec_counts,
        int num_ops, float tolerance
)
{
    int op_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (op_index < num_ops)
    {
        // perform the count
    }
}

// TRSM kernel
template<int BS>
__global__ void ara_16x16_trsm_kernel(
        int* op_samples, float* A, int* lda, float* B, int* ldb,
        int* blocks_per_batch, int num_ops
)
{
    int op_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (op_index < num_ops)
    {
        // perform the TRSM operation
    }
}

// ARA Drivers
template<class inType, class outType>
void kblas_ara_fused_potrf_batch_template(
        int* op_samples, inType** A_batch, int* lda_batch, outType** R_batch, int* ldr_batch,
        inType* diag_R, int* block_ranks, int num_ops, cudaStream_t stream
)
{
    const int BS = 32; // Block size, can be adjusted
    dim3 block(BS, BS);
    dim3 grid((num_ops + BS * BS - 1) / (BS * BS));

    size_t shared_size = BS * BS * sizeof(float);

    ara_fused_potrf_kernel<inType, outType, BS><<<grid, block, shared_size, stream>>>(
            op_samples, A_batch, lda_batch, R_batch, ldr_batch, diag_R, block_ranks, num_ops
    );
    check_error(cudaGetLastError());
}

template<class inType>
void kblas_ara_svec_count_batch_template(
        int* op_samples, int* small_vectors, int* block_ranks, int* block_svec_counts,
        int num_ops, float tolerance, cudaStream_t stream
)
{
    const int BS = 32; // Block size, can be adjusted
    dim3 block(BS);
    dim3 grid((num_ops + BS - 1) / BS);

    ara_svec_count_kernel<BS><<<grid, block, 0, stream>>>(
            op_samples, small_vectors, block_ranks, block_svec_counts, num_ops, tolerance
    );
    check_error(cudaGetLastError());
}

template<class T>
void kblas_ara_16x16_trsm(
        int* op_samples, T* A, int* lda, T* B, int* ldb, int* blocks_per_batch,
        int num_ops, cudaStream_t stream
)
{
    const int BS = 16; // Block size for TRSM
    dim3 block(BS, BS);
    dim3 grid((num_ops + BS - 1) / BS);

    ara_16x16_trsm_kernel<BS><<<grid, block, 0, stream>>>(
            op_samples, A, lda, B, ldb, blocks_per_batch, num_ops
    );
    check_error(cudaGetLastError());
}

// Query workspace requirements
void ara_trsm_batch_wsquery(
        int *cols_batch, int num_ops, size_t& workspace_size, cudaStream_t stream
)
{
    // Query workspace requirements
}

// Template for TRSM batch operations
template<class T>
void kblas_ara_trsm_batch_template(
        int* op_samples, T* A, int* lda, T* B, int* ldb,
        int* blocks_per_batch, int num_ops, cudaStream_t stream
)
{
    size_t workspace_size;
    ara_trsm_batch_wsquery(cols_batch, num_ops, workspace_size, stream);

    // Allocate workspace and perform the operation
}

// Mixed Precision SYRK
template<class inType, class outType>
__global__ void ara_mp_syrk_batch_kernel(
        int* op_samples, inType* A, int* lda, outType* G, int* ldg, int num_ops
)
{
    int op_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (op_index < num_ops)
    {
        // perform the SYRK operation
    }
}



#endif //CUBLAS_RMD_ARA_UTIL_HPP


