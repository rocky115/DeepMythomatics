## Om Sree Ganeshay Namoh
## OM Sree Sree Hanumate Namoh
## Om Sree Sree Bhadraiey Namoh

Creating a CUDA project to use legacy Level-2 BLAS functions (SYMV, GEMV, HEMV) involves several steps, 
including setting up your development environment, writing the CUDA code, and using cuBLAS for these operations.


# ara_util.cuh - README

## Explanation

### Helper Functions
- **Utility functions handle pointer arithmetic and kernel configuration.**
  - **ARASampleSetter**: Sets sample values based on comparisons with `r`, enabling operations on both host and device.
  - **ARAOffsetPointer**: Computes offset pointers for batched matrices, accounting for offsets and leading dimensions.
  - **TRSM_Offset_Dims**: Sets dimensions for triangular solve operations, adjusting based on column count and offsets.
  - **SampleIsZeroPredicate**: Predicate for checking if values are zero.
  - **copyGPUArray**: Template function for copying arrays between device memory locations.
  - **ara_trsm_set_offset_dims**: Configures dimensions for triangular solve operations.
  - **ara_offset_pointers**: Computes and stores pointers with specified offsets.
  - **kblas_ara_batch_set_samples**: Sets sample values for a batch and checks if all values are zero.

### Kernels
- **Perform specific matrix operations, including Cholesky decomposition, counting, and triangular solve.**
  - **warp_max**: Computes the maximum value in a warp using shuffle operations.
  - **warp_sum**: Computes the sum of values in a warp using shuffle operations.
  - **ara_fused_potrf_kernel**: Performs batched Cholesky factorization with shared memory for hierarchical computation.
  - **ara_svec_count_kernel**: Counts small vectors, updates ranks, and small vector counts based on diagonal entries and tolerance.
  - **ara_16x16_trsm_kernel**: Solves triangular systems using a 16x16 block size, updating matrix B in place.

### Drivers
- **Launch kernels with appropriate configurations and manage workspace.**
  - **kblas_ara_fused_potrf_batch_template**: Calls the appropriate `ara_fused_potrf_kernel` based on block size (16 or 32).
  - **kblas_ara_svec_count_batch_template**: Calls the appropriate `ara_svec_count_kernel` based on block size (16 or 32).
  - **kblas_ara_16x16_trsm**: Manages workspace allocation and kernel launches for the 16x16 triangular solve kernel.
  - **ara_trsm_batch_wsquery**: Determines workspace requirements for triangular solve batch operations.
  - **kblas_ara_trsm_batch_template**: Handles triangular solve batch operations with workspace allocation and kernel launches.

### Mixed Precision SYRK
- **ara_mp_syrk_batch_kernel**: Computes matrix product G = A^T * A using mixed precision (single precision for A, double precision for G).

### Notes
- **Workspace Management**: Utilizes `KBlasWorkspace` for managing memory requirements.
- **Error Checking**: Implements functions like `check_error` for handling CUDA errors.

If you need further details or have specific questions about parts of this code, let me know!

Email: dalai115@gmail.com

###########################################################################################################

# CUDA Batch Operations

This repository contains CUDA code for performing batch operations on matrix data. The main file is `batch_ara.cu`, which includes a kernel for processing batched data, as well as host functions for memory management and kernel launching.

## `batch_ara.cu`

### Kernel Function: `batchOperationKernel`

- **Purpose**: Performs operations on the batched data.
- **Details**: Uses indexing to handle batched elements and performs operations such as incrementing each element.

### Host Function: `batchOperation`

- **Purpose**: Manages memory and kernel execution.
- **Details**:
  - **Allocates device memory**: Allocates space on the GPU for the batched data.
  - **Copies data to the device**: Transfers data from host to device memory.
  - **Launches the kernel**: Executes the `batchOperationKernel` with appropriate block and grid sizes.
  - **Copies results back to the host**: Transfers the results from device to host memory.
  - **Frees device memory**: Releases the allocated GPU memory.

### Main Function

- **Purpose**: Initializes data and calls the batch operation function.
- **Details**:
  - **Allocates and initializes host data**: Sets up and initializes the data on the host.
  - **Calls the batch operation function**: Invokes `batchOperation` to process the data.
  - **Prints results**: Displays a portion of the processed data for verification.
  - **Frees host memory**: Releases the allocated memory on the host.

## Usage

1. **Compile**: Use `nvcc` to compile the `batch_ara.cu` file:
    ```bash
    nvcc batch_ara.cu -o batch_ara
    ```

2. **Run**: Execute the compiled program:
    ```bash
    ./batch_ara
    ```

## Notes

- Make sure you have CUDA installed and properly configured on your system to compile and run the code.
- Adjust block sizes and other parameters in the kernel and host function as needed for your specific use case.

## Contact

For further questions or issues, please contact me at: [dalai115@gmail.com](mailto:dalai115@gmail.com)
