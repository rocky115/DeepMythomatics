# CUDA Batch PSTRF

This repository contains CUDA code for performing batched pivoted Cholesky factorization.

## Files

### `batch_pstrf.h`

- **Purpose**: Contains declarations for the batch pivoted Cholesky factorization functions and kernels.
- **Details**:
  - **`batchPstrfKernel`**: Kernel function for performing matrix factorization in a batch.
  - **`batchPstrf`**: Host function for memory management and kernel execution.

### `batch_pstrf.cu`

- **Purpose**: Implements the functions and kernels declared in the header file.
- **Details**:
  - **Kernel Function: `batchPstrfKernel`**: Performs batched matrix operations, such as pivoted Cholesky factorization.
  - **Host Function: `batchPstrf`**: Manages memory allocation, data transfer, kernel execution, and deallocation.
  - **Main Function**: Demonstrates usage by initializing matrices, performing the operation, and printing results.

## Usage

1. **Compile**: Use `nvcc` to compile the `batch_pstrf.cu` file:
    ```bash
    nvcc batch_pstrf.cu -o batch_pstrf
    ```

2. **Run**: Execute the compiled program:
    ```bash
    ./batch_pstrf
    ```

## Notes

- Ensure CUDA is installed and properly configured on your system to compile and run the code.
- Modify matrix sizes and batch sizes in the code as needed for your specific use case.

## Contact

For further questions or issues, please contact me at: [dalai115@gmail.com](mailto:dalai115@gmail.com)
