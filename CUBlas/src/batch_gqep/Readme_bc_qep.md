# CUDA Batch GEQP

This repository contains CUDA code for performing batched matrix operations, specifically focusing on generalized QR decomposition (GEQP) and related operations.

## Files

### `batch_geqp.h`

- **Purpose**: Contains declarations for the batch GEQP functions and kernels.
- **Details**:
  - **`batchGeqpKernel`**: Kernel function for performing matrix operations in a batch.
  - **`batchGeqp`**: Host function for memory management and kernel execution.

### `batch_geqp.cu`

- **Purpose**: Implements the functions and kernels declared in the header file.
- **Details**:
  - **Kernel Function: `batchGeqpKernel`**: Performs batched matrix operations, such as copying matrices.
  - **Host Function: `batchGeqp`**: Manages memory allocation, data transfer, kernel execution, and deallocation.
  - **Main Function**: Demonstrates usage by initializing matrices, performing the operation, and printing results.

## Usage

1. **Compile**: Use `nvcc` to compile the `batch_geqp.cu` file:
    ```bash
    nvcc batch_geqp.cu -o batch_geqp
    ```

2. **Run**: Execute the compiled program:
    ```bash
    ./batch_geqp
    ```

## Notes

- Ensure CUDA is installed and properly configured on your system to compile and run the code.
- Modify matrix sizes and batch sizes in the code as needed for your specific use case.

## Contact

For further questions or issues, please contact me at: [dalai115@gmail.com](mailto:dalai115@gmail.com)
