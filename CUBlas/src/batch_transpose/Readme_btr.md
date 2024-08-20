# CUDA Batch Transpose

This repository contains CUDA code for transposing matrices in a batched manner.

## Files

### `batch_transpose.h`

- **Purpose**: Contains declarations for the batch matrix transposition functions and kernels.
- **Details**:
  - **`batchTransposeKernel`**: Kernel function for transposing matrices.
  - **`batchTranspose`**: Host function for memory management and kernel execution.

### `batch_transpose.cu`

- **Purpose**: Implements the functions and kernels declared in the header file.
- **Details**:
  - **Kernel Function: `batchTransposeKernel`**: Transposes matrices using CUDA kernel.
  - **Host Function: `batchTranspose`**: Manages memory allocation, data transfer, kernel execution, and deallocation.
  - **Main Function**: Demonstrates usage by generating and printing transposed matrices.

## Usage

1. **Compile**: Use `nvcc` to compile the `batch_transpose.cu` file:
    ```bash
    nvcc batch_transpose.cu -o batch_transpose
    ```

2. **Run**: Execute the compiled program:
    ```bash
    ./batch_transpose
    ```

## Notes

- Ensure CUDA is installed and properly configured on your system to compile and run the code.
- Adjust matrix sizes and batch sizes as needed for your specific use case.

## Contact

For further questions or issues, please contact me at: [dalai115@gmail.com](mailto:dalai115@gmail.com)
