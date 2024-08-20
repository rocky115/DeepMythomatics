# CUDA Batch RAND

This repository contains CUDA code for generating random matrices in a batched manner.

## Files

### `batch_rand.h`

- **Purpose**: Contains declarations for the batch random matrix generation functions and kernels.
- **Details**:
  - **`batchRandKernel`**: Kernel function for generating random matrix values.
  - **`batchRand`**: Host function for memory management and kernel execution.

### `batch_rand.cu`

- **Purpose**: Implements the functions and kernels declared in the header file.
- **Details**:
  - **Kernel Function: `batchRandKernel`**: Generates random matrices using the CURAND library.
  - **Host Function: `batchRand`**: Manages memory allocation, data transfer, kernel execution, and deallocation.
  - **Main Function**: Demonstrates usage by generating and printing random matrices.

## Usage

1. **Compile**: Use `nvcc` to compile the `batch_rand.cu` file:
    ```bash
    nvcc batch_rand.cu -o batch_rand -lcurand
    ```

2. **Run**: Execute the compiled program:
    ```bash
    ./batch_rand
    ```

## Notes

- Ensure CUDA and CURAND are installed and properly configured on your system to compile and run the code.
- Modify matrix sizes and batch sizes in the code as needed for your specific use case.
- The CURAND library is used for generating random numbers in the kernel.

## Contact

For further questions or issues, please contact me at: [dalai115@gmail.com](mailto:dalai115@gmail.com)
