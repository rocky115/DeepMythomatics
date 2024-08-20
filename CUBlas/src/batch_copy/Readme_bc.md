# CUDA Batch Block Copy

This repository contains CUDA code for performing block-wise data copy operations on batched data. The main file is `batch_block_copy.cu`, which demonstrates how to efficiently copy data between host and device memory, or between different areas of device memory.

## `batch_block_copy.cu`

### Kernel Function: `batchBlockCopyKernel`

- **Purpose**: Copies data from a source array to a destination array in a block-wise manner.
- **Details**: 
  - Uses indexing to handle batched elements.
  - Each thread copies a portion of the data based on the block size.

### Host Function: `batchBlockCopy`

- **Purpose**: Manages memory allocation, data transfer, and kernel execution.
- **Details**:
  - **Allocates device memory**: Sets up space on the GPU for source and destination data.
  - **Copies data to the device**: Transfers data from host to device memory.
  - **Launches the kernel**: Executes the `batchBlockCopyKernel` with the appropriate block and grid sizes.
  - **Copies results back to the host**: Transfers the copied data from device to host memory.
  - **Frees device memory**: Releases GPU memory used for the copy operation.

### Main Function

- **Purpose**: Initializes data, performs the batch block copy, and verifies the results.
- **Details**:
  - **Allocates and initializes host data**: Sets up the source data on the host.
  - **Calls the batch block copy function**: Invokes `batchBlockCopy` to perform the data copy.
  - **Prints results**: Displays a portion of the copied data for verification.
  - **Frees host memory**: Releases allocated memory on the host.

## Usage

1. **Compile**: Use `nvcc` to compile the `batch_block_copy.cu` file:
    ```bash
    nvcc batch_block_copy.cu -o batch_block_copy
    ```

2. **Run**: Execute the compiled program:
    ```bash
    ./batch_block_copy
    ```

## Notes

- Ensure CUDA is installed and properly configured on your system to compile and run the code.
- Adjust block sizes and other parameters in the kernel and host function as needed for your specific use case.

## Contact

For further questions or issues, please contact me at: [dalai115@gmail.com](mailto:dalai115@gmail.com)
