#include "cublasConvert.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "create-allocate.h"
#include "cudaError.h"

__global__ void floatToBfloat16(float *input, size_t num_elements,
                                __nv_bfloat16 *output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

__global__ void floatToFp16(float *input, size_t num_elements, __half *output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __float2half(input[idx]);
  }
}

void copyAndConvert(cublasDataType_t precision, void *hostA, void *devA, int x, int y, int batchsz)
{

  int hostsz = typeCallHost<sizeofCUDT>(precision);
  int devsz = typeCallDev<sizeofCUDT>(precision);
  if (precision == CUDA_C_16F || precision == CUDA_R_16F)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkCuda(cudaMemcpy(tmpA, hostA, batchsz * x * y * devsz,
                         cudaMemcpyHostToDevice));
    int num_elements = batchsz * x * y;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    floatToFp16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA);
    //__half *output_cpu = (__half *)malloc(num_elements * sizeof(__half));
    // checkCuda(cudaMemcpy(output_cpu, devA, num_elements * sizeof(__half), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < 10; i++)
    //{
    //  printf("Input value: %f, Output value: %f\n", ((float *)hostA)[i], __half2float(output_cpu[i]));
    //}
    cudaFree(tmpA);
  }
  else
  {
    // std::cerr << "Error: Precision not supported" << std::endl;
    //  Simply copy, no convert required
    checkCuda(cudaMemcpy(devA, hostA, batchsz * x * y * devsz,
                         cudaMemcpyHostToDevice));
  }
}

void convertScalar(cublasDataType_t precision, void *scalar)
{

  if (precision == CUDA_R_16F)
  {
    // For whatever reason,
    float scalarVal = *static_cast<float *>(scalar);
    free(scalar);
    scalar = (void *)malloc(sizeof(__half));
    *static_cast<__half *>(scalar) = __float2half(scalarVal);
  }
  else if (precision == CUDA_C_16F)
  {
    // Implement me...
  }
}