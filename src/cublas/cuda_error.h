#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include "backend_error.h"

const char *cublas_get_error_string(cublasStatus_t status);

static inline cudaError_t check_cuda(cudaError_t result) {
  return check_status<cudaError_t, cudaSuccess>(result, "CUDA Runtime",
                                                cudaGetErrorString);
}

static inline cublasStatus_t check_cublas(cublasStatus_t result) {
  return check_status<cublasStatus_t, CUBLAS_STATUS_SUCCESS>(
      result, "cuBLAS Runtime", cublas_get_error_string);
}

static inline nvmlReturn_t check_nvml(nvmlReturn_t result) {
  return check_status<nvmlReturn_t, NVML_SUCCESS>(result, "NVML Runtime",
                                                  nvmlErrorString);
}
