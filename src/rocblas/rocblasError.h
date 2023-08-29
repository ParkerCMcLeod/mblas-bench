#pragma once
#include <assert.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

const char *rocblasGetErrorString(rocblas_status status);
const char *hipblasGetErrorString(hipblasStatus_t status);
//
//// Convenience function for checking CUDA runtime API results
//// can be wrapped around any runtime API call. No-op in release builds.
// inline cudaError_t checkCuda(cudaError_t result);
//
// inline cublasStatus_t checkCublas(cublasStatus_t result);

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
static inline hipError_t checkHip(hipError_t result) {
  if (result != hipSuccess) {
    std::cerr << "HIP Runtime Error: " << hipGetErrorString(result)
              << std::endl;
    // fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
  return result;
}

static inline rocblas_status checkRocblas(rocblas_status result) {
  if (result != rocblas_status_success) {
    std::cerr << "rocBLAS Runtime Error: " << rocblas_status_to_string(result)
              << std::endl;
    // fprintf(stderr, "HIP Runtime Error: %s\n",
    // hipGetErrorString(result));
    assert(result == rocblas_status_success);
  }
  return result;
}

static inline hipblasStatus_t checkHipblas(hipblasStatus_t result) {
  if (result != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "hipBLAS Runtime Error: " << hipblasStatusToString(result)
              << std::endl;
    // fprintf(stderr, "HIP Runtime Error: %s\n",
    // hipGetErrorString(result));
    assert(result == HIPBLAS_STATUS_SUCCESS);
  }
  return result;
}