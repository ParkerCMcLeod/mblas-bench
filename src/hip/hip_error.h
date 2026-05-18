#pragma once
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#include "backend_error.h"

#if MBLAS_WITH_ROCBLAS
#include <rocblas/rocblas.h>
const char *rocblas_get_error_string(rocblas_status status);
#endif

const char *hipblas_get_error_string(hipblasStatus_t status);

static inline hipError_t check_hip(hipError_t result) {
  return check_status<hipError_t, hipSuccess>(result, "HIP Runtime",
                                              hipGetErrorString);
}

#if MBLAS_WITH_ROCBLAS
static inline rocblas_status check_rocblas(rocblas_status result) {
  return check_status<rocblas_status, rocblas_status_success>(
      result, "rocBLAS Runtime", rocblas_status_to_string);
}
#endif

static inline hipblasStatus_t check_hipblas(hipblasStatus_t result) {
  return check_status<hipblasStatus_t, HIPBLAS_STATUS_SUCCESS>(
      result, "hipBLAS Runtime", hipblas_get_error_string);
}
