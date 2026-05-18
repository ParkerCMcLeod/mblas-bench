#pragma once

#include <cassert>
#include <iostream>

// Generic GPU status checking template.
//
// Usage:
//   // For runtime API errors (where SuccessValue is an enum constant):
//   static inline cudaError_t check_cuda(cudaError_t result) {
//       return check_status<cudaError_t, cudaSuccess>(result, "CUDA Runtime",
//                                                      cudaGetErrorString);
//   }
//
//   // For BLAS library errors (where the error string comes from a custom function):
//   static inline cublasStatus_t check_cublas(cublasStatus_t result) {
//       return check_status<cublasStatus_t, CUBLAS_STATUS_SUCCESS>(
//           result, "cuBLAS Runtime", cublas_get_error_string);
//   }
//
// The template is header-only and vendor-neutral: it includes no CUDA or HIP
// headers.  Each backend's thin wrapper header includes the relevant vendor
// headers and instantiates check_status with the correct types.

template <typename StatusType, StatusType SuccessValue, typename ErrorStringFn>
static inline StatusType check_status(StatusType result,
                                      const char *label,
                                      ErrorStringFn error_string_fn) {
  if (result != SuccessValue) {
    std::cerr << label << " Error: " << error_string_fn(result) << std::endl;
    assert(result == SuccessValue);
  }
  return result;
}
