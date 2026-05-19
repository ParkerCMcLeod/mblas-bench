#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <string>

// Thread-local storage for the formatted error string so the returned const
// char* remains valid until the next call on the same thread.
static thread_local std::string g_last_cublas_err;

const char *cublas_get_error_string(cublasStatus_t status) {
  // Prefer cuBLAS's own status->string API (added in CUDA 11). It knows about
  // codes that didn't exist when this code was written, including any added
  // in CUDA 12.x / 13.x. We also append the numeric value so unrecognized
  // codes are at least diagnosable from the printed message.
  const char *name = cublasGetStatusName(status);
  const char *desc = cublasGetStatusString(status);
  g_last_cublas_err = std::string(name ? name : "?")
                    + " (" + std::to_string(static_cast<int>(status)) + "): "
                    + (desc ? desc : "no description");
  return g_last_cublas_err.c_str();
}
