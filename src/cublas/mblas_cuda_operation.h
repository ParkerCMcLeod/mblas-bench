#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mblas_operation.h"
#include "mblas_backend_type.h"

class mblas_cuda_operation: public mblas_backend_type<mblas_cuda_operation, mblas_operation, cublasOperation_t> {
  using Base = mblas_backend_type<mblas_cuda_operation, mblas_operation, cublasOperation_t>;
 public:
  using Base::Base;
  using Base::operator=;

  static const std::map<mblas_operation, cublasOperation_t>& get_mappings();

  // Legacy API compatibility
  static cublasOperation_t convert_to_cuda(mblas_cuda_operation data) { return data.convert(); }
  static cublasOperation_t convert_to_cuda(const mblas_cuda_operation *data) { return data->convert(); }
  cublasOperation_t convert_to_cuda() { return convert(); }

  std::string to_string() const override { return mblas_operation::to_string("CUBLAS"); }
};
