#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mblas_compute_type.h"
#include "mblas_backend_type.h"

class mblas_cuda_compute_type: public mblas_backend_type<mblas_cuda_compute_type, mblas_compute_type, cublasComputeType_t, mblas_compute_type_enum> {
  using Base = mblas_backend_type<mblas_cuda_compute_type, mblas_compute_type, cublasComputeType_t, mblas_compute_type_enum>;
 public:
  using Base::Base;
  using Base::operator=;

  static const std::map<mblas_compute_type_enum, cublasComputeType_t>& get_mappings();

  // Legacy API compatibility
  static cublasComputeType_t convert_to_cuda(mblas_cuda_compute_type data) { return data.convert(); }
  static cublasComputeType_t convert_to_cuda(const mblas_cuda_compute_type *data) { return data->convert(); }

  std::string to_string() const override { return mblas_compute_type::to_string("CUBLAS"); }
};
