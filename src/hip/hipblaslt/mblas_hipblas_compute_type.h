#pragma once

#include <hipblas/hipblas.h>
#include "mblas_compute_type.h"
#include "mblas_backend_type.h"

class mblas_hipblas_compute_type: public mblas_backend_type<mblas_hipblas_compute_type, mblas_compute_type, hipblasComputeType_t, mblas_compute_type_enum> {
  using Base = mblas_backend_type<mblas_hipblas_compute_type, mblas_compute_type, hipblasComputeType_t, mblas_compute_type_enum>;
 public:
  using Base::Base;
  using Base::operator=;

  static const std::map<mblas_compute_type_enum, hipblasComputeType_t>& get_mappings();

  // Legacy API compatibility
  static hipblasComputeType_t convert_to_hip(mblas_hipblas_compute_type data) { return data.convert(); }
  static hipblasComputeType_t convert_to_hip(const mblas_hipblas_compute_type *data) { return data->convert(); }

  std::string to_string() const override { return mblas_compute_type::to_string("HIPBLAS"); }
};
