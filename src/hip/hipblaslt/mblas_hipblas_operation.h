#pragma once

#include <hipblas/hipblas.h>
#include "mblas_operation.h"
#include "mblas_backend_type.h"

class mblas_hipblas_operation: public mblas_backend_type<mblas_hipblas_operation, mblas_operation, hipblasOperation_t> {
  using Base = mblas_backend_type<mblas_hipblas_operation, mblas_operation, hipblasOperation_t>;
 public:
  using Base::Base;
  using Base::operator=;

  static const std::map<mblas_operation, hipblasOperation_t>& get_mappings();

  // Legacy API compatibility
  static hipblasOperation_t convert_to_hip(mblas_hipblas_operation data) { return data.convert(); }
  static hipblasOperation_t convert_to_hip(const mblas_hipblas_operation *data) { return data->convert(); }
  hipblasOperation_t convert_to_hip() { return convert(); }

  std::string to_string() const override { return mblas_operation::to_string("HIPBLAS"); }
};
