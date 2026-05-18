#pragma once

#include <rocblas/internal/rocblas-types.h>
#include "mblas_operation.h"
#include "mblas_backend_type.h"

class mblas_rocblas_operation: public mblas_backend_type<mblas_rocblas_operation, mblas_operation, rocblas_operation> {
  using Base = mblas_backend_type<mblas_rocblas_operation, mblas_operation, rocblas_operation>;
 public:
  using Base::Base;
  using Base::operator=;

  static const std::map<mblas_operation, rocblas_operation>& get_mappings();

  // Legacy API compatibility
  static rocblas_operation convert_to_rocm(mblas_rocblas_operation data) { return data.convert(); }
  static rocblas_operation convert_to_rocm(const mblas_rocblas_operation *data) { return data->convert(); }
  rocblas_operation convert_to_rocm() { return convert(); }

  std::string to_string() const override { return mblas_operation::to_string("rocblas"); }
};
