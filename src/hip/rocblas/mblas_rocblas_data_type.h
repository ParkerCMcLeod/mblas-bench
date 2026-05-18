#pragma once

#include <rocblas/internal/rocblas-types.h>
#include "mblas_data_type.h"
#include "mblas_backend_type.h"

class mblas_rocblas_data_type: public mblas_backend_type<mblas_rocblas_data_type, mblas_data_type, rocblas_datatype> {
  using Base = mblas_backend_type<mblas_rocblas_data_type, mblas_data_type, rocblas_datatype>;
 public:
  using Base::Base;
  using Base::operator=;

  static const std::map<mblas_data_type, rocblas_datatype>& get_mappings();

  // Legacy API compatibility (convert_to_hip is a misnomer; should be convert_to_rocblas)
  static rocblas_datatype convert_to_hip(mblas_rocblas_data_type data) { return data.convert(); }
  static rocblas_datatype convert_to_hip(const mblas_rocblas_data_type *data) { return data->convert(); }

  std::string to_string() const override { return mblas_data_type::to_string("rocblas"); }
};
