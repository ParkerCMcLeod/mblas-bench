#pragma once

#include <hip/library_types.h>
#include "mblas_data_type.h"
#include "mblas_backend_type.h"

class mblas_hip_data_type: public mblas_backend_type<mblas_hip_data_type, mblas_data_type, hipDataType> {
  using Base = mblas_backend_type<mblas_hip_data_type, mblas_data_type, hipDataType>;
 public:
  using Base::Base;
  using Base::operator=;

  static const std::map<mblas_data_type, hipDataType>& get_mappings();

  // Legacy API compatibility
  static hipDataType convert_to_hip(mblas_hip_data_type data) { return data.convert(); }
  static hipDataType convert_to_hip(const mblas_hip_data_type *data) { return data->convert(); }

  std::string to_string() const override { return mblas_data_type::to_string("HIP"); }
};
