#include "mblas_rocblas_compute_type.h"

#include <iostream>

#include "mblas_rocblas_data_type.h"

// Used for converting mblas type to rocblas type
const std::map<mblas_compute_type_enum, rocblas_computetype> mblas_rocblas_compute_type::compute_mappings = {
    {MBLAS_COMPUTE_32F, rocblas_compute_type_f32},
    {MBLAS_COMPUTE_32F_8F_8F, rocblas_compute_type_f8_f8_f32},
    {MBLAS_COMPUTE_32F_8F_8BF, rocblas_compute_type_f8_bf8_f32},
    {MBLAS_COMPUTE_32F_8BF_8F, rocblas_compute_type_bf8_f8_f32},
    {MBLAS_COMPUTE_32F_8BF_8BF, rocblas_compute_type_bf8_bf8_f32,},
    {MBLAS_COMPUTE_NULL, rocblas_compute_type_invalid},
};

const std::vector<std::pair<mblas_compute_type_enum, mblas_data_type_enum>> prec_mappings = {
    {MBLAS_COMPUTE_64F, MBLAS_R_64F},
    {MBLAS_COMPUTE_32F, MBLAS_R_32F},
    {MBLAS_COMPUTE_16F, MBLAS_R_16F},
    {MBLAS_COMPUTE_32I, MBLAS_R_32I},
    {MBLAS_COMPUTE_64F, MBLAS_C_64F},
    {MBLAS_COMPUTE_32F, MBLAS_C_32F},
};

rocblas_computetype mblas_rocblas_compute_type::convert_to_rocm(mblas_rocblas_compute_type data)  { return convert_to_rocm(&data); }

rocblas_computetype mblas_rocblas_compute_type::convert_to_rocm(const mblas_rocblas_compute_type *data) {
  try {
    return compute_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to rocBLAS Compute Type " << data->to_string() << std::endl;
    throw e;
    return rocblas_compute_type_f32;
  }
}

mblas_rocblas_compute_type::operator rocblas_computetype() const {
  return convert_to_rocm(this);
}

rocblas_datatype mblas_rocblas_compute_type::convert_to_rocm_data(const mblas_rocblas_compute_type *data) {
  for (auto ele : prec_mappings) {
    mblas_rocblas_data_type rdata = mblas_rocblas_data_type(ele.second);
    if (ele.first == *data && rdata.is_real() == data->roc_is_real) {
      return rocblas_datatype(rdata);
    }
  }
  std::cout << "Failed to convert to rocBLAS Data Type " << data->to_string() << std::endl;
  throw std::out_of_range("Value not found in list");
}

mblas_rocblas_compute_type::operator rocblas_datatype() const {
  return convert_to_rocm_data(this);
}

// void mblas_rocblas_compute_type::operator = (const rocblas_computetype cudt) {
//   for (auto ele : compute_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblas_rocblas_compute_type & mblas_rocblas_compute_type::operator = (const mblas_rocblas_compute_type& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

void mblas_rocblas_compute_type::set_compute(std::string computestr, mblas_data_type& precision) {
  mblas_compute_type::set_compute(computestr, precision);
  roc_is_real = precision.is_real();
}
