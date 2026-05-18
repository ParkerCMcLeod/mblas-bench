#include "mblas_rocblas_operation.h"
#include <iostream>

rocblas_operation mblas_rocblas_operation::convert_to_rocm(mblas_rocblas_operation data)  { return convert_to_rocm(&data); }

rocblas_operation mblas_rocblas_operation::convert_to_rocm(const mblas_rocblas_operation *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to rocBLAS operation" << data->to_string() << std::endl;
    throw e;
  }
}

rocblas_operation mblas_rocblas_operation::convert_to_rocm() {
  return mblas_rocblas_operation::convert_to_rocm(this);
}

mblas_rocblas_operation & mblas_rocblas_operation::operator = (const mblas_rocblas_operation& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

const std::map<mblas_operation, rocblas_operation> mblas_rocblas_operation::prec_mappings = {
    {MBLAS_OP_N,    rocblas_operation_none},
    {MBLAS_OP_T,    rocblas_operation_transpose},
    {MBLAS_OP_C,    rocblas_operation_conjugate_transpose},
};
