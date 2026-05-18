#include "mblas_cuda_operation.h"
#include <iostream>

cublasOperation_t mblas_cuda_operation::convert_to_cuda(mblas_cuda_operation data)  { return convert_to_cuda(&data); }

cublasOperation_t mblas_cuda_operation::convert_to_cuda(const mblas_cuda_operation *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to CUDA Datatype " << data->to_string() << std::endl;
    throw e;
  }
}

cublasOperation_t mblas_cuda_operation::convert_to_cuda() {
  return mblas_cuda_operation::convert_to_cuda(this);
}

mblas_cuda_operation & mblas_cuda_operation::operator = (const mblas_cuda_operation& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

const std::map<mblas_operation, cublasOperation_t> mblas_cuda_operation::prec_mappings = {
    {MBLAS_OP_N,    CUBLAS_OP_N},
    {MBLAS_OP_T,    CUBLAS_OP_T},
    {MBLAS_OP_C,    CUBLAS_OP_C},
    {MBLAS_OP_CONJG,    CUBLAS_OP_CONJG},
};
