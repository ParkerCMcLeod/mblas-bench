#include "mblas_cuda_operation.h"

static const std::map<mblas_operation, cublasOperation_t> cuda_operation_mappings = {
    {mblas_operation::MBLAS_OP_N,    CUBLAS_OP_N},
    {mblas_operation::MBLAS_OP_T,    CUBLAS_OP_T},
    {mblas_operation::MBLAS_OP_C,    CUBLAS_OP_C},
    {mblas_operation::MBLAS_OP_CONJG,    CUBLAS_OP_CONJG},
};

const std::map<mblas_operation, cublasOperation_t>& mblas_cuda_operation::get_mappings() {
    return cuda_operation_mappings;
}
