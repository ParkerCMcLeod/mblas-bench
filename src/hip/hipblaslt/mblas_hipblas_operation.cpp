#include "mblas_hipblas_operation.h"

static const std::map<mblas_operation, hipblasOperation_t> hipblas_operation_mappings = {
    {mblas_operation::MBLAS_OP_N,    HIPBLAS_OP_N},
    {mblas_operation::MBLAS_OP_T,    HIPBLAS_OP_T},
    {mblas_operation::MBLAS_OP_C,    HIPBLAS_OP_C},
};

const std::map<mblas_operation, hipblasOperation_t>& mblas_hipblas_operation::get_mappings() {
    return hipblas_operation_mappings;
}
