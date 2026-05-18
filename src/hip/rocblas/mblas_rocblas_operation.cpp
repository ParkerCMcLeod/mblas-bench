#include "mblas_rocblas_operation.h"

static const std::map<mblas_operation, rocblas_operation> rocblas_operation_mappings = {
    {mblas_operation::MBLAS_OP_N,    rocblas_operation_none},
    {mblas_operation::MBLAS_OP_T,    rocblas_operation_transpose},
    {mblas_operation::MBLAS_OP_C,    rocblas_operation_conjugate_transpose},
};

const std::map<mblas_operation, rocblas_operation>& mblas_rocblas_operation::get_mappings() {
    return rocblas_operation_mappings;
}
