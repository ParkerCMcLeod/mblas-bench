#include "mblas_hipblas_compute_type.h"

// Used for converting mblas type to hipblas type
static const std::map<mblas_compute_type_enum, hipblasComputeType_t> hipblas_compute_type_mappings = {
    {MBLAS_COMPUTE_16F, HIPBLAS_COMPUTE_16F},
    {MBLAS_COMPUTE_16F_PEDANTIC, HIPBLAS_COMPUTE_16F_PEDANTIC},
    {MBLAS_COMPUTE_32F, HIPBLAS_COMPUTE_32F},
    {MBLAS_COMPUTE_32F_PEDANTIC, HIPBLAS_COMPUTE_32F_PEDANTIC},
    {MBLAS_COMPUTE_32F_FAST_16F, HIPBLAS_COMPUTE_32F_FAST_16F},
    {MBLAS_COMPUTE_32F_FAST_16BF, HIPBLAS_COMPUTE_32F_FAST_16BF},
    {MBLAS_COMPUTE_32F_FAST_TF32, HIPBLAS_COMPUTE_32F_FAST_TF32},
    {MBLAS_COMPUTE_64F, HIPBLAS_COMPUTE_64F},
    {MBLAS_COMPUTE_64F_PEDANTIC, HIPBLAS_COMPUTE_64F_PEDANTIC},
    {MBLAS_COMPUTE_32I, HIPBLAS_COMPUTE_32I},
    {MBLAS_COMPUTE_32I_PEDANTIC, HIPBLAS_COMPUTE_32I_PEDANTIC},
};

const std::map<mblas_compute_type_enum, hipblasComputeType_t>& mblas_hipblas_compute_type::get_mappings() {
    return hipblas_compute_type_mappings;
}
