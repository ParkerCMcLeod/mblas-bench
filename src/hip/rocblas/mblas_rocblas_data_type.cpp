#include "mblas_rocblas_data_type.h"

static const std::map<mblas_data_type, rocblas_datatype> rocblas_data_type_mappings = {
    {mblas_data_type::MBLAS_R_16F,  rocblas_datatype_f16_r},
    {mblas_data_type::MBLAS_C_16F,  rocblas_datatype_f16_c},
    {mblas_data_type::MBLAS_R_16BF, rocblas_datatype_bf16_r},
    {mblas_data_type::MBLAS_C_16BF, rocblas_datatype_bf16_c},
    {mblas_data_type::MBLAS_R_32F,  rocblas_datatype_f32_r},
    {mblas_data_type::MBLAS_C_32F,  rocblas_datatype_f32_c},
    {mblas_data_type::MBLAS_R_64F,  rocblas_datatype_f64_r},
    {mblas_data_type::MBLAS_C_64F,  rocblas_datatype_f64_c},
    {mblas_data_type::MBLAS_R_8I,   rocblas_datatype_i8_r},
    {mblas_data_type::MBLAS_C_8I,   rocblas_datatype_i8_c},
    {mblas_data_type::MBLAS_R_8U,   rocblas_datatype_u8_r},
    {mblas_data_type::MBLAS_C_8U,   rocblas_datatype_u8_c},
    {mblas_data_type::MBLAS_R_32I,  rocblas_datatype_i32_r},
    {mblas_data_type::MBLAS_C_32I,  rocblas_datatype_i32_c},
    {mblas_data_type::MBLAS_R_32U,  rocblas_datatype_u32_r},
    {mblas_data_type::MBLAS_C_32U,  rocblas_datatype_u32_c},
#if (HIP_VERSION < 70000000)
    // Undocumented rocblas fp8 support, removed in rocm 7.0.0
    {mblas_data_type::MBLAS_R_8F_E4M3, rocblas_datatype_f8_r},
    {mblas_data_type::MBLAS_R_8F_E5M2, rocblas_datatype_bf8_r},
#endif
};

const std::map<mblas_data_type, rocblas_datatype>& mblas_rocblas_data_type::get_mappings() {
    return rocblas_data_type_mappings;
}
