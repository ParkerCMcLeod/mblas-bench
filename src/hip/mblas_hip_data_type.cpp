#include "mblas_hip_data_type.h"

static const std::map<mblas_data_type, hipDataType> hip_data_type_mappings = {
    {mblas_data_type::MBLAS_R_16F,  HIP_R_16F},
    {mblas_data_type::MBLAS_C_16F,  HIP_C_16F},
    {mblas_data_type::MBLAS_R_16BF, HIP_R_16BF},
    {mblas_data_type::MBLAS_C_16BF, HIP_C_16BF},
    {mblas_data_type::MBLAS_R_32F,  HIP_R_32F},
    {mblas_data_type::MBLAS_C_32F,  HIP_C_32F},
    {mblas_data_type::MBLAS_R_64F,  HIP_R_64F},
    {mblas_data_type::MBLAS_C_64F,  HIP_C_64F},
    {mblas_data_type::MBLAS_R_4I,   HIP_R_4I},
    {mblas_data_type::MBLAS_C_4I,   HIP_C_4I},
    {mblas_data_type::MBLAS_R_4U,   HIP_R_4U},
    {mblas_data_type::MBLAS_C_4U,   HIP_C_4U},
    {mblas_data_type::MBLAS_R_8I,   HIP_R_8I},
    {mblas_data_type::MBLAS_C_8I,   HIP_C_8I},
    {mblas_data_type::MBLAS_R_8U,   HIP_R_8U},
    {mblas_data_type::MBLAS_C_8U,   HIP_C_8U},
    {mblas_data_type::MBLAS_R_16I,  HIP_R_16I},
    {mblas_data_type::MBLAS_C_16I,  HIP_C_16I},
    {mblas_data_type::MBLAS_R_16U,  HIP_R_16U},
    {mblas_data_type::MBLAS_C_16U,  HIP_C_16U},
    {mblas_data_type::MBLAS_R_32I,  HIP_R_32I},
    {mblas_data_type::MBLAS_C_32I,  HIP_C_32I},
    {mblas_data_type::MBLAS_R_32U,  HIP_R_32U},
    {mblas_data_type::MBLAS_C_32U,  HIP_C_32U},
    {mblas_data_type::MBLAS_R_64I,  HIP_R_64I},
    {mblas_data_type::MBLAS_C_64I,  HIP_C_64I},
    {mblas_data_type::MBLAS_R_64U,  HIP_R_64U},
    {mblas_data_type::MBLAS_C_64U,  HIP_C_64U},
    {mblas_data_type::MBLAS_R_8F_E4M3, HIP_R_8F_E4M3},
    {mblas_data_type::MBLAS_R_8F_E5M2, HIP_R_8F_E5M2},
    #if defined(HIPRT_VERSION) && HIPRT_VERSION >= 70000000
    {mblas_data_type::MBLAS_R_6F_E2M3, HIP_R_6F_E2M3},
    {mblas_data_type::MBLAS_R_6F_E3M2, HIP_R_6F_E3M2},
    {mblas_data_type::MBLAS_R_4F_E2M1, HIP_R_4F_E2M1},
    #endif
};

const std::map<mblas_data_type, hipDataType>& mblas_hip_data_type::get_mappings() {
    return hip_data_type_mappings;
}
