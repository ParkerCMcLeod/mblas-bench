#include "mblas_cuda_data_type.h"
#include <iostream>

static const std::map<mblas_data_type, cudaDataType> cuda_data_type_mappings = {
    {mblas_data_type::MBLAS_R_16F,  CUDA_R_16F},
    {mblas_data_type::MBLAS_C_16F,  CUDA_C_16F},
    {mblas_data_type::MBLAS_R_16BF, CUDA_R_16BF},
    {mblas_data_type::MBLAS_C_16BF, CUDA_C_16BF},
    {mblas_data_type::MBLAS_R_32F,  CUDA_R_32F},
    {mblas_data_type::MBLAS_C_32F,  CUDA_C_32F},
    {mblas_data_type::MBLAS_R_64F,  CUDA_R_64F},
    {mblas_data_type::MBLAS_C_64F,  CUDA_C_64F},
    {mblas_data_type::MBLAS_R_4I,   CUDA_R_4I},
    {mblas_data_type::MBLAS_C_4I,   CUDA_C_4I},
    {mblas_data_type::MBLAS_R_4U,   CUDA_R_4U},
    {mblas_data_type::MBLAS_C_4U,   CUDA_C_4U},
    {mblas_data_type::MBLAS_R_8I,   CUDA_R_8I},
    {mblas_data_type::MBLAS_C_8I,   CUDA_C_8I},
    {mblas_data_type::MBLAS_R_8U,   CUDA_R_8U},
    {mblas_data_type::MBLAS_C_8U,   CUDA_C_8U},
    {mblas_data_type::MBLAS_R_16I,  CUDA_R_16I},
    {mblas_data_type::MBLAS_C_16I,  CUDA_C_16I},
    {mblas_data_type::MBLAS_R_16U,  CUDA_R_16U},
    {mblas_data_type::MBLAS_C_16U,  CUDA_C_16U},
    {mblas_data_type::MBLAS_R_32I,  CUDA_R_32I},
    {mblas_data_type::MBLAS_C_32I,  CUDA_C_32I},
    {mblas_data_type::MBLAS_R_32U,  CUDA_R_32U},
    {mblas_data_type::MBLAS_C_32U,  CUDA_C_32U},
    {mblas_data_type::MBLAS_R_64I,  CUDA_R_64I},
    {mblas_data_type::MBLAS_C_64I,  CUDA_C_64I},
    {mblas_data_type::MBLAS_R_64U,  CUDA_R_64U},
    {mblas_data_type::MBLAS_C_64U,  CUDA_C_64U},
    {mblas_data_type::MBLAS_R_8F_E4M3, CUDA_R_8F_E4M3},
    {mblas_data_type::MBLAS_R_8F_E5M2, CUDA_R_8F_E5M2},
#if (ENABLE_CUDA_FP4)
    {mblas_data_type::MBLAS_R_8F_UE4M3, CUDA_R_8F_UE4M3},
    {mblas_data_type::MBLAS_R_8F_UE8M0, CUDA_R_8F_UE8M0},
    {mblas_data_type::MBLAS_R_6F_E2M3, CUDA_R_6F_E2M3},
    {mblas_data_type::MBLAS_R_6F_E3M2, CUDA_R_6F_E3M2},
    {mblas_data_type::MBLAS_R_4F_E2M1, CUDA_R_4F_E2M1},
#endif
};

const std::map<mblas_data_type, cudaDataType>& mblas_cuda_data_type::get_mappings() {
    return cuda_data_type_mappings;
}

mblas_cuda_data_type mblas_cuda_data_type::get_scale_type() {
  if (*this == MBLAS_R_8F_E4M3 || *this == MBLAS_R_8F_E5M2) {
    return mblas_data_type_enum::MBLAS_R_8F_UE8M0;
  } else if (*this == MBLAS_R_4F_E2M1 ) {
    return mblas_data_type_enum::MBLAS_R_8F_UE4M3;
  }
  return mblas_data_type_enum::MBLAS_R_32F;
}

#if (ENABLE_CUDA_FP4)
cublasLtMatmulMatrixScale_t mblas_cuda_data_type::get_scale_mode() {
  if (*this == MBLAS_R_8F_E4M3 || *this == MBLAS_R_8F_E5M2) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  } else if (*this == MBLAS_R_4F_E2M1 ) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  }
  return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
}
#endif
