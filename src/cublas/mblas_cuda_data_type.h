#pragma once

#include <cuda_runtime.h>
//#include <cublas.h>
#include <cublasLt.h>
#include "mblas_data_type.h"
#include "mblas_backend_type.h"

class mblas_cuda_data_type: public mblas_backend_type<mblas_cuda_data_type, mblas_data_type, cudaDataType> {
  using Base = mblas_backend_type<mblas_cuda_data_type, mblas_data_type, cudaDataType>;
 public:
  using Base::Base;
  using Base::operator=;

  static const std::map<mblas_data_type, cudaDataType>& get_mappings();

  // Legacy API compatibility
  static cudaDataType convert_to_cuda(mblas_cuda_data_type data) { return data.convert(); }
  static cudaDataType convert_to_cuda(const mblas_cuda_data_type *data) { return data->convert(); }

  std::string to_string() const override { return mblas_data_type::to_string("CUDA"); }

  // CUDA-specific methods
  mblas_cuda_data_type get_scale_type();

#if (ENABLE_CUDA_FP4)
  cublasLtMatmulMatrixScale_t get_scale_mode();
#endif
};
