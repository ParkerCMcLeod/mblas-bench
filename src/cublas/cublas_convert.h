#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if (ENABLE_CUDA_FP4)
#include <cuda_fp4.h>
#endif
#include "mblas_cuda_data_type.h"

void copy_and_convert(mblas_cuda_data_type precision, void *host_a, void *devA, long x,
                    long y, int batchsz, long long stride);
void * convert_scalar(mblas_cuda_data_type precision, void *scalar);
void copy_and_convert_scalar(mblas_cuda_data_type scalarPrecision, void *hostScalar,
                          void *devScalar);
