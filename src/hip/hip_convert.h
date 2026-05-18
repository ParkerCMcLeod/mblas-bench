#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include "mblas_data_type.h"

void copy_and_convert(mblas_data_type precision, void *host_a, void *devA, long x,
                    long y, int batchsz, long long stride);
void *convert_scalar(mblas_data_type precision, void *scalar);
void copyAndConvertScalar(mblas_data_type scalarPrecision, void *hostScalar,
                          void *devScalar);
