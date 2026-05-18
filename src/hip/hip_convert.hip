#include "hip_convert.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include "hip_create_allocate.h"
#include "hip_error.h"
#include "backend_convert.h"

// ---------------------------------------------------------------------------
// Vendor-specific conversion functors for intrinsics not handled by
// static_cast (BF16, FP16, FP8).  These wrap HIP intrinsics and are
// passed to the generic convert_kernel_fn template.
// ---------------------------------------------------------------------------

struct FloatToBF16 {
  __device__ hip_bfloat16 operator()(float v) const {
    return __float2bfloat16(v);
  }
};

struct FloatToFP16 {
  __device__ __half operator()(float v) const {
    return __float2half(v);
  }
};

struct FloatToFP8 {
  __hip_fp8_interpretation_t interp;
  __device__ __hip_fp8_storage_t operator()(float v) const {
    return __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, interp);
  }
};

void copy_and_convert(mblas_data_type precision, void *hostA, void *devA, long x, long y,
                      int batchsz, long long stride)
{
  if (batchsz * x * y == 0)
  {
    return;
  }
  long long base = x * y;
  long long total_elements = stride * (batchsz - 1) + base;
  long hostsz = type_call_host<sizeofCUDT>(precision);
  if (precision == mblas_data_type(MBLAS_C_16F) || precision == mblas_data_type(MBLAS_R_16F))
  {
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz, hipMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size, num_blocks;
    convert_grid_dims(num_elements, num_blocks, block_size);
    convert_kernel_fn<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA, FloatToFP16{});
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else if (precision == mblas_data_type(MBLAS_C_16BF) || precision == mblas_data_type(MBLAS_R_16BF))
  {
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz, hipMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size, num_blocks;
    convert_grid_dims(num_elements, num_blocks, block_size);
    convert_kernel_fn<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (hip_bfloat16 *)devA, FloatToBF16{});
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else if (precision == mblas_data_type::MBLAS_R_8F_E4M3 ||
           precision == mblas_data_type::MBLAS_R_8F_E5M2)
  {
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz, hipMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size, num_blocks;
    convert_grid_dims(num_elements, num_blocks, block_size);
    __hip_fp8_interpretation_t interp;
    if (precision == mblas_data_type::MBLAS_R_8F_E4M3)
    {
      interp = __HIP_E4M3;
    }
    else if (precision == mblas_data_type::MBLAS_R_8F_E5M2)
    {
      interp = __HIP_E5M2;
    }
    convert_kernel_fn<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__hip_fp8_storage_t *)devA, FloatToFP8{interp});
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else
  {
    check_hip(hipMemcpy(devA, hostA,
                        get_malloc_size_host(precision, x, y, batchsz, stride),
                        hipMemcpyHostToDevice));
  }
}

void *convert_scalar(mblas_data_type precision, void *scalar)
{
  return convert_scalar_impl(precision, scalar);
}
