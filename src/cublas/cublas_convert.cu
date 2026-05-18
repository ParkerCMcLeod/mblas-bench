#include "cublas_convert.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if (ENABLE_CUDA_FP4)
#include <cuda_fp4.h>
#endif
#include "cublas_create_allocate.h"
#include "cuda_error.h"
#include "mblas_cuda_data_type.h"
#include "generic_setup.h"
#include "backend_convert.h"

// ---------------------------------------------------------------------------
// Vendor-specific conversion functors for intrinsics not handled by
// static_cast (FP8, E8M0, FP4).  These wrap CUDA intrinsics and are
// passed to the generic convert_kernel_fn template.
// ---------------------------------------------------------------------------

struct FloatToBF16 {
  __device__ __nv_bfloat16 operator()(float v) const {
    return __float2bfloat16(v);
  }
};

struct FloatToFP16 {
  __device__ __half operator()(float v) const {
    return __float2half(v);
  }
};

struct FloatToFP8 {
  __nv_fp8_interpretation_t interp;
  __device__ __nv_fp8_storage_t operator()(float v) const {
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, interp);
  }
};

struct FloatToE8M0 {
  __device__ __nv_fp8_storage_t operator()(float v) const {
    /*
     Rounding only controls the direction of rounding
     https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html
    */
    return __nv_cvt_float_to_e8m0(v, __NV_SATFINITE, cudaRoundZero);
  }
};

#if (ENABLE_CUDA_FP4)
struct Float2ToFP4x2 {
  __device__ __nv_fp4x2_storage_t operator()(float2 v) const {
    return __nv_cvt_float2_to_fp4x2(v, __NV_E2M1, cudaRoundNearest);
  }
};
#endif

/*
FYI:

cudaRoundMode
    cudaRoundNearest
    cudaRoundZero
    cudaRoundPosInf
    cudaRoundMinInf
*/

void copy_and_convert(mblas_cuda_data_type precision, void *host_a, void *devA, long x,
                      long y, int batchsz, long long stride)
{
  if (batchsz * x * y == 0)
  {
    // Matrix not used, don't copy
    return;
  }
  long hostsz = type_call_host<sizeofCUDT>(precision);
  long devsz = type_call_dev<sizeofCUDT>(precision);
  long long base = x * y;
  long long total_elements = stride * (batchsz - 1) + base;
  if (precision == mblas_data_type::MBLAS_C_16F || precision == mblas_data_type::MBLAS_R_16F)
  {
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size, num_blocks;
    convert_grid_dims(num_elements, num_blocks, block_size);
    convert_kernel_fn<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA, FloatToFP16{});
    check_cuda(cudaGetLastError());
    check_cuda(cudaFree(tmpA));
  }
  else if (precision == mblas_data_type::MBLAS_C_16BF || precision == mblas_data_type::MBLAS_R_16BF)
  {
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size, num_blocks;
    convert_grid_dims(num_elements, num_blocks, block_size);
    convert_kernel_fn<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_bfloat16 *)devA, FloatToBF16{});
    check_cuda(cudaGetLastError());
    check_cuda(cudaFree(tmpA));
  }
  else if (precision == mblas_data_type::MBLAS_R_8F_E4M3 ||
           precision == mblas_data_type::MBLAS_R_8F_E5M2 ||
           precision == mblas_data_type::MBLAS_R_8F_UE4M3)
  {
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size, num_blocks;
    convert_grid_dims(num_elements, num_blocks, block_size);
    __nv_fp8_interpretation_t interp;
    if (precision == mblas_data_type::MBLAS_R_8F_E4M3)
    {
      interp = __NV_E4M3;
    }
    else if (precision == mblas_data_type::MBLAS_R_8F_UE4M3)
    {
      interp = __NV_E4M3;
    }
    else if (precision == mblas_data_type::MBLAS_R_8F_E5M2)
    {
      interp = __NV_E5M2;
    }
    convert_kernel_fn<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_fp8_storage_t *)devA, FloatToFP8{interp});
    check_cuda(cudaGetLastError());
    check_cuda(cudaFree(tmpA));
  }
  else if (precision == mblas_data_type::MBLAS_R_8F_UE8M0)
  {
#if (ENABLE_CUDA_FP4)
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size, num_blocks;
    convert_grid_dims(num_elements, num_blocks, block_size);
    convert_kernel_fn<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_fp8_storage_t *)devA, FloatToE8M0{});
    check_cuda(cudaGetLastError());
    check_cuda(cudaFree(tmpA));
#endif
  }
  else if (precision == mblas_data_type::MBLAS_R_4F_E2M1)
  {
#if (ENABLE_CUDA_FP4)
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = ceil_division(total_elements, 2ll);
    long long block_size, num_blocks;
    convert_grid_dims(num_elements, num_blocks, block_size);
    convert_kernel_fn<<<num_blocks, block_size>>>((float2 *)tmpA, num_elements, (__nv_fp4x2_storage_t *)devA, Float2ToFP4x2{});
    check_cuda(cudaGetLastError());
    check_cuda(cudaFree(tmpA));
#endif
  }
  // else if (precision == mblas_data_type::MBLAS_C_8I || precision == mblas_data_type::MBLAS_R_8I)
  //{
  //   // Allocate memory in the device for host precision (float)
  //   void *tmpA = allocate_host_dev_array(precision, x, y, batchsz);
  //   check_cuda(cudaMemcpy(tmpA, host_a, batchsz * x * y * hostsz,
  //                        cudaMemcpyHostToDevice));
  //   int num_elements = batchsz * x * y;
  //   int block_size = 256;
  //   int num_blocks = (num_elements + block_size - 1) / block_size;
  //   floatToBfloat16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_bfloat16 *)devA);
  //   cudaFree(tmpA);
  // }
  else
  {
    check_cuda(cudaMemcpy(
        devA, host_a,
        static_cast<size_t>(get_malloc_size_host(precision, x, y, batchsz, stride)),
        cudaMemcpyHostToDevice));
  }
}

void *convert_scalar(mblas_cuda_data_type precision, void *scalar)
{
  return convert_scalar_impl(precision, scalar);
}
