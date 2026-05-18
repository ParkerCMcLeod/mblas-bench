#pragma once

// Generic conversion kernel templates shared by CUDA and HIP backends.
//
// Both nvcc and hipcc understand __global__, __device__, blockIdx, blockDim,
// and threadIdx, so this header compiles under either toolchain without any
// vendor-specific includes.
//
// Vendor-specific intrinsics (e.g. __nv_cvt_float_to_fp8 vs
// __hip_cvt_float_to_fp8) are NOT used here. Instead, backends supply a
// functor to convert_kernel_fn that wraps the appropriate intrinsic.

#include <complex>

// ---------------------------------------------------------------------------
// Generic element-wise conversion kernel (uses static_cast)
// ---------------------------------------------------------------------------
// Suitable for conversions where static_cast produces the correct result,
// e.g. float -> __half via __float2half (which both compilers expose).
// For conversions requiring special intrinsics (FP8, FP4, E8M0), backends
// should use convert_kernel_fn with a custom functor instead.
template <typename SrcType, typename DstType>
__global__ void convert_kernel(const SrcType *input, size_t num_elements,
                               DstType *output) {
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < static_cast<long>(num_elements)) {
    output[idx] = static_cast<DstType>(input[idx]);
  }
}

// ---------------------------------------------------------------------------
// Generic element-wise conversion kernel with a user-supplied functor
// ---------------------------------------------------------------------------
// The functor F must provide:
//   __device__ DstType operator()(SrcType val) const;
//
// Example usage (CUDA FP8):
//   struct FloatToFP8 {
//     __nv_fp8_interpretation_t interp;
//     __device__ __nv_fp8_storage_t operator()(float v) const {
//       return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, interp);
//     }
//   };
//   convert_kernel_fn<<<blocks, threads>>>(input, n, output, FloatToFP8{interp});
template <typename SrcType, typename DstType, typename F>
__global__ void convert_kernel_fn(const SrcType *input, size_t num_elements,
                                  DstType *output, F func) {
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < static_cast<long>(num_elements)) {
    output[idx] = func(input[idx]);
  }
}

// ---------------------------------------------------------------------------
// Kernel launch helpers — compute grid dimensions
// ---------------------------------------------------------------------------
inline void convert_grid_dims(long long num_elements, long long &num_blocks,
                              long long &block_size) {
  block_size = 256;
  num_blocks = (num_elements + block_size - 1) / block_size;
}

// ---------------------------------------------------------------------------
// convert_scalar: in-place scalar conversion (float -> __half)
// ---------------------------------------------------------------------------
// Both CUDA and HIP define __half and __float2half identically, so this
// template works under either compiler.  The DataType parameter is the
// backend's precision enum (mblas_cuda_data_type or mblas_data_type).
//
// Only FP16 (real + complex) requires conversion; all other types are a no-op.
template <typename DataType>
void *convert_scalar_impl(DataType precision, void *scalar) {
  if (precision == DataType::MBLAS_R_16F) {
    float scalarVal = *static_cast<float *>(scalar);
    __half *hscalar = (__half *)scalar;
    *hscalar = __float2half(scalarVal);
    return scalar;
  } else if (precision == DataType::MBLAS_C_16F) {
    std::complex<float> *cFloat = static_cast<std::complex<float> *>(scalar);
    float realVal = cFloat->real();
    float imagVal = cFloat->imag();

    std::complex<__half> *cHalf = (std::complex<__half> *)scalar;
    *cHalf = std::complex<__half>(__float2half(realVal), __float2half(imagVal));
    return scalar;
  } else {
    return scalar;
  }
}
