#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

#include "cublas/cublasGemm.h"
#include "genericGemm.h"

// #include "fp16_conversion.h"
#include "third_party/cxxopts.hpp"

//#include "error_handling.h"
//#include "create-allocate.h"
#include "cublas/cudaError.h"
using std::cout;
using std::endl;

// const char *cublasGetErrorString(cublasStatus_t status) {
//  switch (status) {
//  case CUBLAS_STATUS_SUCCESS:
//    return "CUBLAS_STATUS_SUCCESS";
//  case CUBLAS_STATUS_NOT_INITIALIZED:
//    return "CUBLAS_STATUS_NOT_INITIALIZED";
//  case CUBLAS_STATUS_ALLOC_FAILED:
//    return "CUBLAS_STATUS_ALLOC_FAILED";
//  case CUBLAS_STATUS_INVALID_VALUE:
//    return "CUBLAS_STATUS_INVALID_VALUE";
//  case CUBLAS_STATUS_ARCH_MISMATCH:
//    return "CUBLAS_STATUS_ARCH_MISMATCH";
//  case CUBLAS_STATUS_MAPPING_ERROR:
//    return "CUBLAS_STATUS_MAPPING_ERROR";
//  case CUBLAS_STATUS_EXECUTION_FAILED:
//    return "CUBLAS_STATUS_EXECUTION_FAILED";
//  case CUBLAS_STATUS_INTERNAL_ERROR:
//    return "CUBLAS_STATUS_INTERNAL_ERROR";
//  }
//  return "unknown error";
//}
//
//// Convenience function for checking CUDA runtime API results
//// can be wrapped around any runtime API call. No-op in release builds.
// inline cudaError_t checkCuda(cudaError_t result) {
//  if (result != cudaSuccess) {
//    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//    assert(result == cudaSuccess);
//  }
//  return result;
//}
//
// inline cublasStatus_t checkCublas(cublasStatus_t result) {
//  if (result != CUBLAS_STATUS_SUCCESS) {
//    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
//    assert(result == CUBLAS_STATUS_SUCCESS);
//  }
//  return result;
//}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(double *A, int nr_rows_A, int nr_cols_A, int batch = 1) {
  int a = 1;

  for (int i = 0; i < nr_rows_A * nr_cols_A * batch; i++) {
    A[i] = (double)rand() / (double)(RAND_MAX / a);
  }
}

int main(int argc, char **argv) {
  // print device info
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Device " << i << ": " << prop.name << ", "
              << prop.clockRate / 1000 << " MHZ"
              << ", " << prop.memoryClockRate / 1000 << " MHZ" << std::endl;
  }
  // parse input arguments
  cxxopts::Options options("cublas_bench", "Benchmark Cublas");
  std::string supPrec =
      "h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r";
  auto opp_adder = options.add_options();
  opp_adder("m,sizem", "Specific matrix size",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("n,sizen", "Specific matrix size",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("k,sizek", "Specific matrix size",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("f,function", "BLAS function to test",
            cxxopts::value<std::string>());
  opp_adder("r,precision", "Precision. Options: " + supPrec + " ",
            cxxopts::value<std::string>()->default_value("f32_r"));
  opp_adder("transposeA", "transposeA",
            cxxopts::value<std::string>()->default_value("N"));
  opp_adder("transposeB", "transposeB",
            cxxopts::value<std::string>()->default_value("N"));
  opp_adder("alpha", "specifies the scalar alpha  (Default value is: 1)",
            cxxopts::value<std::string>()->default_value("1"));
  opp_adder(
      "alphai",
      "specifies the imaginary part of the scalar alpha  (Default value is: 0)",
      cxxopts::value<std::string>()->default_value("0"));
  opp_adder("beta", "specifies the scalar beta  (Default value is: 0)",
            cxxopts::value<std::string>()->default_value("0"));
  opp_adder(
      "betai",
      "specifies the imaginary part of the scalar beta  (Default value is: 0)",
      cxxopts::value<std::string>()->default_value("0"));
  opp_adder("lda",
            "Leading dimension of matrix A, is only applicable to BLAS-2 & "
            "BLAS-3.  (Default value is: 128)",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("ldb",
            "Leading dimension of matrix A, is only applicable to BLAS-2 & "
            "BLAS-3.  (Default value is: 128)",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("ldc",
            "Leading dimension of matrix A, is only applicable to BLAS-2 & "
            "BLAS-3.  (Default value is: 128)",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("ldd",
            "Leading dimension of matrix A, is only applicable to BLAS-EX.  "
            "(Default value is: 128)",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("stride_a",
            "Specific stride of strided_batched matrix A, is only applicable "
            "to strided batchedBLAS-2 and BLAS-3: second dimension * leading "
            "dimension.  (Default value is: 16384)",
            cxxopts::value<long long int>()->default_value("16384"));
  opp_adder("stride_b",
            "Specific stride of strided_batched matrix B, is only applicable "
            "to strided batchedBLAS-2 and BLAS-3: second dimension * leading "
            "dimension.  (Default value is: 16384)",
            cxxopts::value<long long int>()->default_value("16384"));
  opp_adder("stride_c",
            "Specific stride of strided_batched matrix C, is only applicable "
            "to strided batchedBLAS-2 and BLAS-3: second dimension * leading "
            "dimension.  (Default value is: 16384)",
            cxxopts::value<long long int>()->default_value("16384"));
  opp_adder("stride_d",
            "Specific stride of strided_batched matrix D, is only applicable "
            "to strided batchedBLAS_EX: second dimension * leading dimension.  "
            "(Default value is: 16384)",
            cxxopts::value<long long int>()->default_value("16384"));
  opp_adder("a_type",
            "Precision of matrix A. Options:" + supPrec + ". " +
                "Defaults to the value of -r/--precision",
            cxxopts::value<std::string>()->default_value(""));
  opp_adder("b_type",
            "Precision of matrix B. Options:" + supPrec + ". " +
                "Defaults to the value of -r/--precision",
            cxxopts::value<std::string>()->default_value(""));
  opp_adder("c_type",
            "Precision of matrix C. Options:" + supPrec + ". " +
                "Defaults to the value of -r/--precision",
            cxxopts::value<std::string>()->default_value(""));
  opp_adder("d_type",
            "Precision of matrix D. Options:" + supPrec + ". " +
                "Defaults to the value of -r/--precision",
            cxxopts::value<std::string>()->default_value(""));
  opp_adder("compute_type",
            "What gemm kernel to use for the gemmEx family of functions"
            "Defaults to a value based on -r/--precision when not specified",
            cxxopts::value<std::string>()->default_value(""));
  opp_adder("scalar_type",
            "What scalar type to use "
            "Defaults to a value based on -r/--precision when not specified",
            cxxopts::value<std::string>()->default_value(""));
  opp_adder("batch_count",
            "Number of matrices. Only applicable to batched and "
            "strided_batched routines  (Default value is: 1)",
            cxxopts::value<int>()->default_value("1"));
  opp_adder("device", "GPU device(s) to run on",
            cxxopts::value<int>()->default_value("0"));
  opp_adder("instances", "Number of instances to run on each GPU",
            cxxopts::value<int>()->default_value("1"));
  opp_adder("i,iters",
            "Iterations to run inside timing loop  (Default value is: 10)",
            cxxopts::value<int>()->default_value("10"));
  opp_adder("j,cold_iters",
            " Cold Iterations to run before entering the timing loop ",
            cxxopts::value<int>()->default_value("2"));
  opp_adder("h,help", "Print Usage");

  // ParseResultE asdf;
  cxxopts::ParseResult result = options.parse(argc, argv);

  if (result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  genericGemm *gemm = new cublasGemm(result);
  gemm->prepareArray();
  // gemm->allocHost();
  // gemm->allocDev();
  // gemm->fillHost();
  // gemm->copyHostToDev();

  float gflops = gemm->test();
  std::cout << std::fixed;
  std::cout << "Gflops: " << gflops << std::endl;

  gemm->freeMem();

  return 0;
}
