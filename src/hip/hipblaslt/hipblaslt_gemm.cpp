#include "hipblaslt_gemm.h"

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <future>
#include <iomanip>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

#include "hip_convert.h"
#include "hip_create_allocate.h"
#include "hip_datatype_utils.h"
#include "hip_error.h"
#include "hip_timing.h"
#include "cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;
using namespace mblas_timing;

namespace {

void validate_gpu_capability(int device_id,
                              const mblas_data_type& a_type,
                              const mblas_data_type& b_type) {
  hipDeviceProp_t prop{};
  check_hip(hipGetDeviceProperties(&prop, device_id));

  int gfx = 0;
  std::sscanf(prop.gcnArchName, "gfx%d", &gfx);

  auto requires_gfx = [&](int req_gfx, const char* feature, const char* arch_name) {
    if (gfx >= req_gfx) return;
    throw std::runtime_error(
      std::string(feature) + " requires gfx" + std::to_string(req_gfx) +
      "+ (" + arch_name + "), but device " + std::to_string(device_id) +
      " (" + prop.name + ") is " + prop.gcnArchName);
  };

  for (const auto* type : {&a_type, &b_type}) {
    if (type->is_fp8())
      requires_gfx(942, "FP8", "MI300");
    if (type->is_fp4())
      requires_gfx(950, "FP4", "MI350");
    if (*type == mblas_data_type::MBLAS_R_6F_E2M3 ||
        *type == mblas_data_type::MBLAS_R_6F_E3M2)
      requires_gfx(950, "FP6", "MI350");
  }
}

}  // namespace

// clang-format off
std::vector<matmul_prec_type> hipblaslt_gemm::matmul_supported = {
  // Compute type                 Scale Type    A Type        B Type        C Type        D Type        Bias Type
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F_FAST_TF32,   MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32I,             MBLAS_R_32I,  MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_ANY},
};

std::vector<matmul_prec_type_f8> hipblaslt_gemm::matmulSupportedF8 = {
  // Scale Type  C Type           D Type            Bias Type
  {MBLAS_R_32F,  MBLAS_R_16F,     MBLAS_R_16F,      MBLAS_R_16F },
  {MBLAS_R_32F,  MBLAS_R_16BF,    MBLAS_R_16BF,     MBLAS_R_16BF},
  {MBLAS_R_32F,  MBLAS_R_32F,     MBLAS_R_32F,      MBLAS_R_16BF},
  {MBLAS_R_32F,  MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3,  MBLAS_R_16F },
  {MBLAS_R_32F,  MBLAS_R_8F_E5M2, MBLAS_R_8F_E5M2,  MBLAS_R_16F },
  // FP32 bias variants
  // Scale Type  C Type           D Type            Bias Type
  {MBLAS_R_32F,  MBLAS_R_16F,     MBLAS_R_16F,      MBLAS_R_32F },
  {MBLAS_R_32F,  MBLAS_R_16BF,    MBLAS_R_16BF,     MBLAS_R_32F },
  {MBLAS_R_32F,  MBLAS_R_32F,     MBLAS_R_32F,      MBLAS_R_32F },
  {MBLAS_R_32F,  MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3,  MBLAS_R_32F },
  {MBLAS_R_32F,  MBLAS_R_8F_E5M2, MBLAS_R_8F_E5M2,  MBLAS_R_32F },
};
// clang-format on

// parse_dev_iters: now inlined in header via parse_dev_iters_impl

void hipblaslt_gemm::parse_problem_type(string computeTStr, string scalarTStr,
                               string aStr, string bStr, string cStr,
                               string dStr) {
  compute.set_compute(computeTStr, precision);
  scalar.set_scalar(scalarTStr, precision, compute);
  bool noParse = false;
  if (aStr == "" || bStr == "" || cStr == "") {
    // Precision not completely specified, default to precision
    // cerr << "Precision incorrectly specified, setting precision to "
    //         "-r/--precision"
    //      << endl;
    a_type = precision;
    b_type = precision;
    c_type = precision;
    d_type = precision;
    inplace = true;
    noParse = true;
  }

  if (dStr == "") {
    // Assume the user means C = D, so also establish that here
    dStr = cStr;
    inplace = true;
  }

  // Parse each precision
  if (!noParse) {
    a_type = mblas_hip_data_type(aStr);
    b_type = mblas_hip_data_type(bStr);
    c_type = mblas_hip_data_type(cStr);
    d_type = mblas_hip_data_type(dStr);
  }
}

void hipblaslt_gemm::validate_parameters() {
  // Validate that data types exist in table of supported configurations
  matmul_prec_type selType = {
      compute, scalar, a_type, b_type, c_type, d_type, mblas_hip_data_type(MBLAS_ANY)};
  auto result =
      std::find(begin(matmul_supported), end(matmul_supported), selType);
  if (result != end(matmul_supported)) {
    return;
  } else if (compute == mblas_hipblas_compute_type::MBLAS_COMPUTE_32F && a_type.is_fp8() && b_type.is_fp8()) {
    // Special FP8 type filtering
    matmul_prec_type_f8 selTypeF8 = {
        scalar, c_type, d_type, mblas_hip_data_type(MBLAS_ANY)};
    auto result = std::find(begin(matmulSupportedF8), end(matmulSupportedF8), selTypeF8);
    if (result != end(matmulSupportedF8)) {
      return;
    }
  }
  // Unable to find matching config, not supported
  string errorString =
      "Invalid GEMM specification for MatMul.  Combination of parameters "
      "not supported"
      "\nCompute type: " +
      compute.to_string() + "\nScalar type: " + scalar.to_string() +
      "\nA type: " + a_type.to_string() +
      "\nB type: " + b_type.to_string() +
      "\nC type: " + c_type.to_string() +
      "\nD type: " + d_type.to_string();
  throw std::invalid_argument(errorString);
}

hipblaslt_gemm::hipblaslt_gemm(cxxopts::ParseResult result) : generic_gemm(result) {
  // Grab precision from command line
  precision = mblas_hip_data_type(result["precision"].as<string>());
  // Grab compute type from command line
  string computeT = result["compute_type"].as<string>();
  string scalarT = result["scalar_type"].as<string>();
  string aT = result["a_type"].as<string>();
  string bT = result["b_type"].as<string>();
  string cT = result["c_type"].as<string>();
  string dT = result["d_type"].as<string>();
  parse_problem_type(computeT, scalarT, aT, bT, cT, dT);

  parse_dev_iters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblas_hipblas_operation(result["transposeA"].as<std::string>());
  transB = mblas_hipblas_operation(result["transposeB"].as<std::string>());
  validate_parameters();

  // Pull in alpha and beta, alloc memory and save to pointers
  string salpha = result["alpha"].as<string>();
  string salphai = result["alphai"].as<string>();
  alpha = malloc(get_malloc_size_scalar(precision));
  type_call_host<set_scalar>(precision, alpha, salpha, salphai);
  
  string sbeta = result["beta"].as<string>();
  string sbetai = result["betai"].as<string>();
  beta = malloc(get_malloc_size_scalar(precision));
  type_call_host<set_scalar>(precision, beta, sbeta, sbetai);
  // std::cout << *((float *)alpha) << std::endl;
  // std::cout << *((float *)beta) << std::endl;
  set_flush_batch_count(
      {(uint64_t)a_props.rows_mem, (uint64_t)a_props.cols_mem, type_call_dev<sizeofCUDT>(a_type), a_type.get_packing_count()},
      {(uint64_t)b_props.rows_mem, (uint64_t)b_props.cols_mem, type_call_dev<sizeofCUDT>(b_type), b_type.get_packing_count()},
      {(uint64_t)c_props.rows_mem, (uint64_t)c_props.cols_mem, type_call_dev<sizeofCUDT>(c_type), c_type.get_packing_count()},
      {(uint64_t)d_props.rows_mem, (uint64_t)d_props.cols_mem, type_call_dev<sizeofCUDT>(d_type), d_type.get_packing_count()},
      inplace);
}

string hipblaslt_gemm::prepare_array() {
  alpha = convert_scalar(scalar, alpha);
  beta = convert_scalar(scalar, beta);
  this->alloc_host();
  this->fill_host();

  int num_devices;
  check_hip(hipGetDeviceCount(&num_devices));
  // Check range of devices here
  // This implementation may not work if
  // HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES is set to something weird
  for (auto &instance : mat_ptrs) {
    if (instance.devIDX >= num_devices) {
      string errorString =
          "Invalid device id"
          "\nNumber of detected devices: " +
          std::to_string(num_devices) +
          "\nDevice selection:           " + std::to_string(instance.devIDX);
      throw std::invalid_argument(errorString);
    }
    validate_gpu_capability(instance.devIDX, a_type, b_type);
  }
  // for (auto &instance : mat_ptrs) {
  //  this->alloc_dev(&instance);
  //  this->copy_host_to_dev(&instance);
  //}
  run_threaded(&hipblaslt_gemm::alloc_dev);
  run_threaded(&hipblaslt_gemm::copy_host_to_dev);
  run_threaded(&hipblaslt_gemm::prepare_matrix);
  // Enable tuning with a parameter later
  if (false) {
  } else {
    run_threaded(&hipblaslt_gemm::no_tuning);
  }
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "hipBLASLt-Gflops,hipBLASLt-GB/s,hipBLASLt-us," << endl;
  return ossHeader.str();
}

// run_threaded: now inlined in header via run_threaded_impl

void hipblaslt_gemm::alloc_host() {
  ptr_host_a =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(a_type));
  ptr_host_b =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(b_type));
  ptr_host_c =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(c_type));
  if (!inplace) {
    ptr_host_d =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(d_type));
  } else {
    ptr_host_d = ptr_host_c;
  }


  for (int i = 0; i < flush_batch_count; i++) {
    ptr_host_a[i] = malloc(get_malloc_size_host(a_type, a_props.rows_mem, a_props.cols_mem, batch_count, a_props.stride));
    ptr_host_b[i] = malloc(get_malloc_size_host(b_type, b_props.rows_mem, b_props.cols_mem, batch_count, b_props.stride));
    ptr_host_c[i] = malloc(get_malloc_size_host(c_type, c_props.rows_mem, c_props.cols_mem, batch_count, c_props.stride));
    if (!inplace) {
      ptr_host_d[i] = malloc(get_malloc_size_host(d_type, d_props.rows_mem, d_props.cols_mem, batch_count, d_props.stride));
    }
  }
}

void hipblaslt_gemm::alloc_dev(hipblaslt_gemm_inst *mat) {
  check_hip(hipSetDevice(mat->devIDX));

  mat->ptr_dev_a =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(a_type));
  mat->ptr_dev_b =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(b_type));
  mat->ptr_dev_c =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(c_type));
  if (!inplace) {
    mat->ptr_dev_d =
        (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(d_type));
  } else {
    mat->ptr_dev_d = mat->ptr_dev_c;
  }

  for (int i = 0; i < flush_batch_count; i++) {
    hipMalloc(&mat->ptr_dev_a[i], get_malloc_size_dev(a_type, a_props.rows_mem, a_props.cols_mem, batch_count, a_props.stride));
    hipMalloc(&mat->ptr_dev_b[i], get_malloc_size_dev(b_type, b_props.rows_mem, b_props.cols_mem, batch_count, b_props.stride));
    hipMalloc(&mat->ptr_dev_c[i], get_malloc_size_dev(c_type, c_props.rows_mem, c_props.cols_mem, batch_count, c_props.stride));
    if (!inplace) {
      hipMalloc(&mat->ptr_dev_d[i], get_malloc_size_dev(d_type, d_props.rows_mem, d_props.cols_mem, batch_count, d_props.stride));
    }
  }
  mat->wSZ = workspace_size;
  check_hip(hipMalloc(&mat->devWork, mat->wSZ));
}

void hipblaslt_gemm::fill_host() {
  type_call_host<initHost>(a_type, initialization, ptr_host_a, a_props.rows, a_props.cols, lda,
                         batch_count, a_props.stride, flush_batch_count, a_props.control, a_props.constant, filename_a);
  type_call_host<initHost>(b_type, initialization, ptr_host_b, b_props.rows, b_props.cols, ldb,
                         batch_count, b_props.stride, flush_batch_count, b_props.control, b_props.constant, filename_b);
  type_call_host<initHost>(c_type, initialization, ptr_host_c, c_props.rows, c_props.cols, ldc,
                         batch_count, c_props.stride, flush_batch_count, c_props.control, c_props.constant, filename_c);
}

void hipblaslt_gemm::copy_host_to_dev(hipblaslt_gemm_inst *mat) {
  check_hip(hipSetDevice(mat->devIDX));
  for (int i = 0; i < flush_batch_count; i++) {
    copy_and_convert(a_type, ptr_host_a[i], mat->ptr_dev_a[i], a_props.rows_mem, a_props.cols_mem, batch_count, a_props.stride);
    copy_and_convert(b_type, ptr_host_b[i], mat->ptr_dev_b[i], b_props.rows_mem, b_props.cols_mem, batch_count, b_props.stride);
    copy_and_convert(c_type, ptr_host_c[i], mat->ptr_dev_c[i], c_props.rows_mem, c_props.cols_mem, batch_count, c_props.stride);
  }
}

void hipblaslt_gemm::prepare_matrix(hipblaslt_gemm_inst *mat) {
  check_hipblas(hipblasLtMatmulDescCreate(&mat->desc_op, compute, scalar));
  // These values are read in with no type, so they need to be convirted first
  // Thanks for the wonderful standard Nvidia :D!
  hipblasOperation_t transA_local = transA.convert_to_hip();
  hipblasOperation_t transB_local = transB.convert_to_hip();
  check_hipblas(hipblasLtMatmulDescSetAttribute(
      mat->desc_op, HIPBLASLT_MATMUL_DESC_TRANSA, &transA_local, sizeof(transA_local)));
  check_hipblas(hipblasLtMatmulDescSetAttribute(
      mat->desc_op, HIPBLASLT_MATMUL_DESC_TRANSB, &transB_local, sizeof(transB_local)));

  check_hipblas(
      hipblasLtMatrixLayoutCreate(&mat->desc_a, a_type, a_props.rows, a_props.cols, lda));
  check_hipblas(
      hipblasLtMatrixLayoutCreate(&mat->desc_b, b_type, b_props.rows, b_props.cols, ldb));
  check_hipblas(
      hipblasLtMatrixLayoutCreate(&mat->desc_c, c_type, c_props.rows, c_props.cols, ldc));
  if (!inplace) {
    check_hipblas(
        hipblasLtMatrixLayoutCreate(&mat->desc_d, d_type, d_props.rows, d_props.cols, ldd));
  } else {
    mat->desc_d = mat->desc_c;
  }
  if (batch_count > 1) {
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_d, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_props.stride, sizeof(a_props.stride)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_props.stride, sizeof(b_props.stride)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_c, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_props.stride, sizeof(c_props.stride)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_d, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &d_props.stride, sizeof(d_props.stride)));
  }

  check_hipblas(hipblasLtMatmulPreferenceCreate(&mat->pref));
  check_hipblas(hipblasLtMatmulPreferenceSetAttribute(
      mat->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mat->wSZ,
      sizeof(mat->wSZ)));
}

void hipblaslt_gemm::no_tuning(hipblaslt_gemm_inst *mat) {
  hipblasStatus_t stat;
  hipblasLtHandle_t handle;
  check_hip(hipSetDevice(mat->devIDX));
  check_hipblas(hipblasLtCreate(&handle));
  int retResults = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult = {0};

  check_hipblas(hipblasLtMatmulAlgoGetHeuristic(
      handle, mat->desc_op, mat->desc_a, mat->desc_b, mat->desc_c, mat->desc_d,
      mat->pref, 1, &heuristicResult, &retResults));

  if (retResults == 0) {
    throw std::runtime_error("hipblasLtMatmulAlgoGetHeuristic returned 0 results: no supported algorithm for this configuration");
  }
  mat->algo = heuristicResult;
  check_hipblas(hipblasLtDestroy(handle));
}
void hipblaslt_gemm::auto_tuning(hipblaslt_gemm_inst *mat) {
  // Not currently implemented, using simple method
  no_tuning(mat);
}

void hipblaslt_gemm::free_mem() {
  free(alpha);
  free(beta);
  for (int i = 0; i < flush_batch_count; i++) {
    free(ptr_host_a[i]);
    free(ptr_host_b[i]);
    free(ptr_host_c[i]);
    if (!inplace) {
      free(ptr_host_d[i]);
    }
  }
  free(ptr_host_a);
  free(ptr_host_b);
  free(ptr_host_c);
  if (!inplace) {
    free(ptr_host_d);
  }
  for (auto mat : mat_ptrs) {
    check_hip(hipSetDevice(mat.devIDX));
    for (int i = 0; i < flush_batch_count; i++) {
      check_hip(hipFree(mat.ptr_dev_a[i]));
      check_hip(hipFree(mat.ptr_dev_b[i]));
      check_hip(hipFree(mat.ptr_dev_c[i]));
      if (!inplace) {
        check_hip(hipFree(mat.ptr_dev_d[i]));
      }
    }
    free(mat.ptr_dev_a);
    free(mat.ptr_dev_b);
    free(mat.ptr_dev_c);
    if (!inplace) {
      free(mat.ptr_dev_d);
    }
    check_hip(hipFree(mat.devWork));
    check_hipblas(hipblasLtMatmulDescDestroy(mat.desc_op));
    check_hipblas(hipblasLtMatrixLayoutDestroy(mat.desc_a));
    check_hipblas(hipblasLtMatrixLayoutDestroy(mat.desc_b));
    check_hipblas(hipblasLtMatrixLayoutDestroy(mat.desc_c));
    if (!inplace) {
      check_hipblas(hipblasLtMatrixLayoutDestroy(mat.desc_d));
    }
    check_hipblas(hipblasLtMatmulPreferenceDestroy(mat.pref));
  }
}

double hipblaslt_gemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : mat_ptrs) {
    threads.push_back(thread(&hipblaslt_gemm::test_matmul, this, &mat));
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Accumulate results from all device instances
  accumulate_results(mat_ptrs);

  return gflop_per_second;
}

std::string hipblaslt_gemm::get_result_string() {
  std::ostringstream ossValues;
  ossValues << std::setprecision(7);
  ossValues << transA.to_string_short() << ',' << transB.to_string_short() << ',' << m
            << ',' << n << ',' << k << ',' << lda << ',' << ldb << ',' << ldc
            << ',';
  if (batched) {
    ossValues << batch_count << ',';
  }
  ossValues << gflop_per_second << ',';
  ossValues << gbyte_per_second << ',';
  ossValues << iter_time_us << ',';
  ossValues << endl;
  return ossValues.str();
}

void hipblaslt_gemm::test_matmul(hipblaslt_gemm_inst *mat) {
  hipblasStatus_t stat;
  hipblasLtHandle_t handle;
  hipStream_t stream;
  check_hip(hipSetDevice(mat->devIDX));
  check_hipblas(hipblasLtCreate(&handle));
  check_hip(hipStreamCreate(&stream));
  auto run_kernel = [&](int rep) {
    int flush_index = rep % flush_batch_count;
    stat = hipblasLtMatmul(handle, mat->desc_op, alpha, mat->ptr_dev_a[flush_index], mat->desc_a,
                          mat->ptr_dev_b[flush_index], mat->desc_b, beta, mat->ptr_dev_c[flush_index], mat->desc_c,
                          mat->ptr_dev_d[flush_index], mat->desc_d, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
  };
  auto cold_kernel = [&](int rep) {
    run_kernel(rep);
    check_hipblas(stat);
    check_hip(hipGetLastError());
  };

  float elapsedTime_ms = gpu_timed_run<HipTimingTraits>(stream, cold_iters, iters, cold_kernel, run_kernel);
  check_hipblas(stat);
  check_hip(hipGetLastError());
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms), iters,
          type_call_dev<sizeofCUDT>(a_type), type_call_dev<sizeofCUDT>(b_type),
          type_call_dev<sizeofCUDT>(d_type),
          a_type.get_packing_count(), b_type.get_packing_count(),
          d_type.get_packing_count(), precision.is_real());

  check_hip(hipStreamDestroy(stream));
  check_hipblas(hipblasLtDestroy(handle));
}
