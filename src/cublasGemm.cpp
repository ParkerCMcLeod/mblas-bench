#include "cublasGemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <regex>
#include <string>

#include "create-allocate.h"
#include "cublasConvert.h"
#include "cudaError.h"
#include "third_party/cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

// clang-format off
std::vector<gemmPrecType> cublasGemm::gemmExSupported = {
    // Compute type                 Scale Type    A/B Type      C Type
    {CUBLAS_COMPUTE_16F,            CUDA_R_16F,   CUDA_R_16F,   CUDA_R_16F  },
    {CUBLAS_COMPUTE_16F_PEDANTIC,   CUDA_R_16F,   CUDA_R_16F,   CUDA_R_16F  },
    {CUBLAS_COMPUTE_32I,            CUDA_R_32I,   CUDA_R_8I,    CUDA_R_32I  },
    {CUBLAS_COMPUTE_32I_PEDANTIC,   CUDA_R_32I,   CUDA_R_8I,    CUDA_R_32I  },
    // Compute type                 Scale Type    A/B Type      C Type
    {CUBLAS_COMPUTE_32F,            CUDA_R_32F,   CUDA_R_16BF,  CUDA_R_16BF },
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_R_32F,   CUDA_R_16BF,  CUDA_R_16BF },
    {CUBLAS_COMPUTE_32F,            CUDA_R_32F,   CUDA_R_16F,   CUDA_R_16F  },
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_R_32F,   CUDA_R_16F,   CUDA_R_16F  },
    {CUBLAS_COMPUTE_32F,            CUDA_R_32F,   CUDA_R_8I,    CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_R_32F,   CUDA_R_8I,    CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F,            CUDA_R_32F,   CUDA_R_16BF,  CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_R_32F,   CUDA_R_16BF,  CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F,            CUDA_R_32F,   CUDA_R_16F,   CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_R_32F,   CUDA_R_16F,   CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F,            CUDA_R_32F,   CUDA_R_32F,   CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_R_32F,   CUDA_R_32F,   CUDA_R_32F  },
    // Compute type                 Scale Type    A/B Type      C Type
    {CUBLAS_COMPUTE_32F,            CUDA_C_32F,   CUDA_C_8I,    CUDA_C_32F  },
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_C_32F,   CUDA_C_8I,    CUDA_C_32F  },
    {CUBLAS_COMPUTE_32F,            CUDA_C_32F,   CUDA_C_32F,   CUDA_C_32F  },
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_C_32F,   CUDA_C_32F,   CUDA_C_32F  },
    // Compute type                 Scale Type    A/B Type      C Type
    {CUBLAS_COMPUTE_32F_FAST_16F,   CUDA_R_32F,   CUDA_R_32F,   CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F_FAST_16BF,  CUDA_R_32F,   CUDA_R_32F,   CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F_FAST_TF32,  CUDA_R_32F,   CUDA_R_32F,   CUDA_R_32F  },
    {CUBLAS_COMPUTE_32F_FAST_16F,   CUDA_C_32F,   CUDA_C_32F,   CUDA_C_32F  },
    {CUBLAS_COMPUTE_32F_FAST_16BF,  CUDA_C_32F,   CUDA_C_32F,   CUDA_C_32F  },
    {CUBLAS_COMPUTE_32F_FAST_TF32,  CUDA_C_32F,   CUDA_C_32F,   CUDA_C_32F  },
    // Compute type                 Scale Type    A/B Type      C Type
    {CUBLAS_COMPUTE_64F,            CUDA_R_64F,   CUDA_R_64F,   CUDA_R_64F  },
    {CUBLAS_COMPUTE_64F_PEDANTIC,   CUDA_R_64F,   CUDA_R_64F,   CUDA_R_64F  },
    {CUBLAS_COMPUTE_64F,            CUDA_C_64F,   CUDA_C_64F,   CUDA_C_64F  },
    {CUBLAS_COMPUTE_64F_PEDANTIC,   CUDA_C_64F,   CUDA_C_64F,   CUDA_C_64F  },
};
// clang-format on

void cublasGemm::initPrecMap() {
  precDType = {
      {"h", CUDA_R_16F},       {"s", CUDA_R_32F},     {"d", CUDA_R_64F},
      {"c", CUDA_C_32F},       {"z", CUDA_C_64F},     {"f16_r", CUDA_R_16F},
      {"f16_c", CUDA_C_16F},   {"f32_r", CUDA_R_32F}, {"f32_c", CUDA_C_32F},
      {"f64_r", CUDA_R_64F},   {"f64_c", CUDA_C_64F}, {"bf16_r", CUDA_R_16BF},
      {"bf16_c", CUDA_C_16BF}, {"i8_r", CUDA_R_8I},   {"i8_c", CUDA_C_8I},
      {"i32_r", CUDA_R_32I},   {"i32_c", CUDA_C_32I},
  };
  computeDType = {
      {"CUBLAS_COMPUTE_16F", CUBLAS_COMPUTE_16F},
      {"CUBLAS_COMPUTE_16F_PEDANTIC", CUBLAS_COMPUTE_16F_PEDANTIC},
      {"CUBLAS_COMPUTE_32F", CUBLAS_COMPUTE_32F},
      {"CUBLAS_COMPUTE_32F_PEDANTIC", CUBLAS_COMPUTE_32F_PEDANTIC},
      {"CUBLAS_COMPUTE_32F_FAST_16F", CUBLAS_COMPUTE_32F_FAST_16F},
      {"CUBLAS_COMPUTE_32F_FAST_16BF", CUBLAS_COMPUTE_32F_FAST_16BF},
      {"CUBLAS_COMPUTE_32F_FAST_TF32", CUBLAS_COMPUTE_32F_FAST_TF32},
      {"CUBLAS_COMPUTE_64F", CUBLAS_COMPUTE_64F},
      {"CUBLAS_COMPUTE_64F_PEDANTIC", CUBLAS_COMPUTE_64F_PEDANTIC},
      {"CUBLAS_COMPUTE_32I", CUBLAS_COMPUTE_32I},
      {"CUBLAS_COMPUTE_32I_PEDANTIC", CUBLAS_COMPUTE_32I_PEDANTIC},
  };
}

cudaDataType_t cublasGemm::precisionStringToDType(std::string stringPrecision) {
  try {
    return precDType.at(stringPrecision);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << stringPrecision << std::endl;
    throw e;
    return CUDA_R_32F;
  }
}

void cublasGemm::selectCompute(std::string computestr) {
  if (computestr == "" || computestr == "PEDANTIC") {
    // If the user doesnt specify, just guess based on precision
    switch (precision) {
      case CUDA_R_64F:
        compute = CUBLAS_COMPUTE_64F;
        break;
      case CUDA_C_64F:
        compute = CUBLAS_COMPUTE_64F;
        break;
      case CUDA_R_32F:
        compute = CUBLAS_COMPUTE_32F;
        break;
      case CUDA_C_32F:
        compute = CUBLAS_COMPUTE_32F;
        break;
      case CUDA_R_32I:
        compute = CUBLAS_COMPUTE_32I;
        break;
      case CUDA_R_16F:
        compute = CUBLAS_COMPUTE_16F;
        break;
      case CUDA_C_16F:
        compute = CUBLAS_COMPUTE_16F;
        break;
      default:
        compute = CUBLAS_COMPUTE_32F;
        break;
    }
    if (computestr == "PEDANTIC") {
      // Borderline insane statement to enable the user to select pedantic
      // version without specifying compute type directly
      compute = static_cast<cublasComputeType_t>(static_cast<int>(compute) + 1);
    }

    return;
  }
  try {
    compute = computeDType.at(computestr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << computestr << std::endl;
    throw e;
    compute = CUBLAS_COMPUTE_32F;
  }
}

void cublasGemm::selectScalar(std::string scalarstr) {
  if (scalarstr == "") {
    // Scalar type not specified, setting to precision
    scalar = precision;
    return;
  }
  scalar = precisionStringToDType(scalarstr);
}

void cublasGemm::parseDevIters(std::string deviceStr, std::string instanceStr) {
  // Parse iters
  int iters = stoi(instanceStr);
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    std::cout << deviceSStr << std::endl;
    for (int i = 0; i < iters; i++) {
      gemmInst val = gemmInst(stoi(deviceSStr), i);
      matPtrs.push_back(val);
    }
  }
  std::cout << matPtrs.size() << std::endl;
}

void cublasGemm::parseMType(string computeTStr, string scalarTStr, string aStr,
                            string bStr, string cStr) {
  selectCompute(computeTStr);
  selectScalar(scalarTStr);
  if (aStr == "" || bStr == "" || cStr == "") {
    // Precision not completely specified, default to precision
    // cerr << "Precision incorrectly specified, setting precision to "
    //         "-r/--precision"
    //      << endl;
    a_type = precision;
    b_type = precision;
    c_type = precision;
    return;
  }
  // Parse each precision
  a_type = precisionStringToDType(aStr);
  b_type = precisionStringToDType(bStr);
  c_type = precisionStringToDType(cStr);

  // Validate against supported precision table (fun)
  if (a_type != b_type) {
    string errorString = "A Type must the same as B Type";
    throw std::invalid_argument(errorString);
  }
  if (function.find("GemmEx")) {
    /*
      Possible functions:
        cublasGemmEx
        cublasGemmExBatched
        cublasGemmExStridedBatched
    */
    gemmPrecType selType = {compute, scalar, a_type, c_type};
    auto result =
        std::find(begin(gemmExSupported), end(gemmExSupported), selType);
    if (result == end(gemmExSupported)) {
      // Unable to find matching config, not supported
      string errorString =
          "Invalid GEMM specification for GemmEx.  Combination of parameters "
          "not supported"
          "\nCompute type: " +
          computeTStr + "\nScalar type: " + scalarTStr + "\nA type: " + aStr +
          "\nB type: " + bStr + "\nC type: " + cStr;
      throw std::invalid_argument(errorString);
    }
  }
}

cublasGemm::cublasGemm(cxxopts::ParseResult result) : genericGemm(result) {
  cublasCreate(&handle);
  checkCublas(cublasCreate(&handle));
  initPrecMap();
  // Grab precision from command line
  precision = precisionStringToDType(result["precision"].as<string>());
  // Grab compute type from command line
  string computeT = result["compute_type"].as<string>();
  string scalarT = result["scalar_type"].as<string>();
  string aT = result["a_type"].as<string>();
  string bT = result["b_type"].as<string>();
  string cT = result["c_type"].as<string>();
  parseMType(computeT, scalarT, aT, bT, cT);

  parseDevIters(result["device"].as<string>(),
                result["instances"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = setOp(result["transposeA"].as<std::string>());
  transB = setOp(result["transposeB"].as<std::string>());

  // Pull in alpha and beta, alloc memory and save to pointers
  string salpha = result["alpha"].as<string>();
  string salphai = result["alphai"].as<string>();
  alpha =
      typeCallHost<allocSetScalar>(precision, salpha.c_str(), salphai.c_str());
  string sbeta = result["beta"].as<string>();
  string sbetai = result["betai"].as<string>();
  beta = typeCallHost<allocSetScalar>(precision, sbeta.c_str(), sbetai.c_str());
  // std::cout << *((float *)alpha) << std::endl;
  // std::cout << *((float *)beta) << std::endl;
}

cublasOperation_t cublasGemm::setOp(std::string str) {
  if (str.size() < 1) {
    str = "N";
  }
  switch (str[0]) {
    case 'N':
      return CUBLAS_OP_N;
      break;
    case 'T':
      return CUBLAS_OP_T;
      break;
    case 'C':
      return CUBLAS_OP_C;
      break;
  }
  return CUBLAS_OP_N;
}

void cublasGemm::prepareArray() {
  this->allocHost();
  this->allocDev();
  this->fillHost();
  this->copyHostToDev();
}

void cublasGemm::allocHost() {
  hostA = allocateHostArr(a_type, m, k, batchct);
  hostB = allocateHostArr(b_type, k, n, batchct);
  hostC = allocateHostArr(c_type, n, m, batchct);
}

void cublasGemm::allocDev() {
  devA = allocateDevArr(a_type, m, k, batchct);
  devB = allocateDevArr(b_type, k, n, batchct);
  devC = allocateDevArr(c_type, n, m, batchct);
}

void cublasGemm::fillHost() {
  typeCallHost<fillRandHost>(a_type, hostA, m, k, batchct);
  typeCallHost<fillRandHost>(b_type, hostB, k, n, batchct);
  typeCallHost<fillRandHost>(c_type, hostC, n, m, batchct);
}

void cublasGemm::copyHostToDev() {
  // int hostsz = typeCallHost<sizeofCUDT>(precision);
  // int devsz = typeCallDev<sizeofCUDT>(precision);
  // if (hostsz == devsz) {
  //   checkCuda(cudaMemcpy(devA, hostA, batchct * m * k * devsz,
  //                        cudaMemcpyHostToDevice));
  //   checkCuda(cudaMemcpy(devB, hostB, batchct * k * n * devsz,
  //                        cudaMemcpyHostToDevice));
  //   checkCuda(cudaMemcpy(devC, hostC, batchct * n * m * devsz,
  //                        cudaMemcpyHostToDevice));
  // } else {
  //   std::cout << "Mismatch, copy time" << std::endl;
  //   copyAndConvert(a_type, hostA, devA, m, k, batchct);
  //   copyAndConvert(b_type, hostB, devB, k, n, batchct);
  //   copyAndConvert(c_type, hostC, devC, n, m, batchct);
  // }
  copyAndConvert(a_type, hostA, devA, m, k, batchct);
  copyAndConvert(b_type, hostB, devB, k, n, batchct);
  copyAndConvert(c_type, hostC, devC, n, m, batchct);
  convertScalar(scalar, alpha);
  convertScalar(scalar, beta);
  if (batched && !strided) {
    // Perform some pointer arithmetic to calculate the arrays we pass to the
    // gpu
    ptrHostA = (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(a_type));
    ptrHostB = (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(b_type));
    ptrHostC = (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(c_type));
    checkCuda(
        cudaMalloc(&ptrDevA, batchct * typeCallHost<sizeofCUDTP>(a_type)));
    checkCuda(
        cudaMalloc(&ptrDevB, batchct * typeCallHost<sizeofCUDTP>(b_type)));
    checkCuda(
        cudaMalloc(&ptrDevC, batchct * typeCallHost<sizeofCUDTP>(c_type)));
    typeCallDev<batchedPtrMagic>(a_type, ptrHostA, ptrDevA, devA, batchct, m,
                                 k);
    typeCallDev<batchedPtrMagic>(b_type, ptrHostB, ptrDevB, devB, batchct, k,
                                 n);
    typeCallDev<batchedPtrMagic>(c_type, ptrHostC, ptrDevC, devC, batchct, n,
                                 m);
  }
}

void cublasGemm::freeMem() {
  cudaFree(alpha);
  cudaFree(beta);
  cudaFree(hostA);
  cudaFree(hostB);
  cudaFree(hostC);
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
}

double cublasGemm::test() {
  // <t>gemm implementation
  if (function == "cublasDgemm" && precision == CUDA_R_64F) {
    std::function<decltype(cublasDgemm)> dgemm_var = cublasDgemm;
    return testTGemm<double>(dgemm_var);
  } else if (function == "cublasSgemm" && precision == CUDA_R_32F) {
    std::function<decltype(cublasSgemm)> sgemm_var = cublasSgemm;
    return testTGemm<float>(sgemm_var);
  } else if (function == "cublasHgemm" && precision == CUDA_R_16F) {
    std::function<decltype(cublasHgemm)> hgemm_var = cublasHgemm;
    return testTGemm<__half>(hgemm_var);
  } else if (function == "cublasZgemm" && precision == CUDA_C_64F) {
    std::function<decltype(cublasZgemm)> zgemm_var = cublasZgemm;
    return testTGemm<cuDoubleComplex>(zgemm_var);
  } else if (function == "cublasCgemm" && precision == CUDA_C_32F) {
    std::function<decltype(cublasCgemm)> cgemm_var = cublasCgemm;
    return testTGemm<cuComplex>(cgemm_var);
  } else if (function == "cublasZgemm3m" && precision == CUDA_C_64F) {
    std::function<decltype(cublasZgemm3m)> zgemm3m_var = cublasZgemm3m;
    return testTGemm<cuDoubleComplex>(zgemm3m_var);
  } else if (function == "cublasCgemm3m" && precision == CUDA_C_32F) {
    std::function<decltype(cublasCgemm3m)> cgemm3m_var = cublasCgemm3m;
    return testTGemm<cuComplex>(cgemm3m_var);
  } else if (function == "cublasDgemmBatched" && precision == CUDA_R_64F) {
    std::function<decltype(cublasDgemmBatched)> dgemm_var = cublasDgemmBatched;
    return testTGemmBatched<double>(dgemm_var);
  } else if (function == "cublasSgemmBatched" && precision == CUDA_R_32F) {
    std::function<decltype(cublasSgemmBatched)> sgemm_var = cublasSgemmBatched;
    return testTGemmBatched<float>(sgemm_var);
  } else if (function == "cublasHgemmBatched" && precision == CUDA_R_16F) {
    std::function<decltype(cublasHgemmBatched)> hgemm_var = cublasHgemmBatched;
    return testTGemmBatched<__half>(hgemm_var);
  } else if (function == "cublasZgemmBatched" && precision == CUDA_C_64F) {
    std::function<decltype(cublasZgemmBatched)> zgemm_var = cublasZgemmBatched;
    return testTGemmBatched<cuDoubleComplex>(zgemm_var);
  } else if (function == "cublasCgemmBatched" && precision == CUDA_C_32F) {
    std::function<decltype(cublasCgemmBatched)> cgemm_var = cublasCgemmBatched;
    return testTGemmBatched<cuComplex>(cgemm_var);
  }
  if (function == "cublasDgemmStridedBatched" && precision == CUDA_R_64F) {
    std::function<decltype(cublasDgemmStridedBatched)> dgemm_var =
        cublasDgemmStridedBatched;
    return testTGemmStridedBatched<double>(dgemm_var);
  } else if (function == "cublasSgemmStridedBatched" &&
             precision == CUDA_R_32F) {
    std::function<decltype(cublasSgemmStridedBatched)> sgemm_var =
        cublasSgemmStridedBatched;
    return testTGemmStridedBatched<float>(sgemm_var);
  } else if (function == "cublasHgemmStridedBatched" &&
             precision == CUDA_R_16F) {
    std::function<decltype(cublasHgemmStridedBatched)> hgemm_var =
        cublasHgemmStridedBatched;
    return testTGemmStridedBatched<__half>(hgemm_var);
  } else if (function == "cublasZgemmStridedBatched" &&
             precision == CUDA_C_64F) {
    std::function<decltype(cublasZgemmStridedBatched)> zgemm_var =
        cublasZgemmStridedBatched;
    return testTGemmStridedBatched<cuDoubleComplex>(zgemm_var);
  } else if (function == "cublasCgemmStridedBatched" &&
             precision == CUDA_C_32F) {
    std::function<decltype(cublasCgemmStridedBatched)> cgemm_var =
        cublasCgemmStridedBatched;
    return testTGemmStridedBatched<cuComplex>(cgemm_var);
  } else if (function == "cublasCgemm3mStridedBatched" &&
             precision == CUDA_C_32F) {
    std::function<decltype(cublasCgemm3mStridedBatched)> cgemm_var =
        cublasCgemm3mStridedBatched;
    return testTGemmStridedBatched<cuComplex>(cgemm_var);
  }

  if (strided && function == "cublasGemmExStridedBatched") {
    // Call the Gemm strided batched deployment script
  } else if (batched && function == "cublasGemmExBatched") {
    // Call the Gemm batched code
  } else if (batched && function == "cublasGemmEx") {
    return testGemmEx();
  }
  std::cerr << "Invalid implementation & precision combination" << std::endl;
  exit(1);
}

// template <typename T, typename F>

// template <typename T, template <typename> typename tFunc>
// double cublasGemm::testTGemm() {
// template <typename T>
// double testTGemm(
//    function<cublasStatus_t(
//        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
//        const T *, const T *, int, const T *, int, const T *, T *, int)>
//        func) {
//  std::cout << "test" << std::endl;
//}

template <typename T>
double cublasGemm::testTGemm(
    std::function<cublasStatus_t(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
        const T *, const T *, int, const T *, int, const T *, T *, int)>
        func) {
  // std::cout << "Alpha: " << *((T *)alpha) << std::endl;
  // std::cout << "Beta: " << *((T *)beta) << std::endl;
  // std::cout << "test" << std::endl;
  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(devA);
  T *devBP = static_cast<T *>(devB);
  T *devCP = static_cast<T *>(devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc);

    checkCublas(stat);

    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double totalTime_ms = 0.0;
  for (int rep = 0; rep < iters; rep++) {
    cudaEventRecord(start, 0);
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkCublas(stat);
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }

    float elapsedTime_ms;

    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    totalTime_ms += static_cast<double>(elapsedTime_ms);
  }
  // for (int i = 0; i < 10; i++) {
  //  std::cout << devCP
  //}
  std::cout << totalTime_ms << std::endl;
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;
  double totalSize;

  totalSize = batchct * static_cast<double>(m) * static_cast<double>(n) *
              static_cast<double>(k);

  double gflop = totalSize * 2.0f / 1e9;
  double gflopPerSec = gflop / avgTime_s;

  return gflopPerSec;
}

template <typename T>
double cublasGemm::testTGemmBatched(
    std::function<cublasStatus_t(cublasContext *, cublasOperation_t,
                                 cublasOperation_t, int, int, int, T const *,
                                 T const *const *, int, T const *const *, int,
                                 T const *, T *const *, int, int)>
        func) {
  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T **devAP = reinterpret_cast<T **>(ptrDevA);
  T **devBP = reinterpret_cast<T **>(ptrDevB);
  T **devCP = reinterpret_cast<T **>(ptrDevC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc, batchct);

    checkCublas(stat);

    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double totalTime_ms = 0.0;
  for (int rep = 0; rep < iters; rep++) {
    cudaEventRecord(start, 0);
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc, batchct);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkCublas(stat);
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }

    float elapsedTime_ms;

    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    totalTime_ms += static_cast<double>(elapsedTime_ms);
  }
  // for (int i = 0; i < 10; i++) {
  //  std::cout << devCP
  //}
  std::cout << totalTime_ms << std::endl;
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;
  double totalSize;

  totalSize = batchct * static_cast<double>(m) * static_cast<double>(n) *
              static_cast<double>(k);

  double gflop = totalSize * 2.0f / 1e9;
  double gflopPerSec = gflop / avgTime_s;

  return gflopPerSec;
}

template <typename T>
double cublasGemm::testTGemmStridedBatched(
    std::function<cublasStatus_t(
        cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int,
        T const *, T const *, int, long long, T const *, int, long long,
        T const *, T *, int, long long, int)>
        func) {
  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(devA);
  T *devBP = static_cast<T *>(devB);
  T *devCP = static_cast<T *>(devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, stride_a,
                devBP, ldb, stride_b, betaP, devCP, ldc, stride_c, batchct);

    checkCublas(stat);

    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double totalTime_ms = 0.0;
  for (int rep = 0; rep < iters; rep++) {
    cudaEventRecord(start, 0);
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, stride_a,
                devBP, ldb, stride_b, betaP, devCP, ldc, stride_c, batchct);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkCublas(stat);

    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }

    float elapsedTime_ms;

    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    totalTime_ms += static_cast<double>(elapsedTime_ms);
  }
  // for (int i = 0; i < 10; i++) {
  //  std::cout << devCP
  //}
  std::cout << totalTime_ms << std::endl;
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;
  double totalSize;

  totalSize = batchct * static_cast<double>(m) * static_cast<double>(n) *
              static_cast<double>(k);

  double gflop = totalSize * 2.0f / 1e9;
  double gflopPerSec = gflop / avgTime_s;

  return gflopPerSec;
}

template <typename T>
double cublasGemm::testTGemmEx(
    std::function<cublasStatus_t(
        cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int,
        T const *, void const *, cudaDataType_t, int, void const *,
        cudaDataType_t, int, T const *, void *, cudaDataType_t, int)>
        func) {
  return 0.0;
}

double cublasGemm::testGemmEx() {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // float *alphaP = static_cast<float *>(alpha);
  // float *betaP = static_cast<float *>(beta);
  //  float alphaP = __half2float(*((__half *)alpha));
  //  float betaP = __half2float(*((__half *)beta));
  // __half alphaP = __float2half(*((float *)alpha));
  // __half betaP = __float2half(*((float *)beta));
  // float betaP = static_cast<float *>(beta);
  // std::cout << alphaP << " " << betaP << std::endl;
  double totalTime_ms = 0.0;
  for (int rep = 0; rep < iters; rep++) {
    cudaEventRecord(start, 0);
    stat = cublasGemmEx(handle, transA, transB, m, n, k, alpha, devA, a_type,
                        lda, devB, b_type, ldb, beta, devC, c_type, ldc,
                        compute, CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // std::cerr << cublasGetErrorString(stat) << std::endl;
    checkCublas(stat);
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }

    float elapsedTime_ms;

    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    totalTime_ms += static_cast<double>(elapsedTime_ms);
  }
  // for (int i = 0; i < 10; i++) {
  //  std::cout << devCP
  //}
  std::cout << totalTime_ms << std::endl;
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;
  double totalSize;

  totalSize = batchct * static_cast<double>(m) * static_cast<double>(n) *
              static_cast<double>(k);

  double gflop = totalSize * 2.0f / 1e9;
  double gflopPerSec = gflop / avgTime_s;

  return gflopPerSec;
}