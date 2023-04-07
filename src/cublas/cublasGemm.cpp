#include "cublasGemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include "cublasConvert.h"
#include "cublasCreateAllocate.h"
#include "cudaError.h"
#include "third_party/cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

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
    {CUBLAS_COMPUTE_32F_PEDANTIC,   CUDA_R_32F,   CUDA_R_16F,   CUDA_R_16F  }, {CUBLAS_COMPUTE_32F,            CUDA_R_32F,   CUDA_R_8I,    CUDA_R_32F  },
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

std::vector<TgemmPrecType> cublasGemm::TgemmExSupported = {
    {CUDA_R_16BF, CUDA_R_16BF}, {CUDA_R_16F, CUDA_R_16F},
    {CUDA_R_8I, CUDA_R_32F},    {CUDA_R_16BF, CUDA_R_32F},
    {CUDA_R_16F, CUDA_R_32F},   {CUDA_R_32F, CUDA_R_32F},
    {CUDA_C_8I, CUDA_C_32F},    {CUDA_C_32F, CUDA_C_32F},

};

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

void cublasGemm::parseDevIters(std::string deviceStr, int instance) {
  // Parse iters
  int iters = instance;
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    ThreadBarrier *devSync = new ThreadBarrier(iters);

    for (int i = 0; i < iters; i++) {
      gemmInst val = gemmInst(devInt, i);
      val.devSync = devSync;
      matPtrs.push_back(val);
    }
  }
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
  } else if (function.find("gemmEx")) {
    TgemmPrecType selType = {a_type, c_type};
    auto result =
        std::find(begin(TgemmExSupported), end(TgemmExSupported), selType);
    if (result == end(TgemmExSupported)) {
      // Unable to find matching config, not supported
      string errorString =
          "Invalid GEMM specification for GemmEx.  Combination of parameters "
          "not supported"
          "\nA type: " +
          aStr + "\nB type: " + bStr + "\nC type: " + cStr;
      throw std::invalid_argument(errorString);
    }
  }
}

cublasGemm::cublasGemm(cxxopts::ParseResult result) : genericGemm(result) {
  // cublasCreate(&handle);
  // checkCublas(cublasCreate(&handle));
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

  parseDevIters(result["device"].as<string>(), result["instances"].as<int>());
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
  convertScalar(scalar, alpha);
  convertScalar(scalar, beta);
  this->allocHost();
  this->fillHost();

  int num_devices;
  cudaGetDeviceCount(&num_devices);
  // Check range of devices here
  // This implementation may not work if
  // CUDA_VISIBLE_DEVICES is set to something weird
  for (auto &instance : matPtrs) {
    if (instance.devIDX >= num_devices) {
      string errorString =
          "Invalid device id"
          "\nNumber of detected devices: " +
          std::to_string(num_devices) +
          "\nDevice selection:           " + std::to_string(instance.devIDX);
      throw std::invalid_argument(errorString);
    }
  }
  // for (auto &instance : matPtrs) {
  //  this->allocDev(&instance);
  //  this->copyHostToDev(&instance);
  //}
  vector<thread> threads;
  for (auto &instance : matPtrs) {
    threads.push_back(thread(&cublasGemm::allocDev, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
  threads.clear();
  for (auto &instance : matPtrs) {
    threads.push_back(thread(&cublasGemm::copyHostToDev, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void cublasGemm::allocHost() {
  hostA = allocateHostArr(a_type, m, k, batchct);
  hostB = allocateHostArr(b_type, k, n, batchct);
  hostC = allocateHostArr(c_type, n, m, batchct);
}

void cublasGemm::allocDev(gemmInst *mat) {
  cudaSetDevice(mat->devIDX);
  mat->devA = allocateDevArr(a_type, m, k, batchct);
  mat->devB = allocateDevArr(b_type, k, n, batchct);
  mat->devC = allocateDevArr(c_type, n, m, batchct);
  mat->wSZ = 4 * 1024 * 1024;
  cudaMalloc(&mat->devWork, mat->wSZ);
}

void cublasGemm::fillHost() {
  typeCallHost<fillRandHost>(a_type, hostA, m, k, batchct);
  typeCallHost<fillRandHost>(b_type, hostB, k, n, batchct);
  typeCallHost<fillRandHost>(c_type, hostC, n, m, batchct);
}

void cublasGemm::copyHostToDev(gemmInst *mat) {
  cudaSetDevice(mat->devIDX);
  copyAndConvert(a_type, hostA, mat->devA, m, k, batchct);
  copyAndConvert(b_type, hostB, mat->devB, k, n, batchct);
  copyAndConvert(c_type, hostC, mat->devC, n, m, batchct);
  if (batched && !strided) {
    // Perform some pointer arithmetic to calculate the arrays we pass to the
    // gpu
    mat->ptrHostA =
        (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(a_type));
    mat->ptrHostB =
        (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(b_type));
    mat->ptrHostC =
        (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(c_type));
    checkCuda(
        cudaMalloc(&mat->ptrDevA, batchct * typeCallHost<sizeofCUDTP>(a_type)));
    checkCuda(
        cudaMalloc(&mat->ptrDevB, batchct * typeCallHost<sizeofCUDTP>(b_type)));
    checkCuda(
        cudaMalloc(&mat->ptrDevC, batchct * typeCallHost<sizeofCUDTP>(c_type)));
    typeCallDev<batchedPtrMagic>(a_type, mat->ptrHostA, mat->ptrDevA, mat->devA,
                                 batchct, m, k);
    typeCallDev<batchedPtrMagic>(b_type, mat->ptrHostB, mat->ptrDevB, mat->devB,
                                 batchct, k, n);
    typeCallDev<batchedPtrMagic>(c_type, mat->ptrHostC, mat->ptrDevC, mat->devC,
                                 batchct, n, m);
  }
}

void cublasGemm::freeMem() {
  free(alpha);
  free(beta);
  free(hostA);
  free(hostB);
  free(hostC);
  for (auto mat : matPtrs) {
    cudaFree(mat.devA);
    cudaFree(mat.devB);
    cudaFree(mat.devC);
    if (batched && !strided) {
      free(mat.ptrHostA);
      free(mat.ptrHostB);
      free(mat.ptrHostC);
      cudaFree(mat.ptrDevA);
      cudaFree(mat.ptrDevB);
      cudaFree(mat.ptrDevC);
    }
  }
}

double cublasGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : matPtrs) {
    if (function == "cublasDgemm" && precision == CUDA_R_64F) {
      std::function<decltype(cublasDgemm)> dgemm_var = cublasDgemm;
      threads.push_back(
          thread(&cublasGemm::testTGemm<double>, this, dgemm_var, &mat));
    } else if (function == "cublasSgemm" && precision == CUDA_R_32F) {
      std::function<decltype(cublasSgemm)> sgemm_var = cublasSgemm;
      threads.push_back(
          thread(&cublasGemm::testTGemm<float>, this, sgemm_var, &mat));
    } else if (function == "cublasHgemm" && precision == CUDA_R_16F) {
      std::function<decltype(cublasHgemm)> hgemm_var = cublasHgemm;
      threads.push_back(
          thread(&cublasGemm::testTGemm<__half>, this, hgemm_var, &mat));
    } else if (function == "cublasZgemm" && precision == CUDA_C_64F) {
      std::function<decltype(cublasZgemm)> zgemm_var = cublasZgemm;
      threads.push_back(thread(&cublasGemm::testTGemm<cuDoubleComplex>, this,
                               zgemm_var, &mat));
    } else if (function == "cublasCgemm" && precision == CUDA_C_32F) {
      std::function<decltype(cublasCgemm)> cgemm_var = cublasCgemm;
      threads.push_back(
          thread(&cublasGemm::testTGemm<cuComplex>, this, cgemm_var, &mat));
    } else if (function == "cublasZgemm3m" && precision == CUDA_C_64F) {
      std::function<decltype(cublasZgemm3m)> zgemm3m_var = cublasZgemm3m;
      threads.push_back(thread(&cublasGemm::testTGemm<cuDoubleComplex>, this,
                               zgemm3m_var, &mat));
    } else if (function == "cublasCgemm3m" && precision == CUDA_C_32F) {
      std::function<decltype(cublasCgemm3m)> cgemm3m_var = cublasCgemm3m;
      threads.push_back(
          thread(&cublasGemm::testTGemm<cuComplex>, this, cgemm3m_var, &mat));
    } else if (function == "cublasDgemmBatched" && precision == CUDA_R_64F) {
      std::function<decltype(cublasDgemmBatched)> dgemm_var =
          cublasDgemmBatched;
      threads.push_back(
          thread(&cublasGemm::testTGemmBatched<double>, this, dgemm_var, &mat));
    } else if (function == "cublasSgemmBatched" && precision == CUDA_R_32F) {
      std::function<decltype(cublasSgemmBatched)> sgemm_var =
          cublasSgemmBatched;
      threads.push_back(
          thread(&cublasGemm::testTGemmBatched<float>, this, sgemm_var, &mat));
    } else if (function == "cublasHgemmBatched" && precision == CUDA_R_16F) {
      std::function<decltype(cublasHgemmBatched)> hgemm_var =
          cublasHgemmBatched;
      threads.push_back(
          thread(&cublasGemm::testTGemmBatched<__half>, this, hgemm_var, &mat));
    } else if (function == "cublasZgemmBatched" && precision == CUDA_C_64F) {
      std::function<decltype(cublasZgemmBatched)> zgemm_var =
          cublasZgemmBatched;
      threads.push_back(thread(&cublasGemm::testTGemmBatched<cuDoubleComplex>,
                               this, zgemm_var, &mat));
    } else if (function == "cublasCgemmBatched" && precision == CUDA_C_32F) {
      std::function<decltype(cublasCgemmBatched)> cgemm_var =
          cublasCgemmBatched;
      threads.push_back(thread(&cublasGemm::testTGemmBatched<cuComplex>, this,
                               cgemm_var, &mat));
    }
    if (function == "cublasDgemmStridedBatched" && precision == CUDA_R_64F) {
      std::function<decltype(cublasDgemmStridedBatched)> dgemm_var =
          cublasDgemmStridedBatched;
      threads.push_back(thread(&cublasGemm::testTGemmStridedBatched<double>,
                               this, dgemm_var, &mat));
    } else if (function == "cublasSgemmStridedBatched" &&
               precision == CUDA_R_32F) {
      std::function<decltype(cublasSgemmStridedBatched)> sgemm_var =
          cublasSgemmStridedBatched;
      threads.push_back(thread(&cublasGemm::testTGemmStridedBatched<float>,
                               this, sgemm_var, &mat));
    } else if (function == "cublasHgemmStridedBatched" &&
               precision == CUDA_R_16F) {
      std::function<decltype(cublasHgemmStridedBatched)> hgemm_var =
          cublasHgemmStridedBatched;
      threads.push_back(thread(&cublasGemm::testTGemmStridedBatched<__half>,
                               this, hgemm_var, &mat));
    } else if (function == "cublasZgemmStridedBatched" &&
               precision == CUDA_C_64F) {
      std::function<decltype(cublasZgemmStridedBatched)> zgemm_var =
          cublasZgemmStridedBatched;
      threads.push_back(
          thread(&cublasGemm::testTGemmStridedBatched<cuDoubleComplex>, this,
                 zgemm_var, &mat));
    } else if (function == "cublasCgemmStridedBatched" &&
               precision == CUDA_C_32F) {
      std::function<decltype(cublasCgemmStridedBatched)> cgemm_var =
          cublasCgemmStridedBatched;
      threads.push_back(thread(&cublasGemm::testTGemmStridedBatched<cuComplex>,
                               this, cgemm_var, &mat));
    } else if (function == "cublasCgemm3mStridedBatched" &&
               precision == CUDA_C_32F) {
      std::function<decltype(cublasCgemm3mStridedBatched)> cgemm_var =
          cublasCgemm3mStridedBatched;
      threads.push_back(thread(&cublasGemm::testTGemmStridedBatched<cuComplex>,
                               this, cgemm_var, &mat));
    }

    if (strided && function == "cublasGemmExStridedBatched") {
      // Call the Gemm strided batched deployment script
    } else if (batched && function == "cublasGemmExBatched") {
      // Call the Gemm batched code
    } else if (batched && function == "cublasGemmEx") {
      threads.push_back(thread(&cublasGemm::testGemmEx, this, &mat));
    }
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflops =
      std::accumulate(begin(matPtrs), end(matPtrs), 0,
                      [](int i, const gemmInst &o) { return o.gflops + i; });
  return gflops;

  // <t>gemm implementation
  // std::cerr << "Invalid implementation & precision combination" <<
  // std::endl; exit(1);
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
        func,
    gemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  checkCuda(cudaSetDevice(mat->devIDX));
  checkCublas(cublasCreate(&handle));
  checkCuda(cudaStreamCreate(&stream));
  checkCublas(cublasSetStream(handle, stream));
  cublasSetWorkspace(handle, mat->devWork, mat->wSZ);
  // std::cout << "Alpha: " << *((T *)alpha) << std::endl;
  // std::cout << "Beta: " << *((T *)beta) << std::endl;
  // double alphaaa = 1;
  // double betaaa = 0;
  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc);

    cudaDeviceSynchronize();
    checkCublas(stat);

    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  mat->devSync->Sync();
  cudaEventRecord(start, stream);
  mat->devSync->Sync();
  for (int rep = 0; rep < iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc);
    // cuBLAS calls are asynchronous, so we have to wait
    // after each call to the BLAS function
    cudaStreamSynchronize(stream);
  }
  mat->devSync->Sync();
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  checkCublas(stat);
  cudaError_t lastError = cudaGetLastError();
  if (lastError != cudaSuccess) {
    std::cerr << cudaGetErrorString(lastError) << std::endl;
  }

  float elapsedTime_ms;
  double totalTime_ms;
  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  totalTime_ms = static_cast<double>(elapsedTime_ms);
  // double totalTime_ms = 0.0;
  // for (int rep = 0; rep < iters; rep++) {
  //  cudaEventRecord(start, 0);
  //  stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP,
  //  ldb,
  //              betaP, devCP, ldc);

  //  cudaEventRecord(stop, 0);
  //  cudaEventSynchronize(stop);

  //  checkCublas(stat);
  //  cudaError_t lastError = cudaGetLastError();
  //  if (lastError != cudaSuccess) {
  //    std::cerr << cudaGetErrorString(lastError) << std::endl;
  //  }

  //  float elapsedTime_ms;

  //  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  //  totalTime_ms += static_cast<double>(elapsedTime_ms);
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
  mat->gflops = gflopPerSec;
  return gflopPerSec;
}

template <typename T>
double cublasGemm::testTGemmBatched(
    std::function<cublasStatus_t(cublasContext *, cublasOperation_t,
                                 cublasOperation_t, int, int, int, T const *,
                                 T const *const *, int, T const *const *, int,
                                 T const *, T *const *, int, int)>
        func,
    gemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;

  cudaSetDevice(mat->devIDX);
  cudaStreamCreate(&stream);
  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T **devAP = reinterpret_cast<T **>(mat->ptrDevA);
  T **devBP = reinterpret_cast<T **>(mat->ptrDevB);
  T **devCP = reinterpret_cast<T **>(mat->ptrDevC);

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
        func,
    gemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  cudaSetDevice(mat->devIDX);
  cudaStreamCreate(&stream);
  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

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
        func,
    gemmInst *mat) {
  return 0.0;
}

double cublasGemm::testGemmEx(gemmInst *mat) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  cudaSetDevice(mat->devIDX);
  cudaStreamCreate(&stream);

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
    stat = cublasGemmEx(handle, transA, transB, m, n, k, alpha, mat->devA,
                        a_type, lda, mat->devB, b_type, ldb, beta, mat->devC,
                        c_type, ldc, compute, CUBLAS_GEMM_DEFAULT);

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