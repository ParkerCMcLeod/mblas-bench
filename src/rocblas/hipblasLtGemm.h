#pragma once
#include <hipblaslt/hipblaslt.h>
// #include <hip/hip_runtime.h>
#include <cxxabi.h>

#include <iostream>
#include <string>

#include "genericGemm.h"

struct matmulPrecType {
  hipblasLtComputeType_t compute;
  hipblasDatatype_t scalar;
  hipblasDatatype_t a_type;
  hipblasDatatype_t b_type;
  hipblasDatatype_t c_type;
  hipblasDatatype_t d_type;
  hipblasDatatype_t bias_type;
  bool operator==(const matmulPrecType rhs) const {
    return rhs.compute == compute && rhs.scalar == scalar &&
           rhs.a_type == a_type && rhs.b_type == b_type &&
           rhs.c_type == c_type && rhs.d_type == d_type &&
           // Omitting bias type is acceptable
           (rhs.bias_type == bias_type ||
            rhs.bias_type == (hipblasDatatype_t)(-1));
  }
};

struct hipblasLtGemmInst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  void *devA;
  void *devB;
  void *devC;
  void *devD;
  hipblasLtMatmulDesc_t descOP;
  hipblasLtMatrixLayout_t descA;
  hipblasLtMatrixLayout_t descB;
  hipblasLtMatrixLayout_t descC;
  hipblasLtMatrixLayout_t descD;
  hipblasLtMatmulPreference_t pref;
  hipblasLtMatmulHeuristicResult_t algo;
  void *devWork;
  long wSZ;
  hipblasLtGemmInst(int devID) { devIDX = devID; }
};

class hipblasLtGemm : public genericGemm {
 private:
  void *hostA;
  void *hostB;
  void *hostC;

  void *alpha;
  void *beta;

  bool inplace = false;
  hipblasOperation_t transA;
  hipblasOperation_t transB;

  hipblasDatatype_t precision;
  hipblasLtComputeType_t compute;
  hipblasDatatype_t scalar;
  hipblasDatatype_t a_type;
  hipblasDatatype_t b_type;
  hipblasDatatype_t c_type;
  hipblasDatatype_t d_type;
  hipblasDatatype_t bias_type;

  int workspaceSz = 64 * 1024 * 1024;

  static std::vector<matmulPrecType> matmulSupported;
  std::vector<hipblasLtGemmInst> matPtrs;

 private:
  // hipblasDatatype_t precisionStringToHipblasDType(std::string stringPrecision);
  // void parseMType(std::string a, std::string b, std::string c);
  void parseMType(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr,
                  std::string dStr);
  void validateParameters();
  void parseDevIters(std::string);
  void allocHost();
  void allocDev(hipblasLtGemmInst *);
  void fillHost();
  void copyHostToDev(hipblasLtGemmInst *);
  void prepareMatrix(hipblasLtGemmInst *);
  void noTuning(hipblasLtGemmInst *);
  void autoTuning(hipblasLtGemmInst *);
  void runThreaded(void (hipblasLtGemm::*func)(hipblasLtGemmInst *));
  std::tuple<double, double, double> calculateFOM(double totalTime_ms);
  void testMatmul(hipblasLtGemmInst *mat);

 public:
  hipblasLtGemm(cxxopts::ParseResult result);
  std::string prepareArray();
  double test();
  std::string getResultString();
  virtual void freeMem();
};