#pragma once
//#include <hipblaslt/hipblaslt.h>
// #include <hip/hip_runtime.h>
//#include <cxxabi.h>

#include <string>

#include "genericGemm.h"


class hipblasLtGemm : public genericGemm {
 public:
  hipblasLtGemm(cxxopts::ParseResult result);
  std::string prepareArray();
  double test();
  std::string getResultString();
  virtual void freeMem();
};