#pragma once

#include <string>

#include "genericGemm.h"

class cublasLtGemm : public genericGemm {
 public:
  cublasLtGemm(cxxopts::ParseResult result);
  std::string prepareArray();
  double test();
  std::string getResultString();
  virtual void freeMem();
};
