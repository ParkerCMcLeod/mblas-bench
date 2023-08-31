#pragma once

#include <string>

#include "genericGemm.h"

class cublasGemm : public genericGemm {
 public:
  cublasGemm(cxxopts::ParseResult result);
  std::string prepareArray();
  double test();
  std::string getResultString();
  virtual void freeMem();
};
