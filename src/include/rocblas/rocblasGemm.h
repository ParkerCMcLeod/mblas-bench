#pragma once
//#include <rocblas/rocblas.h>
//#include <hip/hip_runtime.h>
#include <cxxabi.h>

//#include <iostream>
//#include <vector>
#include <string>

#include "genericGemm.h"



class rocblasGemm : public genericGemm {
 public:
  rocblasGemm(cxxopts::ParseResult result);
  std::string prepareArray();
  double test();
  std::string getResultString();
  virtual void freeMem();

};
