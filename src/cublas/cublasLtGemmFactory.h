#pragma once 
#include <genericGemmFactory.h>
#include <cxxopts.hpp>

class cublasLtGemmFactory : public genericGemmFactory {
 public:
  void createGemm(cxxopts::ParseResult) override;
};