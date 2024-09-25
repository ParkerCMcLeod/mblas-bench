#pragma once 
#include <genericGemmFactory.h>
#include <cxxopts.hpp>

class cublasGemmFactory : public genericGemmFactory {
 public:
  void createGemm(cxxopts::ParseResult) override;
};