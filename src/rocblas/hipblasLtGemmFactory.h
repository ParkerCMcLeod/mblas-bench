#pragma once 
#include <genericGemmFactory.h>

class hipblasLtGemmFactory : public genericGemmFactory {
 public:
  void createGemm(cxxopts::ParseResult) override;
};