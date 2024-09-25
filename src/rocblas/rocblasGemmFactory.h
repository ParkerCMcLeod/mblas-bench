#pragma once 
#include <genericGemmFactory.h>

class rocblasGemmFactory : public genericGemmFactory {
 public:
  void createGemm(cxxopts::ParseResult) override;
};