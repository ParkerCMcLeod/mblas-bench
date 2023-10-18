#pragma once 
#include <genericGemmFactory.h>
#include <exception>

class hipblasLtGemmFactory : public genericGemmFactory {
 public:
  void createGemm(cxxopts::ParseResult) override {
    throw std::runtime_error("Support for hipblasLt backend not compiled");
  }
};