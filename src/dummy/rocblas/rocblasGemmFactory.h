#pragma once 
#include <genericGemmFactory.h>
#include <exception>

class rocblasGemmFactory : public genericGemmFactory {
 public:
  void createGemm(cxxopts::ParseResult) override {
    throw std::runtime_error("Support for rocblas backend not compiled");
  }
};