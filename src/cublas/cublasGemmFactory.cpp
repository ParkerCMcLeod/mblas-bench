#include <cxxopts.hpp>
#include <genericGemm.h>
#include <cublasGemm.h>
#include <cublasGemmFactory.h>

//genericGemm* cublasGemmFactory::createGemm(cxxopts::ParseResult result) const {
//  return new cublasGemm(result);
//}

void cublasGemmFactory::createGemm(cxxopts::ParseResult result) {
  gemm = new cublasGemm(result);
}