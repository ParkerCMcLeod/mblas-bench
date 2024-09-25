#include <cxxopts.hpp>
#include <genericGemm.h>
#include <rocblasGemm.h>
#include <rocblasGemmFactory.h>

void rocblasGemmFactory::createGemm(cxxopts::ParseResult result) {
  gemm = new rocblasGemm(result);
}