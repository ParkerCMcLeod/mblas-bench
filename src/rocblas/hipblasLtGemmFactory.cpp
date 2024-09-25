#include <cxxopts.hpp>
#include <genericGemm.h>
#include <hipblasLtGemm.h>
#include <hipblasLtGemmFactory.h>

void hipblasLtGemmFactory::createGemm(cxxopts::ParseResult result) {
  gemm = new hipblasLtGemm(result);
}