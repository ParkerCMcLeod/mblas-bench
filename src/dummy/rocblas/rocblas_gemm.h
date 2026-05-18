#pragma once
#include <memory>
#include <stdexcept>
#include <generic_gemm.h>
#include <cxxopts.hpp>

inline std::unique_ptr<generic_gemm> make_rocblas_gemm(cxxopts::ParseResult) {
  throw std::runtime_error("Support for rocblas backend not compiled");
}
