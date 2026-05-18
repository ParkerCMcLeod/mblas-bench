#pragma once

#include <cassert>
#include <iostream>
#include <cstdint>

#include "generic_gemm.h"

uint64_t round_up(uint64_t numToRound, uint64_t multiple);

template <typename T>
T ceil_division(T x, T y) {
  return (x + y - 1) / y;
}

uint64_t calculate_offsets(
    const generic_gemm::matrix_alloc_desc& a, const generic_gemm::matrix_alloc_desc& b,
    const generic_gemm::matrix_alloc_desc& c, const generic_gemm::matrix_alloc_desc& d,
    int batch_count, bool inplace
);
