#include "generic_setup.h"

#include <cassert>
#include <iostream>

/*
 * Round's a variable up to the nearest number divisible by 16
 * Used for ensuring that all our pointers fall on a 16 byte boundary
*/
uint64_t round_up(uint64_t numToRound, uint64_t multiple)
{
  uint64_t remainder = numToRound % multiple;
  if (remainder == 0) return numToRound;
  uint64_t new_num = numToRound + (multiple - remainder);
  assert(new_num % multiple == 0);
  return new_num;
}

/*
 * Calculates the offset between blocks in the rotating tensor area
 * Returns: total_block_size
 * Also updated: <a-d>_offset
 */
uint64_t calculate_offsets(
    const generic_gemm::matrix_alloc_desc& a, const generic_gemm::matrix_alloc_desc& b,
    const generic_gemm::matrix_alloc_desc& c, const generic_gemm::matrix_alloc_desc& d,
    int batch_count, bool inplace
) {
  uint64_t a_size, b_size, c_size, d_size;
  uint64_t a_size_round, b_size_round, c_size_round, d_size_round;
  uint64_t total_block_size;
  a_size = ceil_division(a.rows_mem * a.cols_mem * batch_count * a.type_size, uint64_t(a.type_pack));
  a_size_round = round_up(a_size, 16);
  b_size = ceil_division(b.rows_mem * b.cols_mem * batch_count * b.type_size, uint64_t(b.type_pack));
  b_size_round = round_up(b_size, 16);
  c_size = ceil_division(c.rows_mem * c.cols_mem * batch_count * c.type_size, uint64_t(c.type_pack));
  c_size_round = round_up(c_size, 16);
  if (!inplace) {
    d_size = ceil_division(d.rows_mem * d.cols_mem * batch_count * d.type_size, uint64_t(d.type_pack));
    d_size_round = round_up(d_size, 16);
  } else {
    d_size = c_size;
    d_size_round = 0;
  }

  total_block_size = a_size_round + b_size_round + c_size_round + d_size_round;
  return total_block_size;
}