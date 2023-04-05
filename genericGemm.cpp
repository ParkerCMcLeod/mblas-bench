#include "genericGemm.h"

#include <string>

#include "third_party/cxxopts.hpp"

using std::string;

genericGemm::genericGemm(cxxopts::ParseResult result) {
  // Parse basic information
  m = result["m"].as<int>();
  n = result["n"].as<int>();
  k = result["k"].as<int>();

  lda = result["lda"].as<int>();
  ldb = result["ldb"].as<int>();
  ldc = result["ldc"].as<int>();

  // Implement ldd support later
  if (false) {
    ldd = result["ldd"].as<int>();
  }

  strided = false;
  batched = false;
  function = result["function"].as<string>();

  iters = result["iters"].as<int>();
  cold_iters = result["cold_iters"].as<int>();
  batchct = 1;
  if (function.find("Batched") != string::npos) {
    batched = true;
    batchct = result["batch_count"].as<int>();
  }

  if (function.find("Strided") != string::npos) {
    strided = true;
    stride_a = result["stride_a"].as<long long int>();
    stride_b = result["stride_b"].as<long long int>();
    stride_c = result["stride_c"].as<long long int>();
  }
}
