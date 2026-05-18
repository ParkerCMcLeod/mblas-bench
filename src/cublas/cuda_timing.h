#pragma once

#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include "cuda_error.h"

namespace mblas_timing {

struct timed_result { int iters; double gpu_ms; };

inline timed_result run_serialized(cudaStream_t stream, int fixed_iters, int time_budget_ms,
                                   std::function<void(int)> fn) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start));
  check_cuda(cudaEventCreate(&stop));

  int completed = 0;
  double total_ms = 0.0;

  if (time_budget_ms > 0) {
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      check_cuda(cudaEventRecord(start, stream));
      fn(completed);
      check_cuda(cudaEventRecord(stop, stream));
      check_cuda(cudaEventSynchronize(stop));

      float iter_ms = 0.0f;
      check_cuda(cudaEventElapsedTime(&iter_ms, start, stop));
      total_ms += static_cast<double>(iter_ms);
      completed++;

      auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t0).count();
      if (wall_ms > time_budget_ms) break;
    }
  } else {
    for (int rep = 0; rep < fixed_iters; rep++) {
      check_cuda(cudaEventRecord(start, stream));
      fn(rep);
      check_cuda(cudaEventRecord(stop, stream));
      check_cuda(cudaEventSynchronize(stop));

      float iter_ms = 0.0f;
      check_cuda(cudaEventElapsedTime(&iter_ms, start, stop));
      total_ms += static_cast<double>(iter_ms);
    }
    completed = fixed_iters;
  }

  check_cuda(cudaEventDestroy(start));
  check_cuda(cudaEventDestroy(stop));
  return {completed, total_ms};
}

inline timed_result run_pipelined(cudaStream_t stream, int fixed_iters, int time_budget_ms,
                                  std::function<void(int)> fn) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start));
  check_cuda(cudaEventCreate(&stop));

  int completed = 0;

  check_cuda(cudaEventRecord(start, stream));
  if (time_budget_ms > 0) {
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      fn(completed);
      completed++;
      auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t0).count();
      if (wall_ms > time_budget_ms) break;
    }
  } else {
    for (int rep = 0; rep < fixed_iters; rep++) {
      fn(rep);
    }
    completed = fixed_iters;
  }
  check_cuda(cudaEventRecord(stop, stream));
  check_cuda(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&total_ms, start, stop));
  check_cuda(cudaEventDestroy(start));
  check_cuda(cudaEventDestroy(stop));
  return {completed, static_cast<double>(total_ms)};
}

} // namespace mblas_timing
