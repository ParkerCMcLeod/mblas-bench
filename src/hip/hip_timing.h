#pragma once

#include <chrono>
#include <functional>
#include <hip/hip_runtime.h>
#include "hip_error.h"

namespace mblas_timing {

struct timed_result { int iters; double gpu_ms; };

inline timed_result run_serialized(hipStream_t stream, int fixed_iters, int time_budget_ms,
                                   std::function<void(int)> fn) {
  hipEvent_t start, stop;
  check_hip(hipEventCreate(&start));
  check_hip(hipEventCreate(&stop));

  int completed = 0;
  double total_ms = 0.0;

  if (time_budget_ms > 0) {
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      check_hip(hipEventRecord(start, stream));
      fn(completed);
      check_hip(hipEventRecord(stop, stream));
      check_hip(hipEventSynchronize(stop));

      float iter_ms = 0.0f;
      check_hip(hipEventElapsedTime(&iter_ms, start, stop));
      total_ms += static_cast<double>(iter_ms);
      completed++;

      auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t0).count();
      if (wall_ms > time_budget_ms) break;
    }
  } else {
    for (int rep = 0; rep < fixed_iters; rep++) {
      check_hip(hipEventRecord(start, stream));
      fn(rep);
      check_hip(hipEventRecord(stop, stream));
      check_hip(hipEventSynchronize(stop));

      float iter_ms = 0.0f;
      check_hip(hipEventElapsedTime(&iter_ms, start, stop));
      total_ms += static_cast<double>(iter_ms);
    }
    completed = fixed_iters;
  }

  check_hip(hipEventDestroy(start));
  check_hip(hipEventDestroy(stop));
  return {completed, total_ms};
}

inline timed_result run_pipelined(hipStream_t stream, int fixed_iters, int time_budget_ms,
                                  std::function<void(int)> fn) {
  hipEvent_t start, stop;
  check_hip(hipEventCreate(&start));
  check_hip(hipEventCreate(&stop));

  int completed = 0;

  check_hip(hipEventRecord(start, stream));
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
  check_hip(hipEventRecord(stop, stream));
  check_hip(hipEventSynchronize(stop));

  float total_ms = 0.0f;
  check_hip(hipEventElapsedTime(&total_ms, start, stop));
  check_hip(hipEventDestroy(start));
  check_hip(hipEventDestroy(stop));
  return {completed, static_cast<double>(total_ms)};
}

} // namespace mblas_timing
