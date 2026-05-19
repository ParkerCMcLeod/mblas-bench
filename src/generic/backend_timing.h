#pragma once

#include <chrono>
#include <functional>

// Generic GPU event-based timing helper.
//
// This header is vendor-neutral: it contains no CUDA or HIP includes.
// All GPU operations are provided via a traits struct so that each backend
// supplies its own types and API calls at compile time.
//
// Usage (CUDA example):
//
//   struct CudaTimingTraits {
//       using Event  = cudaEvent_t;
//       using Stream = cudaStream_t;
//       static void streamSynchronize(Stream s) { cudaStreamSynchronize(s); }
//       static void eventCreate(Event *e)       { cudaEventCreate(e); }
//       static void eventDestroy(Event e)       { cudaEventDestroy(e); }
//       static void eventRecord(Event e, Stream s) { cudaEventRecord(e, s); }
//       static void eventSynchronize(Event e)   { cudaEventSynchronize(e); }
//       static void eventElapsedTime(float *ms, Event start, Event stop) {
//           cudaEventElapsedTime(ms, start, stop);
//       }
//   };
//
//   float ms = gpu_timed_run<CudaTimingTraits>(stream, cold_iters, iters,
//                                               cold_kernel, hot_kernel);

// Run a kernel bracketed by GPU events on `stream`, returning elapsed wall
// time in milliseconds.
//
// `cold_fn` is called `cold_iters` times before timing begins (typically
// includes error checking after each call).
//
// `hot_fn` is called `iters` times inside the timed region (no per-iteration
// error checking to avoid skewing measurements).
//
// Both functions take a single int argument: the iteration index.
template <typename Traits>
float gpu_timed_run(typename Traits::Stream stream,
                    int cold_iters,
                    int iters,
                    std::function<void(int)> cold_fn,
                    std::function<void(int)> hot_fn) {
  // Cold iterations (untimed, with error checking)
  for (int rep = 0; rep < cold_iters; ++rep) {
    cold_fn(rep);
  }
  Traits::streamSynchronize(stream);

  // Create timing events
  typename Traits::Event start, stop;
  Traits::eventCreate(&start);
  Traits::eventCreate(&stop);

  // Timed iterations
  Traits::eventRecord(start, stream);
  for (int rep = 0; rep < iters; ++rep) {
    hot_fn(rep);
  }
  Traits::eventRecord(stop, stream);
  Traits::eventSynchronize(stop);

  float elapsed_ms = 0.0f;
  Traits::eventElapsedTime(&elapsed_ms, start, stop);

  Traits::eventDestroy(start);
  Traits::eventDestroy(stop);

  return elapsed_ms;
}

// Convenience overload: same kernel for both cold and hot iterations.
template <typename Traits>
float gpu_timed_run(typename Traits::Stream stream,
                    int cold_iters,
                    int iters,
                    std::function<void(int)> kernel_fn) {
  return gpu_timed_run<Traits>(stream, cold_iters, iters, kernel_fn, kernel_fn);
}

// Result of a time-budgeted run: GPU elapsed time AND the number of hot
// iterations actually completed (so callers can compute throughput).
struct gpu_timed_result {
  float elapsed_ms;
  int iters_completed;
};

// Iter-or-time-budgeted run. For each phase (cold, hot) the *_time_ms argument
// takes precedence if > 0: the loop launches kernels until that many ms have
// elapsed (CPU clock). Otherwise the corresponding *_iters argument is used as
// a fixed iteration count.
//
// Hot phase uses GPU events to time only the kernels actually launched; the
// returned iters_completed is the count of hot iterations executed (which is
// what calculate_figure_of_merit needs in time-budgeted mode).
template <typename Traits>
gpu_timed_result gpu_timed_run_budget(typename Traits::Stream stream,
                                      int cold_iters, int cold_time_ms,
                                      int iters,      int hot_time_ms,
                                      std::function<void(int)> cold_fn,
                                      std::function<void(int)> hot_fn) {
  using clock = std::chrono::steady_clock;

  // Cold phase
  if (cold_time_ms > 0) {
    auto t0 = clock::now();
    int rep = 0;
    while (true) {
      cold_fn(rep++);
      Traits::streamSynchronize(stream);
      if (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count() >= cold_time_ms)
        break;
    }
  } else {
    for (int rep = 0; rep < cold_iters; ++rep) cold_fn(rep);
  }
  Traits::streamSynchronize(stream);

  // Hot phase — GPU event timing
  typename Traits::Event start_ev, stop_ev;
  Traits::eventCreate(&start_ev);
  Traits::eventCreate(&stop_ev);

  Traits::eventRecord(start_ev, stream);
  int hot_rep = 0;
  if (hot_time_ms > 0) {
    auto t0 = clock::now();
    while (true) {
      hot_fn(hot_rep++);
      if (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count() >= hot_time_ms)
        break;
    }
  } else {
    for (; hot_rep < iters; ++hot_rep) hot_fn(hot_rep);
  }
  Traits::eventRecord(stop_ev, stream);
  Traits::eventSynchronize(stop_ev);

  float elapsed_ms = 0.0f;
  Traits::eventElapsedTime(&elapsed_ms, start_ev, stop_ev);

  Traits::eventDestroy(start_ev);
  Traits::eventDestroy(stop_ev);

  return {elapsed_ms, hot_rep};
}
