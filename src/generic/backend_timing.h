#pragma once

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
