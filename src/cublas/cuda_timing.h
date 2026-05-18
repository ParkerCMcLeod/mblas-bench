#pragma once
#include <cuda_runtime.h>

#include "backend_timing.h"

// CUDA-specific timing traits for the generic gpu_timed_run<> template.
struct CudaTimingTraits {
  using Event  = cudaEvent_t;
  using Stream = cudaStream_t;

  static void streamSynchronize(Stream s) { cudaStreamSynchronize(s); }
  static void eventCreate(Event *e)       { cudaEventCreate(e); }
  static void eventDestroy(Event e)       { cudaEventDestroy(e); }
  static void eventRecord(Event e, Stream s) { cudaEventRecord(e, s); }
  static void eventSynchronize(Event e)   { cudaEventSynchronize(e); }
  static void eventElapsedTime(float *ms, Event start, Event stop) {
    cudaEventElapsedTime(ms, start, stop);
  }
};
