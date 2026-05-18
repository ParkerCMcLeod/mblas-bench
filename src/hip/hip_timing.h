#pragma once
#include <hip/hip_runtime.h>

#include "backend_timing.h"

// HIP-specific timing traits for the generic gpu_timed_run<> template.
struct HipTimingTraits {
  using Event  = hipEvent_t;
  using Stream = hipStream_t;

  static void streamSynchronize(Stream s) { hipStreamSynchronize(s); }
  static void eventCreate(Event *e)       { hipEventCreate(e); }
  static void eventDestroy(Event e)       { hipEventDestroy(e); }
  static void eventRecord(Event e, Stream s) { hipEventRecord(e, s); }
  static void eventSynchronize(Event e)   { hipEventSynchronize(e); }
  static void eventElapsedTime(float *ms, Event start, Event stop) {
    hipEventElapsedTime(ms, start, stop);
  }
};
