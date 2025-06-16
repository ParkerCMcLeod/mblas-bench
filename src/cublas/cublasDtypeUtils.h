#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>
#include "mblasCuDataType.h"
#include "mblasCuComputeType.h"

bool match_gemm_type(mblasDataType precision, std::string function, mblasDataType desiredPrec, std::vector<std::string> acceptable);

#if (ENABLE_CUDA_FP4)
std::pair<size_t, size_t> get_scale_tensor_size(int rows, int cols, cublasLtMatmulMatrixScale_t ScaleMode);
#endif

static size_t roundoff(size_t  x, size_t granul);