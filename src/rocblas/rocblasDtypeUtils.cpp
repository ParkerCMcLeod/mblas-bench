#include "rocblasDtypeUtils.h"

#include <hip/hip_runtime.h>

#include <iostream>
#include <map>
#include <string>
using namespace std;

bool isReal(rocblas_datatype type) {
  // You could also do this based on the string version with _R_ or _C_, but
  // those are hardcoded anyway
  switch (type) {
    case rocblas_datatype_f16_r:
    case rocblas_datatype_bf16_r:
    case rocblas_datatype_f32_r:
    case rocblas_datatype_f64_r:
    case rocblas_datatype_i8_r:
    case rocblas_datatype_u8_r:
    case rocblas_datatype_i32_r:
    case rocblas_datatype_u32_r:
      return true;
      break;

    // Complex numbers
    case rocblas_datatype_f16_c:
    case rocblas_datatype_bf16_c:
    case rocblas_datatype_f32_c:
    case rocblas_datatype_f64_c:
    case rocblas_datatype_i8_c:
    case rocblas_datatype_u8_c:
    case rocblas_datatype_i32_c:
    case rocblas_datatype_u32_c:
      return false;
      break;
    // Assume real I guess
    default:
      return true;
  }
}

bool isReal(hipblasltDatatype_t type) {
  // You could also do this based on the string version with _R_ or _C_, but
  // those are hardcoded anyway
  switch (type) {
    case HIPBLASLT_R_16F:
    case HIPBLASLT_R_16B:
    case HIPBLASLT_R_32F:
      return true;
      break;

    // Complex numbers
    // case HIPBLASLT_C_16F:
    // case HIPBLASLT_C_16B:
    // case HIPBLASLT_C_32F:
    //   return false;
    //   break;
    // Assume real I guess
    default:
      return true;
  }
}

bool isFp8(rocblas_datatype precision) {
  // if (precision == CUDA_R_8F_E4M3 || precision == CUDA_R_8F_E5M2) {
  //   return true;
  // }
  return false;
}

bool isFp8(hipblasltDatatype_t precision) {
  if (precision == HIPBLASLT_R_8F_E4M3 || precision == HIPBLASLT_R_8F_E5M2) {
    return true;
  }
  return false;
}

std::string precToString(rocblas_datatype precision) {
  for (auto ele : precRocblasDType) {
    if (ele.second == precision && ele.first.find("rocblas_datatype") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

std::string precToString(hipblasltDatatype_t precision) {
  for (auto ele : precHipblasltDType) {
    if (ele.second == precision && ele.first.find("HIPBLASLT") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

std::string computeToString(rocblas_datatype compute) {
  for (auto ele : computeRocblasDType) {
    if (ele.second == compute && ele.first.find("rocblas_datatype") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

std::string computeToString(hipblasLtComputeType_t compute) {
  for (auto ele : computeHipblasltDType) {
    if (ele.second == compute && ele.first.find("HIPBLASLT_COMPUTE") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

rocblas_datatype precisionStringToRocblasDType(std::string stringPrecision) {
  try {
    return precRocblasDType.at(stringPrecision);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << stringPrecision << std::endl;
    throw e;
    return rocblas_datatype_f32_r;
  }
}

hipblasltDatatype_t precisionStringToHipblasltDType(std::string stringPrecision) {
  try {
    return precHipblasltDType.at(stringPrecision);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << stringPrecision << std::endl;
    throw e;
    return HIPBLASLT_R_32F;
  }
}

rocblas_datatype selectCompute(std::string computestr,
                               rocblas_datatype precision) {
  if (computestr == "") {
    // If the user doesnt specify, just guess based on precision
    rocblas_datatype compute;
    try {
      compute = precToRocblasCompute.at(precision);
    } catch (std::out_of_range &e) {
      compute = rocblas_datatype_f32_r;
    }
    return compute;
  }
  try {
    return computeRocblasDType.at(computestr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << computestr << std::endl;
    throw e;
    return rocblas_datatype_f32_r;
  }
}

hipblasLtComputeType_t selectCompute(std::string computestr,
                                  hipblasltDatatype_t precision) {
  if (computestr == "") {
    // If the user doesnt specify, just guess based on precision
    hipblasLtComputeType_t compute;
    try {
      compute = precToHipblasltCompute.at(precision);
    } catch (std::out_of_range &e) {
      compute = HIPBLASLT_COMPUTE_F32;
    }
    return compute;
  }
  try {
    return computeHipblasltDType.at(computestr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << computestr << std::endl;
    throw e;
    return HIPBLASLT_COMPUTE_F32;
  }
}

rocblas_datatype selectScalar(std::string scalarstr, rocblas_datatype precision,
                              rocblas_datatype compute) {
  if (scalarstr == "") {
    // Scalar type not specified, setting based on compute type
    for (auto ele : precToRocblasCompute) {
      if (ele.second == compute && isReal(precision) == isReal(ele.first)) {
        return ele.first;
      }
    }
    // something terrible has happened
    return precision;
  }
  return precisionStringToRocblasDType(scalarstr);
}

hipblasltDatatype_t selectScalar(std::string scalarstr, hipblasltDatatype_t precision,
                              hipblasLtComputeType_t compute) {
  if (scalarstr == "") {
    // Scalar type not specified, setting based on compute type
    for (auto ele : precToHipblasltCompute) {
      if (ele.second == compute && isReal(precision) == isReal(ele.first)) {
        return ele.first;
      }
    }
    // something terrible has happened
    return precision;
  }
  return precisionStringToHipblasltDType(scalarstr);
}

rocblas_operation opStringToRocblasOp(std::string opstr) {
  if (opstr.empty()) {
    return rocblas_operation_none;
  }
  try {
    return rocblasOpType.at(opstr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << opstr << std::endl;
    throw e;
  }
}

hipblasOperation_t opStringToHipblasOp(std::string opstr) {
  if (opstr.empty()) {
    return HIPBLAS_OP_N;
  }
  try {
    return hipblasOpType.at(opstr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << opstr << std::endl;
    throw e;
  }
}

std::string opToString(rocblas_operation op) {
  for (auto ele : rocblasOpType) {
    if (ele.second == op) {
      return ele.first;
    }
  }
  return "N";
}

std::string opToString(hipblasOperation_t op) {
  for (auto ele : hipblasOpType) {
    if (ele.second == op) {
      return ele.first;
    }
  }
  return "N";
}