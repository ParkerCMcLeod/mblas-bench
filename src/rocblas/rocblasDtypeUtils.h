#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <map>
#include <string>

bool isReal(rocblas_datatype type);
bool isReal(hipblasltDatatype_t type);
bool isFp8(rocblas_datatype precision);
bool isFp8(hipblasltDatatype_t precision);
std::string precToString(rocblas_datatype precision);
std::string precToString(hipblasltDatatype_t precision);
std::string computeToString(rocblas_datatype compute);
std::string computeToString(hipblasLtComputeType_t compute);
rocblas_datatype precisionStringToRocblasDType(std::string stringPrecision);
hipblasltDatatype_t precisionStringToHipblasltDType(std::string stringPrecision);
rocblas_datatype selectScalar(std::string scalarstr, rocblas_datatype precision,
                              rocblas_datatype compute);
hipblasltDatatype_t selectScalar(std::string scalarstr, hipblasltDatatype_t precision,
                              hipblasLtComputeType_t compute);
rocblas_datatype selectCompute(std::string computestr,
                               rocblas_datatype precision);
hipblasLtComputeType_t selectCompute(std::string computestr,
                               hipblasltDatatype_t precision);
rocblas_operation opStringToRocblasOp(std::string opstr);
hipblasOperation_t opStringToHipblasOp(std::string opstr);
std::string opToString(rocblas_operation);
std::string opToString(hipblasOperation_t);

// data

// clang-format off
const std::map<std::string, rocblas_operation> rocblasOpType = {
  {"N", rocblas_operation_none},
  {"T", rocblas_operation_transpose},
  {"C", rocblas_operation_conjugate_transpose},
};

const std::map<std::string, hipblasOperation_t> hipblasOpType = {
  {"N", HIPBLAS_OP_N},
  {"T", HIPBLAS_OP_T},
  {"C", HIPBLAS_OP_C},
};

const std::map<std::string, rocblas_datatype> precRocblasDType = {
    {"h", rocblas_datatype_f16_r},       {"s", rocblas_datatype_f32_r},     {"d", rocblas_datatype_f64_r},
    {"c", rocblas_datatype_f32_c},       {"z", rocblas_datatype_f64_c},     {"f16_r", rocblas_datatype_f16_r},
    {"f16_c", rocblas_datatype_f16_c},   {"f32_r", rocblas_datatype_f32_r}, {"f32_c", rocblas_datatype_f32_c},
    {"f64_r", rocblas_datatype_f64_r},   {"f64_c", rocblas_datatype_f64_c}, {"bf16_r", rocblas_datatype_bf16_r},
    {"bf16_c", rocblas_datatype_bf16_c}, {"i8_r", rocblas_datatype_i8_r},   {"i8_c", rocblas_datatype_i8_c},
    {"i32_r", rocblas_datatype_i32_r},   {"i32_c", rocblas_datatype_i32_c},
    {"rocblas_datatype_f16_r",  rocblas_datatype_f16_r},
    {"rocblas_datatype_f16_c",  rocblas_datatype_f16_c},
    {"rocblas_datatype_bf16_r", rocblas_datatype_bf16_r},
    {"rocblas_datatype_bf16_c", rocblas_datatype_bf16_c},
    {"rocblas_datatype_f32_r",  rocblas_datatype_f32_r},
    {"rocblas_datatype_f32_c",  rocblas_datatype_f32_c},
    {"rocblas_datatype_f64_r",  rocblas_datatype_f64_r},
    {"rocblas_datatype_f64_c",  rocblas_datatype_f64_c},
    {"rocblas_datatype_i8_r",   rocblas_datatype_i8_r},
    {"rocblas_datatype_i8_c",   rocblas_datatype_i8_c},
    {"rocblas_datatype_u8_r",   rocblas_datatype_u8_r},
    {"rocblas_datatype_u8_c",   rocblas_datatype_u8_c},
    {"rocblas_datatype_i32_r",  rocblas_datatype_i32_r},
    {"rocblas_datatype_i32_c",  rocblas_datatype_i32_c},
    {"rocblas_datatype_u32_r",  rocblas_datatype_u32_r},
    {"rocblas_datatype_u32_c",  rocblas_datatype_u32_c},
};

const std::map<std::string, hipblasltDatatype_t> precHipblasltDType = {
    {"h", HIPBLASLT_R_16F},         {"s", HIPBLASLT_R_32F},         // {"d", HIPBLASLT_R_64F},
    // {"c", HIPBLASLT_C_32F},         {"z", HIPBLASLT_C_64F},
    {"f16_r", HIPBLASLT_R_16F},     // {"f16_c", HIPBLASLT_C_16F},
    {"f32_r", HIPBLASLT_R_32F},     // {"f32_c", HIPBLASLT_C_32F},
    // {"f64_r", HIPBLASLT_R_64F},     {"f64_c", HIPBLASLT_C_64F},
    {"bf16_r", HIPBLASLT_R_16B},    // {"bf16_c", HIPBLASLT_C_16B},
    // {"i8_r", HIPBLASLT_R_8I},       {"i8_c", HIPBLASLT_C_8I},
    // {"i32_r", HIPBLASLT_R_32I},     {"i32_c", HIPBLASLT_C_32I},
    {"HIPBLASLT_R_16F", HIPBLASLT_R_16F},
    // {"HIPBLASLT_C_16F",   HIPBLASLT_C_16F},
    {"HIPBLASLT_R_16B", HIPBLASLT_R_16B},
    // {"HIPBLASLT_C_16B",   HIPBLASLT_C_16B},
    {"HIPBLASLT_R_32F", HIPBLASLT_R_32F},
    // {"HIPBLASLT_C_32F",   HIPBLASLT_C_32F},
    // {"HIPBLASLT_R_64F",   HIPBLASLT_R_64F},
    // {"HIPBLASLT_C_64F",   HIPBLASLT_C_64F},
    {"HIPBLASLT_R_8F_E4M3", HIPBLASLT_R_8F_E4M3},
    {"HIPBLASLT_R_8F_E5M2", HIPBLASLT_R_8F_E5M2},
};

const std::map<std::string, rocblas_datatype> computeRocblasDType = {
    {"rocblas_datatype_f16_r", rocblas_datatype_f16_r},
    {"rocblas_datatype_f32_r", rocblas_datatype_f32_r},
    {"rocblas_datatype_f64_r", rocblas_datatype_f64_r},
    {"rocblas_datatype_i32_r", rocblas_datatype_i32_r},
    {"f16_r", rocblas_datatype_f16_r},
    {"f32_r", rocblas_datatype_f32_r},
    {"f64_r", rocblas_datatype_f64_r},
    {"i32_r", rocblas_datatype_i32_r},
};

const std::map<std::string, hipblasLtComputeType_t> computeHipblasltDType = {
    {"HIPBLASLT_COMPUTE_F32", HIPBLASLT_COMPUTE_F32},
    {"f32_r", HIPBLASLT_COMPUTE_F32},
};

const std::map<rocblas_datatype, rocblas_datatype> precToRocblasCompute = {
    {rocblas_datatype_f64_r, rocblas_datatype_f64_r},
    {rocblas_datatype_f64_c, rocblas_datatype_f64_r},
    {rocblas_datatype_f32_r, rocblas_datatype_f32_r},
    {rocblas_datatype_f32_c, rocblas_datatype_f32_r},
    {rocblas_datatype_bf16_r, rocblas_datatype_bf16_r},
    {rocblas_datatype_bf16_c, rocblas_datatype_bf16_r},
    {rocblas_datatype_f16_r, rocblas_datatype_f16_r},
    {rocblas_datatype_f16_c, rocblas_datatype_f16_r},
    {rocblas_datatype_i32_r, rocblas_datatype_i32_r},
};

const std::map<hipblasltDatatype_t, hipblasLtComputeType_t> precToHipblasltCompute = {
    {HIPBLASLT_R_32F, HIPBLASLT_COMPUTE_F32},
    // {HIPBLASLT_C_32F, HIPBLASLT_COMPUTE_F32},
};
// clang-format on