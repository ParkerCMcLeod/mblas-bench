#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <map>
#include <string>

bool isReal(rocblas_datatype type);
bool isReal(hipblasDatatype_t type);
bool isFp8(rocblas_datatype precision);
bool isFp8(hipblasDatatype_t precision);
std::string precToString(rocblas_datatype precision);
std::string precToString(hipblasDatatype_t precision);
std::string computeToString(rocblas_datatype compute);
std::string computeToString(hipblasLtComputeType_t compute);
rocblas_datatype precisionStringToRocblasDType(std::string stringPrecision);
hipblasDatatype_t precisionStringToHipblasDType(std::string stringPrecision);
rocblas_datatype selectScalar(std::string scalarstr, rocblas_datatype precision,
                              rocblas_datatype compute);
hipblasDatatype_t selectScalar(std::string scalarstr, hipblasDatatype_t precision,
                              hipblasLtComputeType_t compute);
rocblas_datatype selectCompute(std::string computestr,
                               rocblas_datatype precision);
hipblasLtComputeType_t selectCompute(std::string computestr,
                               hipblasDatatype_t precision);
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

const std::map<std::string, hipblasDatatype_t> precHipblasDType = {
    {"h", HIPBLAS_R_16F},       {"s", HIPBLAS_R_32F},       {"d", HIPBLAS_R_64F},
    {"c", HIPBLAS_C_32F},       {"z", HIPBLAS_C_64F},       {"f16_r", HIPBLAS_R_16F},
    {"f16_c", HIPBLAS_C_16F},   {"f32_r", HIPBLAS_R_32F},   {"f32_c", HIPBLAS_C_32F},
    {"f64_r", HIPBLAS_R_64F},   {"f64_c", HIPBLAS_C_64F},   {"bf16_r", HIPBLAS_R_16B},
    {"bf16_c", HIPBLAS_C_16B},  {"i8_r", HIPBLAS_R_8I},     {"i8_c", HIPBLAS_C_8I},
    {"i32_r", HIPBLAS_R_32I},   {"i32_c", HIPBLAS_C_32I},
    {"HIPBLAS_R_16F",   HIPBLAS_R_16F},
    {"HIPBLAS_C_16F",   HIPBLAS_C_16F},
    {"HIPBLAS_R_16B",   HIPBLAS_R_16B},
    {"HIPBLAS_C_16B",   HIPBLAS_C_16B},
    {"HIPBLAS_R_32F",   HIPBLAS_R_32F},
    {"HIPBLAS_C_32F",   HIPBLAS_C_32F},
    {"HIPBLAS_R_64F",   HIPBLAS_R_64F},
    {"HIPBLAS_C_64F",   HIPBLAS_C_64F},
    {"HIPBLAS_R_8I",    HIPBLAS_R_8I},
    {"HIPBLAS_C_8I",    HIPBLAS_C_8I},
    {"HIPBLAS_R_8U",    HIPBLAS_R_8U},
    {"HIPBLAS_C_8U",    HIPBLAS_C_8U},
    {"HIPBLAS_R_32I",   HIPBLAS_R_32I},
    {"HIPBLAS_C_32I",   HIPBLAS_C_32I},
    {"HIPBLAS_R_32U",   HIPBLAS_R_32U},
    {"HIPBLAS_C_32U",   HIPBLAS_C_32U},
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

const std::map<std::string, hipblasLtComputeType_t> computeHipblasDType = {
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

const std::map<hipblasDatatype_t, hipblasLtComputeType_t> precToHipblasCompute = {
    {HIPBLAS_R_32F, HIPBLASLT_COMPUTE_F32},
    {HIPBLAS_C_32F, HIPBLASLT_COMPUTE_F32},
};
// clang-format on