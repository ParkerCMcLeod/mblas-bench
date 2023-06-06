#include "cublasCreateAllocate.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cuda/std/complex>
#include <iostream>
#include <sstream>
#include <string>

#include "cudaError.h"

using cuda::std::complex;
using std::string;

// struct sizeofCUDTHost
// {
//     int operator()()
//     {

//     }
// };

// template void *allocSetScalar<double>::operator()(string);
// template void *allocSetScalar<float>::operator()(string);

template <typename T>
int sizeofCUDT<T>::operator()() {
  return sizeof(T);
}

template <typename T>
int sizeofCUDTP<T>::operator()() {
  return sizeof(T *);
}

template <typename T>
void *allocSetScalar<T>::operator()(string sval1, string sval2) {
  T dummy;
  return allocSetScalarFunc(sval1, sval2, std::forward<T>(dummy));
}

template <typename T>
void batchedPtrMagic<T>::operator()(void **hptr, void **dptr, void *dAr,
                                    int batchct, int x, int y) {
  T **host = reinterpret_cast<T **>(hptr);
  T *device_array = static_cast<T *>(dAr);
  for (int i = 0; i < batchct; i++) {
    host[i] = device_array + (i * x * y);
  }
  // checkCuda(cudaMalloc(&dptr, batchct * sizeof(T *)));
  // hptr = reinterpret_cast<void **>(host);
  checkCuda(
      cudaMemcpy(dptr, hptr, batchct * sizeof(T *), cudaMemcpyHostToDevice));
}

template <typename T>
void *allocSetScalarFunc(std::string sval, std::string sval2, T dummy) {
  // Only for real numbers, no need to worry about contents from sval2
  void *ptr = (void *)malloc(sizeof(T));
  T *data = (T *)ptr;
  std::istringstream iss(sval.c_str());
  iss >> *data;
  return ptr;
}

template <typename T>
void *allocSetScalarFunc(std::string sval, std::string sval2,
                         complex<T> dummy) {
  // Complex numbers, do something about sval2
  void *ptr = (void *)malloc(sizeof(complex<T>));
  complex<T> *data = (complex<T> *)ptr;
  T val;
  std::istringstream iss(sval.c_str());
  iss >> val;
  data->real(val);
  std::istringstream iss2(sval2.c_str());
  iss2 >> val;
  data->imag(val);
  return ptr;
}

// template <>
// void *allocSetScalar<complex<float>>::operator()(string sval) {
//  return NULL;
//}
//
// template <>
// void *allocSetScalar<complex<double>>::operator()(string sval) {
//  return NULL;
//}

template <template <typename> class tFunc, class... Args>
auto typeCallHost(cudaDataType_t type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case CUDA_R_64F:
      return tFunc<double>()(args...);
      break;
    case CUDA_C_64F:
      return tFunc<complex<double>>()(args...);
      // return tFunc<cuDoubleComplex>()(args...);
      break;
    case CUDA_R_32F:
      return tFunc<float>()(args...);
      break;
    case CUDA_C_32F:
      return tFunc<complex<float>>()(args...);
      // return tFunc<cuComplex>()(args...);
      break;
    case CUDA_R_16BF:
      return tFunc<float>()(args...);
      break;
    case CUDA_C_16BF:
      return tFunc<complex<float>>()(args...);
      break;
    case CUDA_R_16F:
      return tFunc<float>()(args...);
      break;
    case CUDA_C_16F:
      return tFunc<complex<float>>()(args...);
      break;
    case CUDA_R_8F_E4M3:
      return tFunc<float>()(args...);
      break;
    case CUDA_R_8F_E5M2:
      return tFunc<float>()(args...);
      break;
    case CUDA_R_8I:
      return tFunc<__int8_t>()(args...);
      break;
    case CUDA_C_8I:
      return tFunc<complex<__int8_t>>()(args...);
      break;
    case CUDA_R_8U:
      return tFunc<__uint8_t>()(args...);
      break;
    case CUDA_C_8U:
      return tFunc<complex<__uint8_t>>()(args...);
      break;
    case CUDA_R_32I:
      return tFunc<__int32_t>()(args...);
      break;
    case CUDA_C_32I:
      return tFunc<complex<__int32_t>>()(args...);
      break;
    default:
      return tFunc<double>()(args...);
      ;
  }
}

template <template <typename> class tFunc, class... Args>
auto typeCallDev(cudaDataType_t type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case CUDA_R_64F:
      return tFunc<double>()(args...);
      break;
    case CUDA_C_64F:
      return tFunc<complex<double>>()(args...);
      break;
    case CUDA_R_32F:
      return tFunc<float>()(args...);
      break;
    case CUDA_C_32F:
      return tFunc<complex<float>>()(args...);
      break;
    case CUDA_R_16BF:
      return tFunc<__nv_bfloat16>()(args...);
      break;
    case CUDA_C_16BF:
      return tFunc<complex<__nv_bfloat16>>()(args...);
      break;
    case CUDA_R_16F:
      return tFunc<__half>()(args...);
      break;
    case CUDA_C_16F:
      return tFunc<complex<__half>>()(args...);
      break;
    case CUDA_R_8F_E4M3:
      return tFunc<__nv_fp8_e4m3>()(args...);
      break;
    case CUDA_R_8F_E5M2:
      return tFunc<__nv_fp8_e5m2>()(args...);
      break;
    case CUDA_R_8I:
      return tFunc<__int8_t>()(args...);
      break;
    case CUDA_C_8I:
      return tFunc<complex<__int8_t>>()(args...);
      break;
    case CUDA_R_8U:
      return tFunc<__uint8_t>()(args...);
      break;
    case CUDA_C_8U:
      return tFunc<complex<__uint8_t>>()(args...);
      break;
    case CUDA_R_32I:
      return tFunc<__int32_t>()(args...);
      break;
    case CUDA_C_32I:
      return tFunc<complex<__int32_t>>()(args...);
      break;
    default:
      return tFunc<double>()(args...);
      ;
  }
}

void *allocateHostArr(cudaDataType_t type, long x, long y, int batch,
                      bool strided) {
  int typesize = typeCallHost<sizeofCUDT>(type);
  if (!strided) {
    // Not currently implemented...
    return NULL;
  }
  void *data = (void *)malloc(x * y * batch * typesize);
  return data;
}

void *allocateDevArr(cudaDataType_t type, long x, long y, int batch,
                     bool strided) {
  int typesize = typeCallDev<sizeofCUDT>(type);
  if (!strided) {
    // Not currently implemented...
    return NULL;
  }
  void *data;
  checkCuda(cudaMallocManaged(&data, x * y * batch * typesize));
  return data;
}

void *allocateHDevArr(cudaDataType_t type, long x, long y, int batch,
                      bool strided) {
  int typesize = typeCallHost<sizeofCUDT>(type);
  if (!strided) {
    // Not currently implemented...
    return NULL;
  }
  void *data;
  cudaMallocManaged(&data, x * y * batch * typesize);
  return data;
}

// void *allocateScalar(cudaDataType_t type) {
//  int typesize = typeCallDev<sizeofCUDT>(type);
//  void *scalar = (void *)
//}

template <typename T>
void fillRandHostBlasgemm<T>::operator()(void *ptr, int rows_A, int cols_A,
                                         int ld, int batch,
                                         long long int stride) {
  int a = 1;
  T *A = (T *)ptr;
  for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
    A[i] = (T)rand() / (T)(RAND_MAX / a);
    // if(i < 10)
    //      std::cout << *((double *)ptr+i)  << std::endl;
  }
}

template <typename T>
void fillRandHostConstant<T>::operator()(void *ptr, int rows_A, int cols_A,
                                         int ld, int batch,
                                         long long int stride, float constant) {
  int a = 1;
  T *A = (T *)ptr;
  for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
    A[i] = (T)(constant);
  }
}

template <typename T>
void fillRandHostRandInt<T>::operator()(void *ptr, int rows_A, int cols_A,
                                        int ld, int batch,
                                        long long int stride) {
  T *A = (T *)ptr;
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        A[i + offset] = T(rand() % 10 + 1);
      }
    }
  }
}

template <typename T>
void fillRandHostTrigFloat<T>::operator()(void *ptr, int rows_A, int cols_A,
                                          int ld, int batch,
                                          long long int stride) {
  T *A = (T *)ptr;
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        A[i + offset] = T(cos(i + offset));
      }
    }
  }
}

void dummy() {
  // This function forces the compiler to generate the needed templated variants
  // of each function. It is never called
  void *h_A;
  typeCallHost<fillRandHostBlasgemm>(CUDA_R_64F, h_A, 10, 10, 10, 1, 0);
  typeCallHost<sizeofCUDTP>(CUDA_R_64F);
  typeCallHost<allocSetScalar>(CUDA_R_64F, "1", "0");
  typeCallDev<batchedPtrMagic>(CUDA_R_64F, (void **)NULL, (void **)NULL,
                               (void *)NULL, 10, 10, 10);
  // template void *allocSetScalar<double>::operator()(string);
}

void initHost(cudaDataType_t precision, std::string initialization, void *ptr,
              int rows_A, int cols_A, int ld, int batch, long long int stride,
              float constant) {
  if (initialization == "rand_int") {
    typeCallHost<fillRandHostRandInt>(precision, ptr, rows_A, cols_A, ld, batch,
                                      stride);
  } else if (initialization == "trig_float") {
    typeCallHost<fillRandHostTrigFloat>(precision, ptr, rows_A, cols_A, ld,
                                        batch, stride);
  } else if (initialization == "hpl") {
  } else if (initialization == "blasgemm") {
    typeCallHost<fillRandHostBlasgemm>(precision, ptr, rows_A, cols_A, ld,
                                       batch, stride);
  } else if (initialization == "constant") {
    typeCallHost<fillRandHostConstant>(precision, ptr, rows_A, cols_A, ld,
                                       batch, stride, constant);
  }
}

// int sizeof_cudt_host(cudaDataType_t type) {
//     int size = 0;
//     complex<double> z1(1,1.5);
//     switch(type) {
//         case CUDA_R_64F:
//             size = sizeof(double);
//             break;
//         case CUDA_C_64F:
//             size = sizeof(complex<double>);
//             break;
//         case CUDA_R_32F:
//             size = sizeof(float);
//             break;
//         case CUDA_C_32F:
//             size = sizeof(complex<float>);
//             break;
//         case CUDA_R_16BF:
//             size = sizeof(float);
//             break;
//         case CUDA_C_16BF:
//             size = sizeof(complex<float>);
//             break;
//         case CUDA_R_16F:
//             size = sizeof(float);
//             break;
//         case CUDA_C_16F:
//             size = sizeof(complex<float>);
//             break;
//         case CUDA_R_8F_E4M3:
//             size = sizeof(float);
//             break;
//         case CUDA_R_8F_E5M2:
//             size = sizeof(complex<float>);
//             break;
//         case CUDA_R_8I:
//             size = sizeof(__int8_t);
//             break;
//         case CUDA_C_8I:
//             size = sizeof(complex<__int8_t>);
//             break;
//         case CUDA_R_8U:
//             size = sizeof(__uint8_t);
//             break;
//         case CUDA_C_8U:
//             size = sizeof(complex<__uint8_t>);
//             break;
//         case CUDA_R_32I:
//             size = sizeof(__int32_t);
//             break;
//         case CUDA_C_32I:
//             size = sizeof(complex<__int32_t>);
//             break;
//         default:
//             size = sizeof(float);
//     }
//     return size;
// }
