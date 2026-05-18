#pragma once
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <complex>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include "generic_init.h"
#include "mblas_data_type.h"

long long get_malloc_size_host(mblas_data_type type, long x, long y, int batch,
                               long long stride);
long long get_malloc_size_dev(mblas_data_type type, long x, long y, int batch,
                              long long stride);

template <typename T>
struct sizeofCUDT {
  int operator()();
};

template <typename T>
struct sizeofCUDTP {
  int operator()();
};

template <typename T>
struct batchedPtrMagic {
  void operator()(void **hptr, void **dptr, void *hArr, int batch_count, int x,
                  int y);
};

template <typename T>
struct set_scalar {
  void operator()(void *ptr, std::string, std::string);
};

template <template <typename> class tFunc, class... Args>
auto type_call_host(mblas_data_type type, Args... args)
    -> std::invoke_result_t<tFunc<double>, Args...>;

template <template <typename> class tFunc, class... Args>
auto type_call_dev(mblas_data_type type, Args... args)
    -> std::invoke_result_t<tFunc<double>, Args...>;

long get_malloc_size_scalar(mblas_data_type type);

template <typename T>
void set_scalar_val(void *ptr, std::string sval, std::string sval2, T dummy) {
  // Only for real numbers, no need to worry about contents from sval2
  T *data = (T *)ptr;
  std::istringstream iss(sval.c_str());
  iss >> *data;
}

template <typename T>
void set_scalar_val(void *ptr, std::string sval, std::string sval2,
                         std::complex<T> dummy) {
  // Complex numbers, do something about sval2
  std::complex<T> *data = (std::complex<T> *)ptr;
  T val;
  std::istringstream iss(sval.c_str());
  iss >> val;
  data->real(val);
  std::istringstream iss2(sval2.c_str());
  iss2 >> val;
  data->imag(val);
}

template <typename T>
int sizeofCUDT<T>::operator()() {
  return sizeof(T);
}

template <typename T>
int sizeofCUDTP<T>::operator()() {
  return sizeof(T *);
}

template <typename T>
void set_scalar<T>::operator()(void *ptr, std::string sval1, std::string sval2) {
  T dummy;
  return set_scalar_val(ptr, sval1, sval2, std::forward<T>(dummy));
}

template <typename T>
void batchedPtrMagic<T>::operator()(void **hptr, void **dptr, void *dAr,
                                    int batch_count, int x, int y) {
  T **host = reinterpret_cast<T **>(hptr);
  T *device_array = static_cast<T *>(dAr);
  for (int i = 0; i < batch_count; i++) {
    host[i] = device_array + (i * x * y);
  }
  hipMemcpy(dptr, hptr, batch_count * sizeof(T *), hipMemcpyHostToDevice);
}

template <template <typename> class tFunc, class... Args>
auto type_call_host(mblas_data_type type, Args... args)
    -> std::invoke_result_t<tFunc<double>, Args...> {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case MBLAS_R_64F:
      return tFunc<double>()(args...);
    case MBLAS_C_64F:
      return tFunc<std::complex<double>>()(args...);
    case MBLAS_R_32F:
      return tFunc<float>()(args...);
    case MBLAS_C_32F:
      return tFunc<std::complex<float>>()(args...);
    case MBLAS_R_16BF:
      return tFunc<float>()(args...);
    case MBLAS_C_16BF:
      return tFunc<std::complex<float>>()(args...);
    case MBLAS_R_16F:
      return tFunc<float>()(args...);
    case mblas_data_type::MBLAS_R_8F_E4M3:
      return tFunc<float>()(args...);
    case mblas_data_type::MBLAS_R_8F_E5M2:
      return tFunc<float>()(args...);
    case MBLAS_R_8I:
      return tFunc<__int8_t>()(args...);
    case MBLAS_R_8U:
      return tFunc<__uint8_t>()(args...);
    case MBLAS_R_32I:
      return tFunc<__int32_t>()(args...);
    default:
      return tFunc<double>()(args...);
  }
}

template <template <typename> class tFunc, class... Args>
auto type_call_dev(mblas_data_type type, Args... args)
    -> std::invoke_result_t<tFunc<double>, Args...> {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case MBLAS_R_64F:
      return tFunc<double>()(args...);
    case MBLAS_C_64F:
      return tFunc<std::complex<double>>()(args...);
    case MBLAS_R_32F:
      return tFunc<float>()(args...);
    case MBLAS_C_32F:
      return tFunc<std::complex<float>>()(args...);
    case MBLAS_R_16BF:
      return tFunc<hip_bfloat16>()(args...);
    case MBLAS_C_16BF:
      return tFunc<std::complex<hip_bfloat16>>()(args...);
    case MBLAS_R_16F:
      return tFunc<__half>()(args...);
    case mblas_data_type::MBLAS_R_8F_E4M3:
      return tFunc<__hip_fp8_storage_t>()(args...);
    case mblas_data_type::MBLAS_R_8F_E5M2:
      return tFunc<__hip_fp8_storage_t>()(args...);
    case MBLAS_R_8I:
      return tFunc<__int8_t>()(args...);
    case MBLAS_R_8U:
      return tFunc<__uint8_t>()(args...);
    case MBLAS_R_32I:
      return tFunc<__int32_t>()(args...);
    default:
      return tFunc<double>()(args...);
  }
}

