#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cuda/std/complex>
#include <string>

// int sizeof_cudt_host(cudaDataType_t type);
void *allocateHostArr(cudaDataType_t type, long x, long y, int batch = 1,
                      bool strided = true);
void *allocateDevArr(cudaDataType_t type, long x, long y, int batch = 1,
                     bool strided = true);
void *allocateHDevArr(cudaDataType_t type, long x, long y, int batch = 1,
                      bool strided = true);

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
  void operator()(void **hptr, void **dptr, void *hArr, int batchct, int x,
                  int y);
};

template <typename T>
struct allocSetScalar {
  void *operator()(std::string, std::string);
};

template <typename T>
void *allocSetScalarFunc(std::string, std::string, T);

template <typename T>
void *allocSetScalarFunc(std::string, std::string, cuda::std::complex<T>);

template <typename T>
struct fillRandHostBlasgemm {
  void operator()(void *ptr, int rows_A, int cols_A, int ld, int batch,
                  long long int stride);
};

template <typename T>
struct fillRandHostRandInt {
  void operator()(void *ptr, int rows_A, int cols_A, int ld, int batch,
                  long long int stride);
};

template <typename T>
struct fillRandHostTrigFloat {
  void operator()(void *ptr, int rows_A, int cols_A, int ld, int batch,
                  long long int stride);
};

template <template <typename> class tFunc, class... Args>
auto typeCallHost(cudaDataType_t type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type;

template <template <typename> class tFunc, class... Args>
auto typeCallDev(cudaDataType_t type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type;

void initHost(cudaDataType_t precision, std::string initialization, void *ptr,
              int rows_A, int cols_A, int ld, int batch, long long int stride);