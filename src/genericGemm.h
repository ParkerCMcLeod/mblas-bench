#pragma once
#include <string>

#include "third_party/cxxopts.hpp"

class genericGemm {
 protected:
  int m;
  int n;
  int k;

  int lda;
  int ldb;
  int ldc;
  int ldd;

  long long int stride_a;
  long long int stride_b;
  long long int stride_c;
  long long int stride_d;

  bool strided;
  bool batched;
  int batchct;

  int iters;
  int cold_iters;

  std::string function;

 public:
  genericGemm(cxxopts::ParseResult);

  // virtual void setSize();
  // virtual void setTypes();
  int setLd(std::string ld, std::string OP, int x, int y);

  virtual void prepareArray() {}
  virtual void allocHost() {}
  virtual void allocDev() {}
  virtual void fillHost() {}
  virtual void copyHostToDev() {}

  virtual double test() { return 0.0; }

  virtual void freeMem() {}
};
