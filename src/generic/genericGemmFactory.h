#pragma once 
#include <genericGemm.h>
#include <cxxopts.hpp>
#include <iostream>

class genericGemmFactory {
 public:
  virtual ~genericGemmFactory(){};
  virtual void createGemm(cxxopts::ParseResult) = 0;
  /**
   * Also note that, despite its name, the Creator's primary responsibility is
   * not creating products. Usually, it contains some core business logic that
   * relies on Product objects, returned by the factory method. Subclasses can
   * indirectly change that business logic by overriding the factory method and
   * returning a different type of product from it.
   */
 protected:
  genericGemm * gemm;

 public:
  std::string prepareArray() { return gemm->prepareArray(); } 
  void test() { gemm->test(); } 
  std::string getResultString() { return gemm->getResultString(); } 
  void freeMem() { gemm->freeMem(); } 
};