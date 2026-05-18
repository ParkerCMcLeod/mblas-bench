#pragma once
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "cxxopts.hpp"

enum class scaling_type {None, Scalar, Vector, Block};
std::string scaling_string(scaling_type input);

enum class timing_mode { serialized, pipelined };

struct gemm_inst_base {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  void **ptr_dev_a = nullptr;
  void **ptr_dev_b = nullptr;
  void **ptr_dev_c = nullptr;
  void **ptr_dev_d = nullptr;
  void *devWork = nullptr;
  long wSZ = 0;
  gemm_inst_base(int devID) : devIDX(devID) {}
};

class generic_gemm {
 protected:
  struct matrix_desc {
    int rows;
    int cols;
    int rows_mem;
    int cols_mem;
    long long int stride{0};
    bool control{false};
    float constant;
    float scale_factor;
    scaling_type scale_mode = scaling_type::None;
    std::string init;
  };

  matrix_desc a_props;
  matrix_desc b_props;
  matrix_desc c_props;
  matrix_desc d_props;

  int m;
  int n;
  int k;

  int lda;
  int ldb;
  int ldc;
  int ldd;

  bool strided = false;
  bool batched = false;
  bool pure_batched = false;

  int iters;
  int cold_iters;

  int iters_time_ms = 0;
  int cold_iters_time_ms = 0;

  timing_mode timing = timing_mode::pipelined;

  int batch_count;
  int flush_batch_count;
  int flush_memory_size;

  std::string filename_a;
  std::string filename_b;
  std::string filename_c;
  std::string filename_d;

  double gflop_per_second = 0;
  double gbyte_per_second = 0;
  double iter_time_us = 0;

  float avg_sysclk_mhz = 0;
  float med_sysclk_mhz = 0;
  float avg_memclk_mhz = 0;
  float med_memclk_mhz = 0;

  std::string function;

  std::string initialization;
  std::string scale_init;

 public:
  virtual ~generic_gemm() = default;
  generic_gemm(cxxopts::ParseResult);

  int set_ld(std::string ld, std::string OP, int x, int y);
  std::pair<int, int> set_row_col(std::string OP, int d1, int d2);

  virtual std::string prepare_array() = 0;

  virtual double test() = 0;

  virtual std::string get_result_string() = 0;
  virtual void free_mem() = 0;

  static long long int fix_stride(long long int stride, long rows_a, long cols_a, std::string matrix_id);
  static scaling_type set_scale_mode(std::string input);
  static std::string set_init(matrix_desc desc, std::string init, std::string mx_init);

  struct matrix_alloc_desc {
    uint64_t rows_mem, cols_mem;
    int type_size, type_pack;
  };

  void set_flush_batch_count(
      const matrix_alloc_desc& a, const matrix_alloc_desc& b,
      const matrix_alloc_desc& c, const matrix_alloc_desc& d,
      bool inplace);

  std::tuple<double, double, double> calculate_figure_of_merit(
      double totalTime_ms, int iters_completed,
      int a_sz, int b_sz, int out_sz,
      int a_pack, int b_pack, int out_pack,
      bool is_real);

 protected:
  template <typename InstType>
  void parse_dev_iters_impl(const std::string &deviceStr,
                            std::vector<InstType> &mat_ptrs) {
    std::stringstream ss(deviceStr);
    while (ss.good()) {
      std::string deviceSStr;
      std::getline(ss, deviceSStr, ',');
      int devInt = std::stoi(deviceSStr);
      mat_ptrs.push_back(InstType(devInt));
    }
  }

  template <typename Derived, typename InstType>
  void run_threaded_impl(void (Derived::*func)(InstType *),
                         std::vector<InstType> &mat_ptrs) {
    std::vector<std::thread> threads;
    for (auto &instance : mat_ptrs) {
      threads.push_back(
          std::thread(func, static_cast<Derived *>(this), &instance));
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }

  template <typename InstType>
  void accumulate_results(const std::vector<InstType> &mat_ptrs) {
    gflop_per_second = 0;
    gbyte_per_second = 0;
    iter_time_us = 0;
    for (const auto &inst : mat_ptrs) {
      gflop_per_second += inst.gflops;
      gbyte_per_second += inst.gbytes;
      iter_time_us += inst.time_us;
    }
    if (!mat_ptrs.empty()) {
      iter_time_us /= static_cast<double>(mat_ptrs.size());
    }
  }

};

