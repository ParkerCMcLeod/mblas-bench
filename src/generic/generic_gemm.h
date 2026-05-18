#pragma once
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

  int & rows_a = a_props.rows;
  int & cols_a = a_props.cols;
  int & rows_b = b_props.rows;
  int & cols_b = b_props.cols;
  int & rows_c = c_props.rows;
  int & cols_c = c_props.cols;
  int & rows_d = d_props.rows;
  int & cols_d = d_props.cols;

  int & rows_mem_a = a_props.rows_mem;
  int & cols_mem_a = a_props.cols_mem;
  int & rows_mem_b = b_props.rows_mem;
  int & cols_mem_b = b_props.cols_mem;
  int & rows_mem_c = c_props.rows_mem;
  int & cols_mem_c = c_props.cols_mem;
  int & rows_mem_d = d_props.rows_mem;
  int & cols_mem_d = d_props.cols_mem;

  int lda;
  int ldb;
  int ldc;
  int ldd;

  long long int & stride_a = a_props.stride;
  long long int & stride_b = b_props.stride;
  long long int & stride_c = c_props.stride;
  long long int & stride_d = d_props.stride;

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

  bool & control_a = a_props.control;
  bool & control_b = b_props.control;
  bool & control_c = c_props.control;
  bool & control_d = d_props.control;

  float & constant_a = a_props.constant;
  float & constant_b = b_props.constant;
  float & constant_c = c_props.constant;
  float & constant_d = d_props.constant;

  float & scale_factor_a = a_props.scale_factor;
  float & scale_factor_b = b_props.scale_factor;
  float & scale_factor_c = c_props.scale_factor;
  float & scale_factor_d = d_props.scale_factor;

  scaling_type & scale_mode_a = a_props.scale_mode;
  scaling_type & scale_mode_b = b_props.scale_mode;
  scaling_type & scale_mode_c = c_props.scale_mode;
  scaling_type & scale_mode_d = d_props.scale_mode;

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

  void set_flush_batch_count(int a_type_size,  int b_type_size, int c_type_size, int d_type_size,
                        int a_type_packing,  int b_type_packing, int c_type_packing, int d_type_packing,
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

