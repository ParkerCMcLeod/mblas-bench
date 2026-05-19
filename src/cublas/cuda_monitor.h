#pragma once

#include "cuda_error.h"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>
#include <nvml.h>


namespace cuda_monitor {

class monitor
{
private:
    static constexpr double MHz_TO_Hz = 1000000.0;
    static constexpr double Hz_TO_MHz = 1.0 / MHz_TO_Hz;
    static constexpr int SAMPLING_INTERVAL_MS = 50;

    std::thread monitoring_thread;
    std::atomic<bool> should_stop{false};
    mutable std::mutex data_mutex;

    nvmlDevice_t nvml_device{};
    int physical_device_id = 0;

    std::vector<uint64_t> gpu_frequencies;
    std::vector<uint64_t> mem_frequencies;
    uint64_t gpu_freq_sum = 0;
    uint64_t mem_freq_sum = 0;

public:
    monitor(const monitor&) = delete;
    monitor& operator=(const monitor&) = delete;

    monitor() { init_nvml(); }

    ~monitor() { stop(); }

    void set_device_id(int device_id) {
        physical_device_id = device_id;
        check_nvml(nvmlDeviceGetHandleByIndex(physical_device_id, &nvml_device));
    }

    static bool enabled() {
        return std::getenv("CUBLAS_BENCH_FREQ") != nullptr;
    }

    void start() {
        if (!enabled()) return;
        if (monitoring_thread.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(data_mutex);
            gpu_frequencies.clear();
            mem_frequencies.clear();
            gpu_freq_sum = 0;
            mem_freq_sum = 0;
        }
        should_stop = false;
        monitoring_thread = std::thread([this] { collect(); });
    }

    void stop() {
        if (!enabled()) return;
        should_stop = true;
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
    }

    float get_avg_sysclk_mhz() const {
        std::lock_guard<std::mutex> lk(data_mutex);
        if (gpu_frequencies.empty()) return 0.0f;
        return (static_cast<float>(gpu_freq_sum) / gpu_frequencies.size()) * Hz_TO_MHz;
    }

    float get_med_sysclk_mhz() const {
        std::lock_guard<std::mutex> lk(data_mutex);
        return median_mhz(gpu_frequencies);
    }

    float get_avg_memclk_mhz() const {
        std::lock_guard<std::mutex> lk(data_mutex);
        if (mem_frequencies.empty()) return 0.0f;
        return (static_cast<float>(mem_freq_sum) / mem_frequencies.size()) * Hz_TO_MHz;
    }

    float get_med_memclk_mhz() const {
        std::lock_guard<std::mutex> lk(data_mutex);
        return median_mhz(mem_frequencies);
    }

private:
    static void init_nvml() {
        static std::once_flag once;
        std::call_once(once, [] { check_nvml(nvmlInit()); });
    }

    static float median_mhz(const std::vector<uint64_t>& freqs) {
        if (freqs.empty()) return 0.0f;
        auto copy = freqs;
        std::sort(copy.begin(), copy.end());
        size_t n = copy.size();
        double median_hz = (n % 2 == 0)
            ? (copy[n/2 - 1] + copy[n/2]) / 2.0
            : static_cast<double>(copy[n/2]);
        return static_cast<float>(median_hz * Hz_TO_MHz);
    }

    void collect() {
        while (!should_stop) {
            unsigned int gpu_clock = 0, mem_clock = 0;
            nvmlReturn_t gpu_result = nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_GRAPHICS, &gpu_clock);
            nvmlReturn_t mem_result = nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_MEM, &mem_clock);
            {
                std::lock_guard<std::mutex> lk(data_mutex);
                if (gpu_result == NVML_SUCCESS) {
                    uint64_t hz = static_cast<uint64_t>(gpu_clock) * static_cast<uint64_t>(MHz_TO_Hz);
                    gpu_frequencies.push_back(hz);
                    gpu_freq_sum += hz;
                }
                if (mem_result == NVML_SUCCESS) {
                    uint64_t hz = static_cast<uint64_t>(mem_clock) * static_cast<uint64_t>(MHz_TO_Hz);
                    mem_frequencies.push_back(hz);
                    mem_freq_sum += hz;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(SAMPLING_INTERVAL_MS));
        }
    }
};

} // namespace cuda_monitor
