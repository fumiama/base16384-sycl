#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#endif

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>
#include <sycl/sycl.hpp>
#include <vector>

#include "errors.hpp"
#include "test.hpp"
#include "xeinfo.hpp"

constexpr int iter_count = 65536;
constexpr int N = 65536;

int main() {
#ifdef _WIN32
  // Set console code page to UTF-8
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  sycl::queue q;

  const sycl::device device;
  const auto device_name = device.get_info<sycl::info::device::name>();
  std::cout << "执行设备: " << device_name << std::endl;
  std::cout << "设备类型: ";
  if (device.is_cpu()) {
    std::cout << "CPU" << std::endl;
  } else if (device.is_gpu()) {
    std::cout << "GPU" << std::endl;
  } else {
    std::cout << "其他" << std::endl;
  }

  int work_group_size = 64;
  if (device.is_gpu() && device_name.starts_with("Intel")) {
    try {
      auto xeinfo = base16384::xeinfo(device);
      work_group_size = xeinfo.work_group_size;
      std::cout << "\n" << xeinfo.string() << "\n\n";
    } catch (const sycl::exception& e) {
      std::cout << "获取Intel GPU信息失败 (可能不是Intel设备): " << e.what() << std::endl;
      std::cout << "使用默认工作组大小: " << work_group_size << "\n\n";
    }
  }

  // Generate random initial data
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<int> dis{0, 255};

  std::vector<uint8_t> initial_data(N);
  for (auto& byte : initial_data) {
    byte = static_cast<uint8_t>(dis(gen));
  }

  // CPU baseline test
  auto cpu_data = initial_data;

  auto start_time = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < iter_count; j++) {
    for (auto& byte : cpu_data) {
      byte = base16384::test::kernels_basic(byte);
    }
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  std::cout << "CPU (" << std::fixed << std::setprecision(1) << duration.count() / 1000.0
            << " ms):";
  for (int i = 0; i < std::min(N, 64); i++) std::cout << " " << static_cast<int>(cpu_data[i]);
  std::cout << "..." << std::endl;

  auto* data = sycl::malloc_shared<std::uint8_t>(N, q);
  std::copy(initial_data.cbegin(), initial_data.cend(), data);

  // test basic parallel kernel
  start_time = std::chrono::high_resolution_clock::now();
  auto errn = base16384::errors::try_failed([&]() {
    for (int j = 0; j < iter_count; j++) {
      q.parallel_for(sycl::range<1>(N),
                     [=](sycl::id<1> i) { data[i] = base16384::test::kernels_basic(data[i]); });
    }
    q.wait();
  });
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  if (errn) return errn;

  std::cout << "GPU 基本并行 (" << std::fixed << std::setprecision(1) << duration.count() / 1000.0
            << " ms):";
  for (int i = 0; i < std::min(N, 64); i++) std::cout << " " << static_cast<int>(data[i]);
  std::cout << "..." << std::endl;

  // Verify GPU basic parallel result
  for (int i = 0; i < N; i++) {
    if (data[i] != cpu_data[i]) {
      std::cerr << "GPU 基本并行结果验证失败：位置 " << i << " 期望值 "
                << static_cast<int>(cpu_data[i]) << " 实际值 " << static_cast<int>(data[i])
                << std::endl;
      sycl::free(data, q);
      return -1;
    }
  }

  std::copy(initial_data.cbegin(), initial_data.cend(), data);

  start_time = std::chrono::high_resolution_clock::now();
  errn = base16384::errors::try_failed([&]() {
    for (int j = 0; j < iter_count; j++) {
      q.parallel_for(sycl::nd_range<1>(N, work_group_size),
                     [=](sycl::nd_item<1> item) {  // sub-group size
                       const auto i = item.get_global_id(0);
                       data[i] = base16384::test::kernels_basic(data[i]);
                     });
    }
    q.wait();
  });
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  if (errn) return errn;

  std::cout << "GPU 高级并行 (" << std::fixed << std::setprecision(1) << duration.count() / 1000.0
            << " ms):";
  for (int i = 0; i < std::min(N, 64); i++) std::cout << " " << static_cast<int>(data[i]);
  std::cout << "..." << std::endl;

  // Verify GPU advanced parallel result
  for (int i = 0; i < N; i++) {
    if (data[i] != cpu_data[i]) {
      std::cerr << "GPU 高级并行结果验证失败：位置 " << i << " 期望值 "
                << static_cast<int>(cpu_data[i]) << " 实际值 " << static_cast<int>(data[i])
                << std::endl;
      sycl::free(data, q);
      return -1;
    }
  }

  sycl::free(data, q);

  return 0;
}
