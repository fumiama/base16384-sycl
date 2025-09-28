#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

#include "errors.hpp"

static const int N = 65536;
static const int work_group_size = 64;

int main() {
#ifdef _WIN32
  // Set console code page to UTF-8
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  sycl::queue q;

  auto device = q.get_device();
  std::cout << "执行设备: " << device.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "设备类型: ";
  if (device.is_cpu()) {
    std::cout << "CPU" << std::endl;
  } else if (device.is_gpu()) {
    std::cout << "GPU" << std::endl;
  } else {
    std::cout << "其他" << std::endl;
  }

  // CPU baseline test
  std::vector<int> cpu_data(N);
  for (int i = 0; i < N; i++) cpu_data[i] = i;

  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) cpu_data[i] *= 2;
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  std::cout << "CPU (" << duration.count() << " us):" << std::endl;
  for (int i = 0; i < min(N, 64); i++) std::cout << " " << cpu_data[i];
  std::cout << "..." << std::endl;

  int *data = sycl::malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  // test basic parallel kernel
  start_time = std::chrono::high_resolution_clock::now();
  auto errn = base16384_try_failed(
      [&]() { q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { data[i] *= 2; }).wait(); });
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  if (errn) return errn;

  std::cout << "GPU基本并行 (" << duration.count() << " us):" << std::endl;
  for (int i = 0; i < min(N, 64); i++) std::cout << " " << data[i];
  std::cout << "..." << std::endl;

  start_time = std::chrono::high_resolution_clock::now();
  errn = base16384_try_failed([&]() {
    q.parallel_for(sycl::nd_range<1>(N, work_group_size), [=](sycl::nd_item<1> item) {
       int i = item.get_global_id(0);
       data[i] /= 2;
     }).wait();
  });
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  if (errn) return errn;

  std::cout << "GPU高级并行 (" << duration.count() << " us):" << std::endl;
  for (int i = 0; i < min(N, 64); i++) std::cout << " " << data[i];
  std::cout << "..." << std::endl;

  sycl::free(data, q);

  return 0;
}
