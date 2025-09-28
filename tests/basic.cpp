#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

#include "errors.hpp"

static const int N = 4;

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

  int *data = sycl::malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  auto errn = failed([&]() {
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
       for (int i = 0; i < N; i++) {
         data[i] *= 2;
       }
     }).wait();
  });

  if (errn) return errn;

  for (int i = 0; i < N; i++) std::cout << data[i] << std::endl;

  sycl::free(data, q);

  return 0;
}
