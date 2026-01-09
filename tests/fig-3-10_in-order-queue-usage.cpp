// Figure 3-10. In-order queue usage
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <sycl/sycl.hpp>

constexpr int N = 4;
int main() {
  sycl::queue Q{sycl::property::queue::in_order()};
  int* device_array = sycl::malloc_device<int>(N, Q);

  // Task A
  Q.submit(
      [&](sycl::handler& h) { h.parallel_for(N, [=](sycl::id<1> i) { device_array[i] = 0; }); });
  // Task B
  Q.submit([&](sycl::handler& h) { h.parallel_for(N, [=](sycl::id<1> i) { device_array[i]++; }); });
  // Task C
  Q.submit(
      [&](sycl::handler& h) { h.parallel_for(N, [=](sycl::id<1> i) { device_array[i] <<= 2; }); });

  std::array<int, N> host_array;
  Q.submit([&](sycl::handler& h) {
    // copy deviceArray back to hostArray
    h.memcpy(&host_array[0], device_array, N * sizeof(int));
  });

  Q.wait();

  sycl::free(device_array, Q);

  for (int i = 0; i < host_array.size(); i++) {
    if (host_array[i] != 4) {
      std::cerr << "Expect 4 at idx " << i << " but got " << host_array[i] << std::endl;
      return -1;
    }
  }

  std::cout << "Test Passed!!!" << std::endl;

  return 0;
}