// Figure 3-4. USM explicit data movement
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <array>
#include <sycl/sycl.hpp>

constexpr int N = 42;

int main() {
  sycl::queue Q;

  std::array<int, N> host_array;
  int *device_array = sycl::malloc_device<int>(N, Q);

  for (int i = 0; i < N; i++) {
    host_array[i] = N;
  }

  // We will learn how to simplify this example later
  Q.submit([&](sycl::handler &h) {
     // copy hostArray to deviceArray
     h.memcpy(device_array, &host_array[0], N * sizeof(int));
   }).wait();

  Q.submit([&](sycl::handler &h) {
     h.parallel_for(N, [=](sycl::id<1> i) { device_array[i]++; });
   }).wait();

  Q.submit([&](sycl::handler &h) {
     // copy deviceArray back to hostArray
     h.memcpy(&host_array[0], device_array, N * sizeof(int));
   }).wait();

  sycl::free(device_array, Q);

  for (int i = 0; i < host_array.size(); i++) {
    if (host_array[i] != N + 1) {
      std::cerr << "Expect " << N + 1 << " at idx " << i << " but got " << host_array[i]
                << std::endl;
      return -1;
    }
  }

  return 0;
}