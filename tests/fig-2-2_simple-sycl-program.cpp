// Figure 2-2. Simple SYCL program
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <array>
#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  constexpr int size = 16;
  std::array<int, size> data;

  // Create queue on implementation-chosen default device
  sycl::queue Q;

  // Create buffer using host allocated "data" array
  sycl::buffer B{data};

  Q.submit([&](sycl::handler& h) {
    sycl::accessor A{B, h};

    h.parallel_for(size, [=](auto& idx) { A[idx] = idx; });
  });

  // Obtain access to buffer on the host
  // Will wait for device kernel to execute to generate data
  sycl::host_accessor A{B};

  for (int i = 0; i < size; i++) {
    std::cout << "data[" << i << "] = " << A[i] << "\n";
    if (A[i] != i) {
      std::cerr << "unexpected data at idx " << i << std::endl;
      return -1;
    }
  }

  return 0;
}