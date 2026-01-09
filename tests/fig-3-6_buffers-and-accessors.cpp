// Figure 3-6. Buffers and accessors
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <array>
#include <sycl/sycl.hpp>

constexpr int N = 42;

int main() {
  std::array<int, N> my_data{};  // filled with 0
  {
    sycl::queue q;
    sycl::buffer my_buffer(my_data);

    q.submit([&](sycl::handler& h) {
      // create an accessor to update
      // the buffer on the device
      sycl::accessor my_accessor(my_buffer, h);

      h.parallel_for(N, [=](sycl::id<1> i) { my_accessor[i]++; });
    });

    // create host accessor
    sycl::host_accessor host_accessor(my_buffer);

    std::cout << "host_accessor: ";
    for (int i = 0; i < N; i++) {
      // access myBuffer on host
      std::cout << host_accessor[i] << " ";
    }
    std::cout << "\nmy_data_outsc: ";
  }

  // myData is updated when myBuffer is
  // destroyed upon exiting scope
  for (int i = 0; i < N; i++) {
    std::cout << my_data[i] << " ";
    if (my_data[i] != 1) {
      std::cout << "Error at index " << i << ": expected " << 1 << ", got " << my_data[i]
                << std::endl;
      return 1;
    }
  }

  std::cout << "\nTest Passed!!!" << std::endl;
}