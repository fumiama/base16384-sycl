// Figure 3-11. Using events and depends_on
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <sycl/sycl.hpp>

constexpr int N = 4;
int main() {
  sycl::queue Q;
  std::array<int, N> data1;
  sycl::buffer B1{data1};
  std::array<int, N> data2;
  sycl::buffer B2{data2};

  // Task A
  auto eA = Q.submit([&](sycl::handler& h) {
    sycl::accessor A1{B1, h};
    sycl::accessor A2{B2, h};

    h.parallel_for(N, [=](sycl::id<1> i) {
      A1[i] = 233;
      A2[i] = 666;
    });
  });
  eA.wait();
  // Task B
  auto eB = Q.submit([&](sycl::handler& h) {
    sycl::accessor A1{B1, h};
    sycl::accessor A2{B2, h};

    h.parallel_for(N, [=](sycl::id<1> i) {
      A1[i] += i;      // 233 234 235 236
      A2[i] += A1[i];  // 899 900 901 902
    });
  });
  // Task C
  auto eC = Q.submit([&](sycl::handler& h) {
    sycl::accessor A2{B2, h};

    h.depends_on(eB);
    h.parallel_for(N, [=](sycl::id<1> i) {
      A2[i] <<= 1;  // 1798 1800 1802 1804
    });
  });
  // Task D
  auto eD = Q.submit([&](sycl::handler& h) {
    sycl::accessor A1{B1, h};
    sycl::accessor A2{B2, h};

    h.depends_on({eB, eC});
    h.parallel_for(N, [=](sycl::id<1> i) {
      A2[i] += A1[i] * i;  // 1798 2034 2272 2512
    });
  });

  std::array<int, N> expected{1798, 2034, 2272, 2512};
  sycl::host_accessor A2{B2};  // if use data2 directly, the data may have not been synced

  for (int i = 0; i < expected.size(); i++) {
    if (A2[i] != expected[i]) {
      std::cerr << "Expect " << expected[i] << " at idx " << i << " but got " << A2[i] << std::endl;
      return -1;
    }
  }

  std::cout << "Test Passed!!!" << std::endl;

  return 0;
}