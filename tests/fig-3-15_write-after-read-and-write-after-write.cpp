// Figure 3-15. Write-after-Read and Write-after-Write
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <sycl/sycl.hpp>

constexpr int N = 42;
int main() {
  std::array<int, N> a, b;
  for (int i = 0; i < N; i++) {
    a[i] = b[i] = 0;
  }
  sycl::queue Q;
  sycl::buffer A{a};
  sycl::buffer B{b};
  Q.submit([&](sycl::handler& h) {
    sycl::accessor accA(A, h, sycl::read_only);
    sycl::accessor accB(B, h, sycl::write_only);
    h.parallel_for(  // computeB
        N, [=](sycl::id<1> i) { accB[i] = accA[i] + 1; });
  });
  Q.submit([&](sycl::handler& h) {
    // WAR of buffer A
    sycl::accessor accA(A, h, sycl::write_only);
    h.parallel_for(  // rewriteA
        N, [=](sycl::id<1> i) { accA[i] = 21; });
  });
  Q.submit([&](sycl::handler& h) {
    // WAW of buffer B
    sycl::accessor accB(B, h, sycl::write_only);
    h.parallel_for(  // rewriteB
        N, [=](sycl::id<1> i) { accB[i] = 30; });
  });
  sycl::host_accessor host_accA(A, sycl::read_only);
  sycl::host_accessor host_accB(B, sycl::read_only);
  for (int i = 0; i < N; i++) {
    if (host_accA[i] != 21) {
      std::cerr << "Expect host_accA[i] 21 at idx " << i << " but got " << host_accA[i]
                << std::endl;
      return -1;
    }
    if (host_accB[i] != 30) {
      std::cerr << "Expect host_accB[i] 30 at idx " << i << " but got " << host_accB[i]
                << std::endl;
      return -1;
    }
  }
  std::cout << "Test Passed!!!" << std::endl;
  return 0;
}