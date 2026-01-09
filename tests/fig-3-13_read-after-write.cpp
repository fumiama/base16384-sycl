// Figure 3-13. Read-after-Write
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <sycl/sycl.hpp>

constexpr int N = 42;
int main() {
  std::array<int, N> a, b, c;
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = c[i] = 0;
  }
  sycl::queue Q;
  // We will learn how to simplify this example later
  sycl::buffer A{a};
  sycl::buffer B{b};
  sycl::buffer C{c};
  Q.submit([&](sycl::handler& h) {
    sycl::accessor accA(A, h, sycl::read_only);
    sycl::accessor accB(B, h, sycl::write_only);
    h.parallel_for(  // computeB
        N, [=](sycl::id<1> i) { accB[i] = accA[i] + 1; });
  });
  int* datap = static_cast<int*>(sycl::malloc_shared(sizeof(int), Q));
  Q.submit([&](sycl::handler& h) {
    sycl::accessor accA(A, h, sycl::read_only);

    h.parallel_for(  // readA
        N, [=](sycl::id<1> i) {
          // Useful only as an example
          *datap = accA[i];
        });
  });
  Q.submit([&](sycl::handler& h) {
    // RAW of buffer B
    sycl::accessor accB(B, h, sycl::read_only);
    sycl::accessor accC(C, h, sycl::write_only);
    h.parallel_for(  // computeC
        N, [=](sycl::id<1> i) { accC[i] = accB[i] + 3; });
  });
  // read C on host
  sycl::host_accessor host_accC(C, sycl::read_only);
  for (int i = 0; i < N; i++) {
    if (host_accC[i] != 5) {
      std::cerr << "Expect 5 at idx " << i << " but got " << host_accC[i] << std::endl;
      return -1;
    }
  }
  std::cout << "readA: " << *datap << "\n";
  std::cout << "Test Passed!!!" << std::endl;
  return 0;
}