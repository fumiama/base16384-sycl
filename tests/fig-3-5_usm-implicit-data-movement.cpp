// Figure 3-5. USM implicit data movement
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <sycl/sycl.hpp>

constexpr int N = 42;

int main() {
  sycl::queue Q;

  int* host_array = malloc_host<int>(N, Q);
  int* shared_array = malloc_shared<int>(N, Q);

  for (int i = 0; i < N; i++) {
    // Initialize hostArray on host
    host_array[i] = i;
  }

  // We will learn how to simplify this example later
  Q.submit([&](sycl::handler& h) {
     h.parallel_for(N, [=](sycl::id<1> i) {
       // access sharedArray and hostArray on device
       shared_array[i] = host_array[i] + 1;
     });
   }).wait();

  for (int i = 0; i < N; i++) {
    // Verify that sharedArray[i] equals hostArray[i] + 1
    if (shared_array[i] != host_array[i] + 1) {
      std::cout << "Error at index " << i << ": expected " << (host_array[i] + 1) << ", got "
                << shared_array[i] << std::endl;
      free(shared_array, Q);
      free(host_array, Q);
      return 1;
    }
  }

  free(shared_array, Q);
  free(host_array, Q);

  std::cout << "Test Passed!!!" << std::endl;

  return 0;
}
