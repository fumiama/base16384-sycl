// Figure 2-9. Selecting the host device using the host_selector class
// from book - Data Parallel C++
// https://link.springer.com/book/10.1007/978-1-4842-5574-2

#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  // Create queue to use the host device explicitly
  sycl::queue Q{sycl::cpu_selector_v};

  std::cout << "Selected device: " << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << " -> Vendor: " << Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  std::cout << " -> Backend: "
            << Q.get_device().get_platform().get_info<sycl::info::platform::name>() << std::endl;

  auto device_type = Q.get_device().get_info<sycl::info::device::device_type>();
  if (device_type != sycl::info::device_type::cpu) {
    std::cerr << "Error: Selected device is not a CPU device" << std::endl;
    return -1;
  }

  std::cout << "Device is CPU: OK" << std::endl;

  std::cout << "Test Passed!!!" << std::endl;

  return 0;
}