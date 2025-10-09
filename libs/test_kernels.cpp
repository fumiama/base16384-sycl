#include <iostream>
#include <sycl/sycl.hpp>

#include "test.hpp"

SYCL_EXTERNAL std::uint8_t base16384::test::kernels_basic(uint8_t in) {
  in *= in;
  in %= 251;
  in ^= in >> 2;
  in += 17;
  in *= 3;
  return in ^ (in << 1);
}