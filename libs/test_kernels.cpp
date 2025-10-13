#include <iostream>
#include <sycl/sycl.hpp>

#include "test/kernels.hpp"

SYCL_EXTERNAL std::uint8_t base16384::test::kernels_basic(uint8_t in) {
  in *= in;
  in %= (uint8_t)251;
  in ^= in >> 2;
  in += (uint8_t)17;
  in *= (uint8_t)3;
  return in ^ (in << 1);
}
