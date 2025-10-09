#ifndef _TEST_KERNELS_H_
#define _TEST_KERNELS_H_

#include <stdint.h>

namespace base16384 {
class test {
 public:
  // base16384_test_kernels_basic is a demo calculation that implements
  // mod, bit, plus and mul calculations.
  SYCL_EXTERNAL static uint8_t kernels_basic(uint8_t in);
};
}  // namespace base16384

#endif