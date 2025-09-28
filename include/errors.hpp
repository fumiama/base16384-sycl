#ifndef _ERRORS_HPP_
#define _ERRORS_HPP_

#include <functional>

typedef enum {
  errors_code_ok,
  errors_code_sync_sycl_exception,
  errors_code_std_exception,
  errors_code_unknown_exception,
} errors_code_enum_t;

// failed try to exec fn, catch and print .what() when exception is thrown.
errors_code_enum_t failed(std::function<void(void)> fn);

#endif
