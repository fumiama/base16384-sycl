#ifndef _ERRORS_HPP_
#define _ERRORS_HPP_

#include <functional>
#include <iostream>
#include <sycl/sycl.hpp>

typedef enum {
  base16384_errors_code_ok,
  base16384_errors_code_sync_sycl_exception,
  base16384_errors_code_std_exception,
  base16384_errors_code_unknown_exception,
} base16384_errors_code_enum_t;

template <typename T>
concept base16384_has_what_concept_t = requires(T t) { t.what(); };

template <base16384_has_what_concept_t T>
static void base16384_print_what(T e, std::string msg) {
  std::cerr << msg << e.what() << std::endl;
}

// failed try to exec fn, catch and print .what() when exception is thrown.
static base16384_errors_code_enum_t base16384_try_failed(std::function<void(void)> fn) {
  try {
    fn();
  } catch (sycl::exception &e) {
    base16384_print_what(e, "Caught sync SYCL exception: ");
    return base16384_errors_code_sync_sycl_exception;
  } catch (std::exception &e) {
    base16384_print_what(e, "Caught std exception: ");
    return base16384_errors_code_std_exception;
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    return base16384_errors_code_unknown_exception;
  }
  return base16384_errors_code_ok;
}

#endif
