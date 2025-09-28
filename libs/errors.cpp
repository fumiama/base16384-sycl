#include "errors.hpp"

#include <functional>
#include <iostream>
#include <sycl/sycl.hpp>

template <typename T>
concept has_what_concept_t = requires(T t) { t.what(); };

template <has_what_concept_t T>
void print_what(T e, std::string msg) {
  std::cerr << msg << e.what() << std::endl;
}

errors_code_enum_t failed(std::function<void(void)> fn) {
  try {
    fn();
  } catch (sycl::exception &e) {
    print_what(e, "Caught sync SYCL exception: ");
    return errors_code_sync_sycl_exception;
  } catch (std::exception &e) {
    print_what(e, "Caught std exception: ");
    return errors_code_std_exception;
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    return errors_code_unknown_exception;
  }
  return errors_code_ok;
}
