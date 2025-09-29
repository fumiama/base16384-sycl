#ifndef _ERRORS_HPP_
#define _ERRORS_HPP_

#include <functional>
#include <iostream>
#include <sycl/sycl.hpp>

template <typename T>
concept base16384_has_what_concept_t = requires(T t) { t.what(); };

template <typename F>
concept base16384_callable_concept_t = requires(F f) { f(); };

namespace base16384 {
class errors {
 private:
  errors() = default;

  template <base16384_has_what_concept_t T>
  static void print_what(T e, std::string msg) {
    std::cerr << msg << e.what() << std::endl;
  };

 public:
  errors(const errors &) = delete;
  errors(errors &&) = delete;
  errors &operator=(const errors &) = delete;
  errors &operator=(errors &&) = delete;
  auto operator<=>(const errors &) const = delete;
  ~errors() noexcept = default;

  typedef enum {
    code_ok,
    code_sync_sycl_exception,
    code_std_exception,
    code_unknown_exception,
  } code_enum_t;

  // failed try to exec fn, catch and print .what() when exception is thrown.
  template <base16384_callable_concept_t F>
  static code_enum_t try_failed(F &&fn) {
    try {
      fn();
    } catch (sycl::exception &e) {
      print_what(e, "Caught sync SYCL exception: ");
      return code_sync_sycl_exception;
    } catch (std::exception &e) {
      print_what(e, "Caught std exception: ");
      return code_std_exception;
    } catch (...) {
      std::cerr << "Caught unknown exception." << std::endl;
      return code_unknown_exception;
    }
    return code_ok;
  };
};

}  // namespace base16384

#endif
