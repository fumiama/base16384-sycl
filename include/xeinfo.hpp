#ifndef _XEINFO_HPP_
#define _XEINFO_HPP_

#include <stdint.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace base16384 {

class xeinfo {
 private:
  std::pair<size_t, int> calculate_optimal_sizes() const {
    size_t best_sub_group_size = sub_group_sizes[0];
    int best_work_group_size = 0;

    for (auto sg_size : sub_group_sizes) {
      int wg_size = num_thread_per_xecore * sg_size;
      if (wg_size <= max_work_group_size && wg_size > best_work_group_size) {
        best_sub_group_size = sg_size;
        best_work_group_size = 1 << (31 - __builtin_clz(static_cast<unsigned>(wg_size)));
      }
    }

    return {std::move(best_sub_group_size), std::move(best_work_group_size)};
  }

 public:
  xeinfo(sycl::device device) noexcept
      : num_slices(device.get_info<sycl::ext::intel::info::device::gpu_slices>()),
        num_subslices_per_slice(
            device.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>()),
        num_eus_per_subslice(
            device.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>()),
        num_threads_per_eu(
            device.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>()),
        global_mem_size(device.get_info<sycl::info::device::global_mem_size>()),
        local_mem_size(device.get_info<sycl::info::device::local_mem_size>()),
        max_work_group_size(device.get_info<sycl::info::device::max_work_group_size>()),
        sub_group_sizes(device.get_info<sycl::info::device::sub_group_sizes>()),
        num_thread_per_xecore(num_eus_per_subslice * num_threads_per_eu),
        total_xecores(num_slices * num_subslices_per_slice),
        total_vector_engines(num_slices * num_subslices_per_slice * num_eus_per_subslice),
        total_hardware_threads(num_slices * num_subslices_per_slice * num_eus_per_subslice *
                               num_threads_per_eu),
        optimal_sizes(calculate_optimal_sizes()),
        sub_group_size(optimal_sizes.first),
        work_group_size(optimal_sizes.second) {}

  xeinfo(const xeinfo&) = delete;
  xeinfo(xeinfo&&) = delete;
  xeinfo& operator=(const xeinfo&) = delete;
  xeinfo& operator=(xeinfo&&) = delete;
  auto operator<=>(const xeinfo&) const = delete;
  ~xeinfo() noexcept = default;

  const int num_slices;
  const int num_subslices_per_slice;
  const int num_eus_per_subslice;
  const int num_threads_per_eu;
  const uint64_t global_mem_size;
  const int local_mem_size;
  const int max_work_group_size;
  const std::vector<size_t> sub_group_sizes;
  const int num_thread_per_xecore;
  const int total_xecores;
  const int total_vector_engines;
  const int total_hardware_threads;

 private:
  const std::pair<size_t, int> optimal_sizes;

 public:
  const size_t sub_group_size;
  const int work_group_size;

  std::string string() const {
    std::ostringstream builder;
    builder << "Intel GPU 特性:\n";
    builder << "  XeCore 数量: " << total_xecores << "\n";
    builder << "  每个 XeCore 的向量引擎数: " << num_eus_per_subslice << "\n";
    builder << "  向量引擎总数: " << total_vector_engines << "\n";
    builder << "  每个 XeCore 的硬件线程数: " << num_thread_per_xecore << "\n";
    builder << "  每个向量引擎的硬件线程数: " << num_threads_per_eu << "\n";
    builder << "  硬件线程总数: " << total_hardware_threads << "\n";
    builder << "  GPU 内存大小: " << global_mem_size << " B (" << std::fixed << std::setprecision(2)
            << (double)global_mem_size / 1024 / 1024 / 1024 << " GB)\n";
    builder << "  每个工作组的共享本地内存: " << local_mem_size << " B\n";
    builder << "  最大工作组大小: " << max_work_group_size << "\n";
    builder << "  支持的子组大小:";
    for (size_t i = 0; i < sub_group_sizes.size(); i++) builder << " " << sub_group_sizes[i];
    builder << "\n";
    builder << "  推荐选择子组大小: " << sub_group_size << "\n";
    builder << "  100% 占用率工作组大小: " << work_group_size;
    return builder.str();
  }
};

}  // namespace base16384

#endif  // _XEINFO_HPP_