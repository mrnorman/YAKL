// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AOT_CACHE_20230202_HPP
#define AOT_CACHE_20230202_HPP

#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"
#include <bbfft/sycl/online_compiler.hpp>

#include <CL/sycl.hpp>
#include <unordered_set>

#include "aot_compiled_kernels.hpp"

class aot_cache : public bbfft::jit_cache {
  public:
    aot_cache(::sycl::queue q) {

  #if defined(YAKL_SYCL_BBFFT_AOT_LEGACY_UMD)
      extern const uint8_t _binary_kernels_XE_HPC_COREpvc_bin_start, _binary_kernels_XE_HPC_COREpvc_bin_end;

      auto handle = bbfft::sycl::build_native_module(&_binary_kernels_XE_HPC_COREpvc_bin_start,
                                                     &_binary_kernels_XE_HPC_COREpvc_bin_end -
                                                     &_binary_kernels_XE_HPC_COREpvc_bin_start,
                                                     bbfft::module_format::native, q.get_context(), q.get_device());

      module_ = bbfft::sycl::make_shared_handle(handle, q.get_backend());
  #else
      extern const uint8_t _binary_kernels_pvc_bin_start, _binary_kernels_pvc_bin_end;

      auto handle = bbfft::sycl::build_native_module(&_binary_kernels_pvc_bin_start,
                                                     &_binary_kernels_pvc_bin_end -
                                                     &_binary_kernels_pvc_bin_start,
                                                     bbfft::module_format::native, q.get_context(), q.get_device());

      module_ = bbfft::sycl::make_shared_handle(handle, q.get_backend());
  #endif
    }
    
    auto get(bbfft::jit_cache_key const &key) const -> bbfft::shared_handle<bbfft::module_handle_t> {
      if (auto it = aot_compiled_kernels.find(key.kernel_name); it != aot_compiled_kernels.end()) {
        return module_;
      }
      return {};
    }

    void store(bbfft::jit_cache_key const &, bbfft::shared_handle<bbfft::module_handle_t>) {}

  private:
    bbfft::shared_handle<bbfft::module_handle_t> module_;
};

#endif // AOT_CACHE_20230202_HPP
