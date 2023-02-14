// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "aot_cache.hpp"
#include "aot_compiled_kernels.hpp"

#include <bbfft/sycl/online_compiler.hpp>

using namespace bbfft;

aot_cache::aot_cache(::sycl::queue q) {

#if defined(YAKL_SYCL_BBFFT_AOT_LEGACY_UMD)
    extern const uint8_t _binary_kernels_XE_HPC_COREpvc_bin_start, _binary_kernels_XE_HPC_COREpvc_bin_end;

    auto handle = bbfft::sycl::build_native_module(&_binary_kernels_XE_HPC_COREpvc_bin_start,
                                                   &_binary_kernels_XE_HPC_COREpvc_bin_end -
                                                   &_binary_kernels_XE_HPC_COREpvc_bin_start,
                                                   q.get_context(), q.get_device());
    
    module_ = bbfft::sycl::make_shared_handle(handle, q.get_backend());
#else
    extern const uint8_t _binary_kernels_pvc_bin_start, _binary_kernels_pvc_bin_end;

    auto handle = bbfft::sycl::build_native_module(&_binary_kernels_pvc_bin_start,
                                                   &_binary_kernels_pvc_bin_end -
                                                   &_binary_kernels_pvc_bin_start,
                                                   q.get_context(), q.get_device());

    module_ = bbfft::sycl::make_shared_handle(handle, q.get_backend());
#endif
}

auto aot_cache::get(jit_cache_key const &key) const -> shared_handle<module_handle_t> {
    if (auto it = aot_compiled_kernels.find(key.kernel_name); it != aot_compiled_kernels.end()) {
        return module_;
    }
    return {};
}
void aot_cache::store(jit_cache_key const &, shared_handle<module_handle_t>) {}

