// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AOT_CACHE_20230202_HPP
#define AOT_CACHE_20230202_HPP

#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"

#include <CL/sycl.hpp>
#include <unordered_set>

class aot_cache : public bbfft::jit_cache {
  public:
    aot_cache(::sycl::queue q);
    auto get(bbfft::jit_cache_key const &key) const
        -> bbfft::shared_handle<bbfft::module_handle_t> override;
    void store(bbfft::jit_cache_key const &key,
               bbfft::shared_handle<bbfft::module_handle_t> module) override;

  private:
    bbfft::shared_handle<bbfft::module_handle_t> module_;
};

#endif // AOT_CACHE_20230202_HPP
