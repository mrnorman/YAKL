// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "configurations.hpp"

using namespace bbfft;

std::vector<configuration> configurations() {
  auto cfgs = std::vector<configuration>{};
  unsigned long fft_size = 7;
  unsigned long batch_size = 64;
  configuration cfg_template_forward = {
    1, {1, fft_size, batch_size}, precision::f64, direction::forward, transform_type::r2c
  };
  configuration cfg_template_inverse = {
    1, {1, fft_size, batch_size}, precision::f64, direction::backward, transform_type::c2r
  };
  cfg_template_forward.set_strides_default(true);
  cfgs.push_back(cfg_template_forward);
  cfg_template_inverse.set_strides_default(true);
  cfgs.push_back(cfg_template_inverse);

  cfg_template_forward.shape[1] = 100;
  cfgs.push_back(cfg_template_forward);
  // add some power of 2 FFT sizes
  for (unsigned int i = 16; i < 256; i *= 2) {
    if (i != fft_size) {
      cfg_template_forward.shape[1] = i;
      cfg_template_forward.set_strides_default(true);
      cfgs.push_back(cfg_template_forward);
      cfg_template_inverse.shape[1] = i;
      cfg_template_inverse.set_strides_default(true);
      cfgs.push_back(cfg_template_inverse);
    }
  }
  return cfgs;
}
