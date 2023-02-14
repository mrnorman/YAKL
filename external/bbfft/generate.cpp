// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "configurations.hpp"

#include <bbfft/configuration.hpp>
#include <bbfft/device_info.hpp>
#include <bbfft/generator.hpp>

#include <fstream>
#include <iostream>
#include <vector>

using namespace bbfft;

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: aot-generate <kernel_file> <identifier_file>" << std::endl;
        return -1;
    }

    auto kernel_file = std::ofstream(argv[1]);
    auto identifier_file = std::ofstream(argv[2]);

    device_info info = {1024, {16, 32}, 2, 128 * 1024};
    auto kernel_names = generate_fft_kernels(kernel_file, configurations(), info);
    identifier_file << "#include <string>" << std::endl;
    identifier_file << "#include <unordered_set>" << std::endl;
    identifier_file << "std::unordered_set<std::string> aot_compiled_kernels = {";
    for (auto const &name : kernel_names) {
        identifier_file << "\"" << name << "\", ";
    }
    identifier_file.seekp(-2, std::ios::cur);
    identifier_file << "};" << std::endl;

    return 0;
}
