/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_test/column_wrapper.hpp"
#include "jit/row_ir.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <jit/jit.hpp>

#include <chrono>

using namespace cudf;

struct RTCTest : public ::testing::Test {};

TEST_F(RTCTest, CompileKernelBasic)
{
  // TODO: batch kernel precompilation?
  auto fn = [] {
    char const* udf = R"***(
    // #define CUDF_JIT_LITE_EXCLUDE_OPERATORS
    // #include "jit/lite/cudf.cuh"

    // #pragma nv_hdrstop

    // using namespace cudf::lite;

    extern "C" __device__ void binary_operator(void * out_ptr, void const * a_ptr, void const * b_ptr){
      auto a = *static_cast<int const*>(a_ptr);
      auto b = *static_cast<int const*>(b_ptr);
      auto & out = *static_cast<int*>(out_ptr);

      out = ((a + b) * (a - b)) + ((a * b) / (a + 1));
    }


    extern "C" __global__ void transform_kernel(
    int num_rows,
    void const * __restrict__ outputs,
    void const * __restrict__ inputs,
    void const * __restrict__ user_data) {
      auto offset = static_cast<signed long long>(threadIdx.x) + static_cast<signed long long>(blockIdx.x) * static_cast<signed long long>(blockDim.x);
      auto stride = static_cast<signed long long>(blockDim.x) * static_cast<signed long long>(gridDim.x);

      for(signed long long i = offset; i < num_rows; i += stride){
        auto const& out_col = static_cast<int* const *>(outputs)[0];
        auto const& a_col = static_cast<int const * const *>(inputs)[0];
        auto const& b_col = static_cast<int const * const *>(inputs)[1];
        auto a = a_col[i];
        auto b = b_col[i];
        int out;
        binary_operator(&out, &a, &b);
        out_col[i] = out;
      }
    }
    )***";

    auto [lib, _] = cudf::compile_library_uncached("test_kernel",
                                                   udf,
                                                   /*use_pch=*/true,
                                                   /*log_pch=*/true);

    // auto kernel = lib->get_kernel("transform_kernel");

    // EXPECT_EQ("transform_kernel", kernel.get_name());
  };

  fn();  // warm up cache
  fn();
  fn();
  fn();
}

CUDF_TEST_PROGRAM_MAIN()
