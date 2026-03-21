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
    #include "jit/lite/cudf.cuh"

    #pragma nv_hdrstop

    using namespace cudf::lite;

    extern "C" __device__ void binary_operator(void * user_data, size_type index, void * out_ptr, void const * a_ptr, void const * b_ptr){
      auto a = *static_cast<int32_t const*>(a_ptr);
      auto b = *static_cast<int32_t const*>(b_ptr);
      auto & out = *static_cast<int32_t*>(out_ptr);

      out = ((a + b) * (a - b)) + ((a * b) / (a + 1));
    }


    extern "C" __global__ void transform_kernel(
    int32_t num_rows,
    void const * __restrict__ outputs,
    void const * __restrict__ inputs,
    void const * __restrict__ user_data) {
      auto offset = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
      auto stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

      for(int64_t i = offset; i < num_rows; i += stride){
        auto const * __restrict__ out_col = static_cast<column_accessor<true, false, false> const *>(outputs);
        auto const * __restrict__ a_col = static_cast<column_accessor<false, false, false> const *>(inputs);
        auto const * __restrict__ b_col = static_cast<column_accessor<false, false, false> const *>(inputs) + 1;
        auto a = a_col->element<int32_t>(i);
        auto b = b_col->element<int32_t>(i);
        int32_t out;
        binary_operator(user_data, i, &out, &a, &b);
        out_col->assign(i, out);
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
