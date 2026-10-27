/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace {

std::string const add_ptx = R"(
.version 6.4
.target sm_70
.address_size 64
.visible .func (.param .b32 status) add(
  .param .b64 output,
  .param .b32 lhs,
  .param .b32 rhs
)
{
  .reg .b64 %rd;
  .reg .b32 %a, %b, %c;
  ld.param.u64 %rd, [output];
  ld.param.u32 %a, [lhs];
  ld.param.u32 %b, [rhs];
  add.s32 %c, %a, %b;
  st.u32 [%rd], %c;
  st.param.b32 [status], 0;
  ret;
}
)";

// Compare the deprecated API with direct transform calls.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

void binaryop_ptx_add(nvbench::state& state)
{
  auto const rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const nullable = state.get_int64("nullable") != 0;
  auto const type     = cudf::data_type{cudf::type_id::INT32};
  std::vector<int32_t> values(rows);
  std::iota(values.begin(), values.end(), int32_t{0});
  std::mt19937 generator{0};
  std::bernoulli_distribution is_valid{0.5};
  auto make_input = [&] {
    if (nullable) {
      std::vector<bool> validity(rows);
      std::generate(validity.begin(), validity.end(), [&] { return is_valid(generator); });
      return cudf::test::fixed_width_column_wrapper<int32_t>(
               values.begin(), values.end(), validity.begin())
        .release();
    }
    return cudf::test::fixed_width_column_wrapper<int32_t>(values.begin(), values.end()).release();
  };
  auto const lhs_column = make_input();
  auto const rhs_column = make_input();
  auto const lhs        = lhs_column->view();
  auto const rhs        = rhs_column->view();

  auto const api        = state.get_string("api");
  auto const use_binary = api == "binary_operation";
  auto const source_type =
    api == "transform_ptx" ? cudf::udf_source_type::PTX : cudf::udf_source_type::CUDA;
  std::string const cuda_udf =
    "__device__ void add(int32_t* out, int32_t a, int32_t b) { *out = a + b; }";
  auto const& udf = source_type == cudf::udf_source_type::PTX ? add_ptx : cuda_udf;

  cudf::transform_input const inputs[]   = {lhs, rhs};
  cudf::transform_output const outputs[] = {{.type = type}};

  auto run = [&](cuda::stream_ref stream) {
    if (use_binary) {
      cudf::binary_operation(lhs, rhs, add_ptx, type, stream);
    } else {
      cudf::transform(udf,
                      source_type,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      outputs,
                      {},
                      std::nullopt,
                      stream);
    }
  };

  // Compile and populate the in-process kernel cache before measuring warm calls.
  auto const start = std::chrono::steady_clock::now();
  run(cudf::get_default_stream());
  cudf::get_default_stream().sync();
  auto const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  auto& first_call   = state.add_summary("first_call");
  first_call.set_string("name", "First call");
  first_call.set_string("hint", "duration");
  first_call.set_float64("value", elapsed.count());

  state.add_global_memory_reads<int32_t>(2 * static_cast<std::size_t>(rows));
  state.add_global_memory_writes<int32_t>(rows);
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run(launch.get_stream().get_stream()); });
}

#pragma GCC diagnostic pop

}  // namespace

NVBENCH_BENCH(binaryop_ptx_add)
  .set_name("binaryop_ptx_add")
  .add_string_axis("api", {"binary_operation", "transform_ptx", "transform_cuda"})
  .add_int64_axis("num_rows", {1'024, 1'000'000, 10'000'000})
  .add_int64_axis("nullable", {0, 1});
