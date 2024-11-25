/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cudf/detail/utilities/assert.cuh"
#include "cudf/utilities/error.hpp"
#include "nvbench/create.cuh"
#include "nvbench/detail/type_list_impl.cuh"
#include "nvbench/enum_type_list.cuh"
#include "thrust/device_vector.h"
#include "thrust/fill.h"

#include <cudf/ast/expressions.hpp>

#include <nvbench/nvbench.cuh>
#include <nvbench/types.cuh>

#include <cstdint>

enum class binary_op : int32_t {
  ADD       = 0,
  SUB       = 1,
  MUL       = 2,
  DIV       = 3,
  TRUE_DIV  = 4,
  FLOOR_DIV = 5,
  MOD       = 6,
  PYMOD     = 7,
  POW       = 8
};

__device__ float add(float a, float b) { return a + b; }

__device__ float sub(float a, float b) { return a - b; }

__device__ float mul(float a, float b) { return a * b; }

__device__ float div(float a, float b) { return a / b; }

__device__ float true_div(float a, float b) { return a / b; }

__device__ float floor_div(float a, float b) { return a / b; }

__device__ float mod(float a, float b) { return (float)((int64_t)(a) % (int64_t)b); }

__device__ float pymod(float a, float b) { return (float)((int64_t)(a) % (int64_t)b); }

__device__ float pow_fn(float a, float b) { return std::pow(a, b); }

__global__ void switch_dispatch(float const* __restrict a,
                                float const* __restrict b,
                                float* __restrict c,
                                int64_t n,
                                cudf::ast::ast_operator op)
{
  uint64_t i = (uint64_t)threadIdx.x + (uint64_t)blockDim.x * (uint64_t)blockIdx.x;

  switch (op) {
    case cudf::ast::ast_operator::ADD: c[i] = add(a[i], b[i]); return;
    case cudf::ast::ast_operator::SUB: c[i] = sub(a[i], b[i]); return;
    case cudf::ast::ast_operator::MUL: c[i] = mul(a[i], b[i]); return;
    case cudf::ast::ast_operator::DIV: c[i] = div(a[i], b[i]); return;
    case cudf::ast::ast_operator::TRUE_DIV: c[i] = true_div(a[i], b[i]); return;
    case cudf::ast::ast_operator::FLOOR_DIV: c[i] = floor_div(a[i], b[i]); return;
    case cudf::ast::ast_operator::MOD: c[i] = mod(a[i], b[i]); return;
    case cudf::ast::ast_operator::PYMOD: c[i] = pymod(a[i], b[i]); return;
    case cudf::ast::ast_operator::POW: c[i] = pow_fn(a[i], b[i]); return;
    default: CUDF_UNREACHABLE("");
  }
}

__global__ void index_dispatch(float const* __restrict a,
                               float const* __restrict b,
                               float* __restrict c,
                               int64_t n,
                               binary_op op)
{
  uint64_t i = (uint64_t)threadIdx.x + (uint64_t)blockDim.x * (uint64_t)blockIdx.x;

  static constexpr __device__ decltype(add)* ops[] = {
    add, sub, mul, div, true_div, floor_div, mod, pymod, pow_fn};

  c[i] = ops[(int32_t)op](a[i], b[i]);
}

void launch_switch(float const* a,
                   float const* b,
                   float* c,
                   int64_t n,
                   cudf::ast::ast_operator op,
                   cudaStream_t stream)
{
  CUDF_EXPECTS((n & 1023ULL) == 0, "");
  CUDF_EXPECTS((n / 1024ULL) < std::numeric_limits<int32_t>::max(), "");
  dim3 block_dim{1024U, 1, 1};
  dim3 grid_dim{(uint32_t)(n / 1024ULL), 1, 1};
  switch_dispatch<<<grid_dim, block_dim, 0, stream>>>(a, b, c, n, op);
}

void launch_index(
  float const* a, float const* b, float* c, int64_t n, binary_op op, cudaStream_t stream)
{
  CUDF_EXPECTS((n & 1023ULL) == 0, "");
  CUDF_EXPECTS((n / 1024ULL) < std::numeric_limits<int32_t>::max(), "");
  dim3 block_dim{1024U, 1, 1};
  dim3 grid_dim{(uint32_t)(n / 1024ULL), 1, 1};
  index_dispatch<<<grid_dim, block_dim, 0, stream>>>(a, b, c, n, op);
}

template <cudf::ast::ast_operator op>
static void BM_switch(nvbench::state& state, nvbench::type_list<nvbench::enum_type<op>>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  thrust::device_vector<float> a;
  thrust::device_vector<float> b;
  thrust::device_vector<float> c;

  a.resize(num_rows);
  b.resize(num_rows);
  c.resize(num_rows);

  // Use the number of bytes read from global memory
  state.add_element_count(num_rows, "num_ops");
  state.add_global_memory_reads<nvbench::uint8_t>(a.size() * sizeof(float) +
                                                  b.size() * sizeof(float));
  state.add_global_memory_writes<nvbench::int32_t>(c.size() * sizeof(float));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& l) {
    launch_switch(a.data().get(), b.data().get(), c.data().get(), num_rows, op, l.get_stream());
  });
}

template <binary_op op>
static void BM_index(nvbench::state& state, nvbench::type_list<nvbench::enum_type<op>>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  thrust::device_vector<float> a;
  thrust::device_vector<float> b;
  thrust::device_vector<float> c;

  a.resize(num_rows);
  b.resize(num_rows);
  c.resize(num_rows);

  // Use the number of bytes read from global memory
  state.add_element_count(num_rows, "num_ops");
  state.add_global_memory_reads<nvbench::uint8_t>(a.size() * sizeof(float) +
                                                  b.size() * sizeof(float));
  state.add_global_memory_writes<nvbench::int32_t>(c.size() * sizeof(float));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& l) {
    launch_index(a.data().get(), b.data().get(), c.data().get(), num_rows, op, l.get_stream());
  });
}

static char const* to_string(cudf::ast::ast_operator op)
{
  switch (op) {
    case cudf::ast::ast_operator::ADD: return "ADD";
    case cudf::ast::ast_operator::SUB: return "SUB";
    case cudf::ast::ast_operator::MUL: return "MUL";
    case cudf::ast::ast_operator::DIV: return "DIV";
    case cudf::ast::ast_operator::TRUE_DIV: return "TRUE_DIV";
    case cudf::ast::ast_operator::FLOOR_DIV: return "FLOOR_DIV";
    case cudf::ast::ast_operator::MOD: return "MOD";
    case cudf::ast::ast_operator::PYMOD: return "PYMOD";
    case cudf::ast::ast_operator::POW: return "POW";
    default: return "Unidentified";
  }
}

static char const* to_string(binary_op op)
{
  switch (op) {
    case binary_op::ADD: return "ADD";
    case binary_op::SUB: return "SUB";
    case binary_op::MUL: return "MUL";
    case binary_op::DIV: return "DIV";
    case binary_op::TRUE_DIV: return "TRUE_DIV";
    case binary_op::FLOOR_DIV: return "FLOOR_DIV";
    case binary_op::MOD: return "MOD";
    case binary_op::PYMOD: return "PYMOD";
    case binary_op::POW: return "POW";
    default: return "Unidentified";
  }
}

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(cudf::ast::ast_operator, to_string, to_string)
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(binary_op, to_string, to_string)

using switch_binops = nvbench::enum_type_list<cudf::ast::ast_operator::ADD,
                                              // cudf::ast::ast_operator::SUB,
                                              // cudf::ast::ast_operator::MUL,
                                              cudf::ast::ast_operator::DIV,
                                              // cudf::ast::ast_operator::TRUE_DIV,
                                              // cudf::ast::ast_operator::FLOOR_DIV,
                                              cudf::ast::ast_operator::MOD,
                                              // cudf::ast::ast_operator::PYMOD,
                                              cudf::ast::ast_operator::POW>;

using index_binops = nvbench::enum_type_list<binary_op::ADD,
                                             //  binary_op::SUB,
                                             //  binary_op::MUL,
                                             binary_op::DIV,
                                             //  binary_op::TRUE_DIV,
                                             //  binary_op::FLOOR_DIV,
                                             binary_op::MOD,
                                             //  binary_op::PYMOD,
                                             binary_op::POW>;

NVBENCH_BENCH_TYPES(BM_switch, NVBENCH_TYPE_AXES(switch_binops))
  .set_name("switch_dispatch")
  .add_int64_power_of_two_axis("num_rows", {10, 16, 18, 26, 30, 32});

NVBENCH_BENCH_TYPES(BM_index, NVBENCH_TYPE_AXES(index_binops))
  .set_name("index_dispatch")
  .add_int64_power_of_two_axis("num_rows", {10, 16, 18, 26, 30, 32});
