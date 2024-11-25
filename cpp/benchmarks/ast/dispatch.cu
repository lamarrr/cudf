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
  POW       = 8,
  SINCOS    = 9
};

template <typename T>
__device__ T xadd(T a, T b)
{
  return a + b;
}

template <typename T>
__device__ T xsub(T a, T b)
{
  return a - b;
}

template <typename T>
__device__ T xmul(T a, T b)
{
  return a * b;
}

template <typename T>
__device__ T xdiv(T a, T b)
{
  return a / b;
}

template <typename T>
__device__ T xtrue_div(T a, T b)
{
  return (T)((float)a / (float)b);
}

template <typename T>
__device__ T xfloor_div(T a, T b)
{
  return (T)((int64_t)(a / b));
}

template <typename T>
__device__ T xmod(T a, T b)
{
  return (T)((int64_t)(a) % (int64_t)b);
}

template <typename T>
__device__ T xpymod(T a, T b)
{
  return (T)((int64_t)(a) % (int64_t)b);
}

template <typename T>
__device__ T xpow_fn(T a, T b)
{
  return (T)std::pow(a, b);
}

template <typename T>
__device__ T xsincos(T a, T b)
{
  return (T)(std::sin(a) * std::cos(b));
}

template <typename T>
using binary_func = T (*)(T, T);

template <typename T>
__global__ void switch_dispatch(T const* __restrict a,
                                T const* __restrict b,
                                T* __restrict c,
                                int64_t n,
                                cudf::ast::ast_operator op)
{
  uint64_t i = (uint64_t)threadIdx.x + (uint64_t)blockDim.x * (uint64_t)blockIdx.x;

  switch (op) {
    case cudf::ast::ast_operator::ADD: c[i] = xadd(a[i], b[i]); return;
    case cudf::ast::ast_operator::SUB: c[i] = xsub(a[i], b[i]); return;
    case cudf::ast::ast_operator::MUL: c[i] = xmul(a[i], b[i]); return;
    case cudf::ast::ast_operator::DIV: c[i] = xdiv(a[i], b[i]); return;
    case cudf::ast::ast_operator::TRUE_DIV: c[i] = xtrue_div(a[i], b[i]); return;
    case cudf::ast::ast_operator::FLOOR_DIV: c[i] = xfloor_div(a[i], b[i]); return;
    case cudf::ast::ast_operator::MOD: c[i] = xmod(a[i], b[i]); return;
    case cudf::ast::ast_operator::PYMOD: c[i] = xpymod(a[i], b[i]); return;
    case cudf::ast::ast_operator::POW: c[i] = xpow_fn(a[i], b[i]); return;
    case cudf::ast::ast_operator::SINH: c[i] = xsincos(a[i], b[i]); return;
    default: CUDF_UNREACHABLE("");
  }
}

template <typename T>
__global__ void index_dispatch(
  T const* __restrict a, T const* __restrict b, T* __restrict c, int64_t n, binary_op op)
{
  uint64_t i = (uint64_t)threadIdx.x + (uint64_t)blockDim.x * (uint64_t)blockIdx.x;

  static constexpr __device__ binary_func<T> ops[] = {
    xadd, xsub, xmul, xdiv, xtrue_div, xfloor_div, xmod, xpymod, xpow_fn};

  c[i] = ops[(int32_t)op](a[i], b[i]);
}

template <typename T>
void launch_switch(
  T const* a, T const* b, T* c, int64_t n, cudf::ast::ast_operator op, cudaStream_t stream)
{
  CUDF_EXPECTS((n & 1023ULL) == 0, "");
  CUDF_EXPECTS((n / 1024ULL) < std::numeric_limits<int32_t>::max(), "");
  dim3 block_dim{1024U, 1, 1};
  dim3 grid_dim{(uint32_t)(n / 1024ULL), 1, 1};
  switch_dispatch<<<grid_dim, block_dim, 0, stream>>>(a, b, c, n, op);
}

template <typename T>
void launch_index(T const* a, T const* b, T* c, int64_t n, binary_op op, cudaStream_t stream)
{
  CUDF_EXPECTS((n & 1023ULL) == 0, "");
  CUDF_EXPECTS((n / 1024ULL) < std::numeric_limits<int32_t>::max(), "");
  dim3 block_dim{1024U, 1, 1};
  dim3 grid_dim{(uint32_t)(n / 1024ULL), 1, 1};
  index_dispatch<<<grid_dim, block_dim, 0, stream>>>(a, b, c, n, op);
}

template <typename T, cudf::ast::ast_operator op>
static void BM_switch(nvbench::state& state, nvbench::type_list<T, nvbench::enum_type<op>>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  thrust::device_vector<T> a;
  thrust::device_vector<T> b;
  thrust::device_vector<T> c;

  a.resize(num_rows);
  b.resize(num_rows);
  c.resize(num_rows);

  // Use the number of bytes read from global memory
  state.add_element_count(num_rows, "num_ops");
  state.add_global_memory_reads<nvbench::uint8_t>(a.size() * sizeof(T) + b.size() * sizeof(T));
  state.add_global_memory_writes<nvbench::int32_t>(c.size() * sizeof(T));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& l) {
    launch_switch(a.data().get(), b.data().get(), c.data().get(), num_rows, op, l.get_stream());
  });
}

template <typename T, binary_op op>
static void BM_index(nvbench::state& state, nvbench::type_list<T, nvbench::enum_type<op>>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  thrust::device_vector<T> a;
  thrust::device_vector<T> b;
  thrust::device_vector<T> c;

  a.resize(num_rows);
  b.resize(num_rows);
  c.resize(num_rows);

  // Use the number of bytes read from global memory
  state.add_element_count(num_rows, "num_ops");
  state.add_global_memory_reads<nvbench::uint8_t>(a.size() * sizeof(T) + b.size() * sizeof(T));
  state.add_global_memory_writes<nvbench::int32_t>(c.size() * sizeof(T));

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
    case cudf::ast::ast_operator::SINH: return "SINCOS";
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
    case binary_op::SINCOS: return "SINCOS";
    default: return "Unidentified";
  }
}

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(cudf::ast::ast_operator, to_string, to_string)
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(binary_op, to_string, to_string)

using switch_binops = nvbench::enum_type_list<cudf::ast::ast_operator::ADD,
                                              cudf::ast::ast_operator::SUB,
                                              cudf::ast::ast_operator::MUL,
                                              cudf::ast::ast_operator::DIV,
                                              cudf::ast::ast_operator::TRUE_DIV,
                                              cudf::ast::ast_operator::FLOOR_DIV,
                                              cudf::ast::ast_operator::MOD,
                                              cudf::ast::ast_operator::PYMOD,
                                              cudf::ast::ast_operator::POW,
                                              cudf::ast::ast_operator::SINH>;

using index_binops = nvbench::enum_type_list<binary_op::ADD,
                                             binary_op::SUB,
                                             binary_op::MUL,
                                             binary_op::DIV,
                                             binary_op::TRUE_DIV,
                                             binary_op::FLOOR_DIV,
                                             binary_op::MOD,
                                             binary_op::PYMOD,
                                             binary_op::POW,
                                             binary_op::SINCOS>;

using types = nvbench::type_list<int32_t, float>;

NVBENCH_BENCH_TYPES(BM_switch, NVBENCH_TYPE_AXES(types, switch_binops))
  .set_name("switch_dispatch")
  .add_int64_power_of_two_axis("num_rows", nvbench::range(10, 22, 4));

NVBENCH_BENCH_TYPES(BM_index, NVBENCH_TYPE_AXES(types, index_binops))
  .set_name("index_dispatch")
  .add_int64_power_of_two_axis("num_rows", nvbench::range(10, 22, 4));
