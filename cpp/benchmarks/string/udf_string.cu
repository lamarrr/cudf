/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/strings/udf/udf_string.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <nvbench/nvbench.cuh>
#include <nvbench/types.cuh>

__device__ cudf::strings::udf::udf_string join_strings(
  cudf::device_span<cudf::column_device_view const> strings,
  cudf::size_type i,
  cudf::strings::udf::fallback_allocator allocator)
{
  cudf::strings::udf::udf_string out{allocator};
  for (auto const& col : strings) {
    out.append(col.element<cudf::string_view>(i));
  }
  return out;
}

__global__ void string_concat_kernel(cudf::device_span<cudf::strings::udf::udf_string> output,
                                     cudf::device_span<cudf::column_device_view const> columns)
{
  cudf::thread_index_type tid = static_cast<cudf::thread_index_type>(threadIdx.x) +
                                static_cast<cudf::thread_index_type>(blockIdx.x) *
                                  static_cast<cudf::thread_index_type>(blockDim.x);
  cudf::thread_index_type stride = static_cast<cudf::thread_index_type>(blockDim.x) *
                                   static_cast<cudf::thread_index_type>(gridDim.x);

  for (cudf::thread_index_type i = tid; i < output.size(); i += stride) {
    // join_strings(columns, i, {});
    new (output.data() + i) cudf::strings::udf::udf_string{"Hello World"};
  }
}

// Hopper: 2048 threads per SM, 196 SMs; 2048x196 = 393,216 resident threads
//
// at 512 bytes per thread: 201'326'592 ~ 200MB per-launch
__device__ cudf::strings::udf::arena make_thread_arena(cudf::device_span<char> buffer,
                                                       cudf::thread_index_type num_threads,
                                                       cudf::thread_index_type idx)
{
  // [ ] minimum size per-thread
  int64_t size_per_thread = buffer.size() / num_threads;
  int64_t offset          = idx * num_threads * size_per_thread;
  int64_t end             = offset + size_per_thread;
  if (offset >= buffer.size()) { return {}; }
  int64_t size = std::min(end, (int64_t)buffer.size()) - offset;
  return cudf::strings::udf::arena{buffer.data() + offset, static_cast<size_t>(size)};
}

__global__ void string_concat_amortized_kernel(
  cudf::device_span<cudf::strings::udf::udf_string> output,
  cudf::device_span<cudf::column_device_view const> columns,
  cudf::device_span<char> scratch_buffer,
  cudf::device_span<char> output_buffer)
{
  cudf::thread_index_type tid = static_cast<cudf::thread_index_type>(threadIdx.x) +
                                static_cast<cudf::thread_index_type>(blockIdx.x) *
                                  static_cast<cudf::thread_index_type>(blockDim.x);
  cudf::thread_index_type stride = static_cast<cudf::thread_index_type>(blockDim.x) *
                                   static_cast<cudf::thread_index_type>(gridDim.x);

  // cudf::strings::udf::arena thread_arena = make_thread_arena(scratch_buffer, stride, tid);
  cudf::strings::udf::fallback_allocator thread_allocator{nullptr};
  // cudf::strings::udf::arena output_arena = make_thread_arena(output_buffer, stride, tid);
  cudf::strings::udf::fallback_allocator output_allocator{nullptr};

  for (cudf::thread_index_type i = tid; i < output.size(); i += stride) {
    new (output.data() + i)
      cudf::strings::udf::udf_string{join_strings(columns, i, thread_allocator), output_allocator};
    // can reclaim intermediate arena
    // thread_arena.reclaim();
  }
}

template <typename T>
rmm::device_buffer make_typed_buffer_uninit(size_t size,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref resource)
{
  return rmm::device_buffer{size * sizeof(T), stream, resource};
}

template <typename T>
rmm::device_buffer make_typed_buffer(std::vector<T> const& data,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref resource)
{
  auto buffer = make_typed_buffer_uninit<T>(data.size(), stream, resource);
  cudf::detail::cuda_memcpy_async(cudf::device_span<T>{static_cast<T*>(buffer.data()), data.size()},
                                  cudf::host_span<T const>{data.data(), data.size()},
                                  stream);
  return buffer;
}

void free_udf_string_array(cudf::strings::udf::udf_string* d_strings,
                           cudf::size_type size,
                           rmm::cuda_stream_view stream)
{
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator(0),
                     size,
                     [d_strings] __device__(auto idx) { d_strings[idx].reset(); });
}

template <bool amortized>
static void BM_udf_strings(nvbench::state& state)
{
  auto const num_rows                          = state.get_int64("num_rows");
  auto const order                             = state.get_int64("order");
  auto const string_width                      = state.get_int64("string_width");
  auto const num_columns                       = order + 1;
  static constexpr int64_t SCRATCH_BUFFER_SIZE = 100 * (1LL << 20);
  static constexpr int64_t OUTPUT_BUFFER_SIZE  = 100 * (1LL << 20);
  static constexpr int64_t HIT_RATE            = 50;

  std::vector<std::unique_ptr<cudf::column>> columns;
  std::transform(thrust::make_counting_iterator(0L),
                 thrust::make_counting_iterator(num_columns),
                 std::back_inserter(columns),
                 [&](size_t) { return create_string_column(num_rows, string_width, HIT_RATE); });

  std::vector<
    std::unique_ptr<cudf::column_device_view, std::function<void(cudf::column_device_view*)>>>
    column_view_handles;
  std::vector<cudf::column_device_view> column_views;
  std::transform(
    columns.begin(), columns.end(), std::back_inserter(column_view_handles), [&](auto& column) {
      return cudf::column_device_view::create(column->view(), state.get_cuda_stream().get_stream());
    });

  CUDF_CUDA_TRY(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8LL * (1LL << 30)));

  std::transform(column_view_handles.begin(),
                 column_view_handles.end(),
                 std::back_inserter(column_views),
                 [&](auto& view) { return *view; });

  auto d_column_views = make_typed_buffer(
    column_views, state.get_cuda_stream().get_stream(), cudf::get_current_device_resource_ref());

  state.add_global_memory_reads<char>(num_columns * static_cast<int64_t>(num_rows) *
                                      static_cast<int64_t>(string_width));
  state.add_global_memory_writes<char>(static_cast<int64_t>(num_rows) *
                                       static_cast<int64_t>(string_width));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream{launch.get_stream().get_stream()};
    auto mr = cudf::get_current_device_resource_ref();
    stream.synchronize();
    [[maybe_unused]] rmm::device_buffer output =
      make_typed_buffer_uninit<cudf::strings::udf::udf_string>(num_rows, stream, mr);
    [[maybe_unused]] auto output_span = cudf::device_span<cudf::strings::udf::udf_string>{
      static_cast<cudf::strings::udf::udf_string*>(output.data()), static_cast<size_t>(num_rows)};

    CUDF_CUDA_TRY(cudaMemsetAsync(
      output_span.data(), 0, num_rows * sizeof(cudf::strings::udf::udf_string), stream.value()));

    [[maybe_unused]] auto columns_span = cudf::device_span<cudf::column_device_view>{
      static_cast<cudf::column_device_view*>(d_column_views.data()),
      static_cast<size_t>(num_columns)};
    if constexpr (amortized && false) {
      rmm::device_uvector<char> output_buffer{OUTPUT_BUFFER_SIZE, stream, mr};
      rmm::device_uvector<char> scratch_buffer{SCRATCH_BUFFER_SIZE, stream, mr};
      // string_concat_amortized_kernel<<<1024, 196, 0, launch.get_stream()>>>(
      // output_span, columns_span, scratch_buffer, output_buffer);
    } else {
      string_concat_kernel<<<1024, 196, 0, launch.get_stream()>>>(output_span, columns_span);
    }

    free_udf_string_array(output_span.data(), output_span.size(), stream);

    stream.synchronize();
  });
}

#define AST_POLYNOMIAL_BENCHMARK_DEFINE(name, amortized)                      \
  static void name(::nvbench::state& st) { ::BM_udf_strings<amortized>(st); } \
  NVBENCH_BENCH(name)                                                         \
    .set_name(#name)                                                          \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000})             \
    .add_int64_axis("order", {1, 16})                                         \
    .add_int64_axis("string_width", {20, 128})

AST_POLYNOMIAL_BENCHMARK_DEFINE(udf_string_concat, false);

AST_POLYNOMIAL_BENCHMARK_DEFINE(udf_string_concat_amortized, true);
