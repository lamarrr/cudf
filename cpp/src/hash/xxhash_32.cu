/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/context.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/xxhash_32.cuh>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/stream>

namespace cudf {
namespace hashing {
namespace detail {

std::unique_ptr<column> xxhash_32(table_view const& input,
                                  uint32_t seed,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  if (input.num_rows() == 0) { return output; }

  bool const nullable = has_nulls(input);
  auto const row_hasher =
    cudf::detail::row::hash::row_hasher(input, stream, cudf::get_current_device_resource_ref());
  auto output_view = output->mutable_view();

  // Compute the hash value for each row
  auto const output_begin = output_view.begin<hash_value_type>();
  auto const hasher       = row_hasher.device_hasher<XXHash_32>(nullable, seed);
  // thrust::tabulate is slow here, see NVIDIA/cccl#9070
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    input.num_rows(),
    [output_begin, hasher] __device__(size_type i) mutable { output_begin[i] = hasher(i); },
    stream.get()));

  return output;
}

std::unique_ptr<column> xxhash_32_jit(table_view const& input,
                                      uint32_t seed,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  auto const preprocessed = cudf::detail::row::hash::preprocessed_table::create(
    input, stream, cudf::get_current_device_resource_ref());

  cudf::detail::device_scalar<uint32_t> seed_scalar{
    seed, stream, cudf::get_current_device_resource_ref()};
  auto const seed_view = column_view{data_type{type_id::UINT32}, 1, seed_scalar.data(), nullptr, 0};
  static constexpr char const* source = R"***(
// xxHash32 lazy-row ABI v1
#include <hash/jit/xxhash_32.cuh>
)***";
  auto const nullable                 = has_nulls(input);
  std::vector<transform_input> inputs;
  inputs.reserve(preprocessed->table().num_columns() + 1);
  for (auto const& column : preprocessed->table()) {
    inputs.emplace_back(column);
  }
  inputs.emplace_back(scalar_column_view{seed_view});

  auto const udf = cuda_udf{
    source,
    nullable ? "cudf::hashing::jit::xxhash_32_nullable_row" : "cudf::hashing::jit::xxhash_32_row",
    cuda_udf_input_mode::ROW_ACCESSOR};
  auto const outputs =
    std::array{transform_output{data_type{type_id::UINT32}, output_nullability::ALL_VALID}};
  auto result  = transform(udf,
                          nullable ? null_aware::YES : null_aware::NO,
                          std::nullopt,
                          inputs,
                          outputs,
                          {},
                          input.num_rows(),
                          stream,
                          mr);
  auto columns = result->release();
  return std::move(columns.front());
}

}  // namespace detail

std::unique_ptr<column> xxhash_32_jit(table_view const& input,
                                      uint32_t seed,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::xxhash_32_jit(input, seed, stream, mr);
}

std::unique_ptr<column> xxhash_32(table_view const& input,
                                  uint32_t seed,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return get_context().use_jit() ? detail::xxhash_32_jit(input, seed, stream, mr)
                                 : detail::xxhash_32(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
