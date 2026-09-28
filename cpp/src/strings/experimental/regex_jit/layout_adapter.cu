/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <cstdint>

namespace {

using string_index_pair = cudf::strings::detail::string_index_pair;

template <typename Offset>
__device__ std::int64_t load_offset(void const* offsets, cudf::size_type index)
{
  return static_cast<std::int64_t>(static_cast<Offset const*>(offsets)[index]);
}

}  // namespace

extern "C" __device__ cudf::size_type cudf_regex_row_index()
{
  return static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
}

extern "C" __device__ bool cudf_regex_is_valid(cudf::bitmask_type const* mask,
                                                 cudf::size_type row)
{
  return mask == nullptr || cudf::bit_is_set(mask, row);
}

extern "C" __device__ std::int64_t cudf_regex_load_offset_i32(void const* offsets,
                                                               cudf::size_type index)
{
  return load_offset<std::int32_t>(offsets, index);
}

extern "C" __device__ std::int64_t cudf_regex_load_offset_i64(void const* offsets,
                                                               cudf::size_type index)
{
  return load_offset<std::int64_t>(offsets, index);
}

extern "C" __device__ void cudf_regex_write_pair(void* output,
                                                  std::int64_t index,
                                                  char const* data,
                                                  char const* empty,
                                                  std::int64_t begin,
                                                  std::int64_t end,
                                                  bool present)
{
  auto size    = static_cast<cudf::size_type>(end - begin);
  auto pointer = size == 0 ? empty : data + begin;
  static_cast<string_index_pair*>(output)[index] =
    present ? string_index_pair{pointer, size} : string_index_pair{nullptr, 0};
}
