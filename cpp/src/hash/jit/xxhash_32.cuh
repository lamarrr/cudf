/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/operators/xxhash.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <jit/column_device_view_wrappers.cuh>
#include <jit/type_list.cuh>

namespace cudf::hashing::jit {

constexpr uint32_t null_hash = 0xffff'ffffU;

__device__ inline uint32_t combine(uint32_t lhs, uint32_t rhs)
{
  return lhs ^ (rhs + 0x9e37'79b9U + (lhs << 6) + (lhs >> 2));
}

template <typename T>
__device__ uint32_t hash_value(T value, uint32_t seed)
{
  return cudf::detail::ops::xxhash_32(value, seed);
}

template <typename T>
__device__ uint32_t hash_value(cuda::std::optional<T> const& value, uint32_t seed)
{
  return value.has_value() ? hash_value(*value, seed) : null_hash;
}

template <typename T>
__device__ uint32_t seed_value(T value)
{
  return static_cast<uint32_t>(value);
}

template <typename T>
__device__ uint32_t seed_value(cuda::std::optional<T> const& value)
{
  return static_cast<uint32_t>(value.has_value() ? *value : T{0});
}

template <typename Element>
__device__ uint32_t hash_leaf(column_device_view_core const& column,
                              size_type begin,
                              size_type size,
                              uint32_t seed,
                              uint32_t hash,
                              bool check_nulls)
{
  for (size_type i = 0; i < size; ++i) {
    auto const row          = begin + i;
    auto const element_hash = check_nulls && column.is_null(row)
                                ? null_hash
                                : hash_value(column.template element<Element>(row), seed);
    hash                    = combine(hash, element_hash);
  }
  return hash;
}

template <typename Schema>
__device__ uint32_t hash_range(column_device_view_core const& column,
                               size_type begin,
                               size_type size,
                               uint32_t seed,
                               uint32_t hash,
                               bool check_nulls)
{
  return hash_leaf<Schema>(column, begin, size, seed, hash, check_nulls);
}

template <typename Offset, typename Child>
__device__ uint32_t hash_range(column_device_view_core const&,
                               size_type,
                               size_type,
                               uint32_t,
                               uint32_t,
                               bool,
                               cudf::jit::list_row<Offset, Child>*);

template <typename... Children>
__device__ uint32_t hash_range(column_device_view_core const&,
                               size_type,
                               size_type,
                               uint32_t,
                               uint32_t,
                               bool,
                               cudf::jit::struct_row<cudf::jit::type_list<Children...>>*);

template <typename Offset, typename Child>
__device__ uint32_t hash_range(column_device_view_core const& column,
                               size_type begin,
                               size_type size,
                               uint32_t seed,
                               uint32_t hash,
                               bool check_nulls,
                               cudf::jit::list_row<Offset, Child>*)
{
  if (check_nulls) {
    for (size_type i = 0; i < size; ++i) {
      hash = combine(hash, column.is_valid(begin + i) ? uint32_t{0} : null_hash);
    }
  }

  auto const offsets = column.child(offsets_column_index);
  auto const offset  = [&](size_type row) {
    return static_cast<size_type>(offsets.template element<Offset>(column.offset() + row));
  };
  for (size_type i = 0; i < size; ++i) {
    auto const list_size = offset(begin + i + 1) - offset(begin + i);
    hash                 = combine(hash, hash_value(list_size, uint32_t{0}));
  }

  auto const child_begin = offset(begin);
  auto const child_end   = offset(begin + size);
  auto const child       = column.child(1);
  if constexpr (cudf::jit::is_nested_row<Child>) {
    return hash_range(child,
                      child_begin,
                      child_end - child_begin,
                      seed,
                      hash,
                      check_nulls,
                      static_cast<Child*>(nullptr));
  } else {
    return hash_leaf<Child>(child, child_begin, child_end - child_begin, seed, hash, check_nulls);
  }
}

template <typename... Children>
__device__ uint32_t hash_range(column_device_view_core const& column,
                               size_type begin,
                               size_type size,
                               uint32_t seed,
                               uint32_t hash,
                               bool check_nulls,
                               cudf::jit::struct_row<cudf::jit::type_list<Children...>>*)
{
  if (check_nulls) {
    for (size_type i = 0; i < size; ++i) {
      hash = combine(hash, column.is_valid(begin + i) ? uint32_t{0} : null_hash);
    }
  }
  if constexpr (sizeof...(Children) == 0) {
    return hash;
  } else {
    using child_type       = typename cudf::jit::type_list<Children...>::template at<0>;
    auto const child       = column.child(0);
    auto const child_begin = column.offset() + begin;
    if constexpr (cudf::jit::is_nested_row<child_type>) {
      return hash_range(
        child, child_begin, size, seed, hash, check_nulls, static_cast<child_type*>(nullptr));
    } else {
      return hash_leaf<child_type>(child, child_begin, size, seed, hash, check_nulls);
    }
  }
}

template <typename Offset, typename Child>
__device__ uint32_t hash_value(cudf::jit::list_row<Offset, Child> const& value,
                               uint32_t seed,
                               bool check_nulls)
{
  return hash_range(value.column,
                    value.row,
                    1,
                    seed,
                    uint32_t{0},
                    check_nulls,
                    static_cast<cudf::jit::list_row<Offset, Child>*>(nullptr));
}

template <typename Children>
__device__ uint32_t hash_value(cudf::jit::struct_row<Children> const& value,
                               uint32_t seed,
                               bool check_nulls)
{
  return hash_range(value.column,
                    value.row,
                    1,
                    seed,
                    uint32_t{0},
                    check_nulls,
                    static_cast<cudf::jit::struct_row<Children>*>(nullptr));
}

template <typename T>
__device__ uint32_t hash_value(T const& value, uint32_t seed, bool)
{
  return hash_value(value, seed);
}

template <bool CheckNulls, typename Output, typename Tuple, int... I>
__device__ void hash_row(Output* output,
                         Tuple const& values,
                         cuda::std::integer_sequence<int, I...>)
{
  constexpr auto seed_index = sizeof...(I);
  auto const seed_arg       = cuda::std::get<seed_index>(values);
  auto const seed           = seed_value(seed_arg);
  uint32_t hash             = seed;
  if constexpr (sizeof...(I) > 0) {
    bool first = true;
    (([&] {
       auto const current = hash_value(cuda::std::get<I>(values), seed, CheckNulls);
       hash               = first ? current : combine(hash, current);
       first              = false;
     }()),
     ...);
  }
  *output = hash;
}

template <bool CheckNulls, typename Output, typename... Args>
__device__ void xxhash_32_impl(Output* output, Args... args)
{
  static_assert(sizeof...(Args) >= 1, "xxhash_32 requires a seed argument");
  auto const values = cuda::std::forward_as_tuple(args...);
  hash_row<CheckNulls>(
    output, values, cuda::std::make_integer_sequence<int, sizeof...(Args) - 1>{});
}

template <typename Output, typename... Args>
__device__ void xxhash_32(Output* output, Args... args)
{
  xxhash_32_impl<false>(output, args...);
}

template <typename Output, typename... Args>
__device__ void xxhash_32_nullable(Output* output, Args... args)
{
  xxhash_32_impl<true>(output, args...);
}

template <bool CheckNulls, int Begin, typename Row, int... I>
__device__ __noinline__ uint32_t
hash_accessor_chunk(Row row, uint32_t seed, uint32_t hash, cuda::std::integer_sequence<int, I...>)
{
  (([&] {
     auto const current = hash_value(row.template get<Begin + I>(), seed, CheckNulls);
     if constexpr (Begin + I == 0) {
       hash = current;
     } else {
       hash = combine(hash, current);
     }
   }()),
   ...);
  return hash;
}

template <bool CheckNulls, int Begin = 0, typename Row>
__device__ uint32_t hash_accessor_chunks(Row row, uint32_t seed, uint32_t hash)
{
  constexpr int32_t input_count = Row::size - 1;
  constexpr int32_t chunk_size  = 16;
  if constexpr (Begin >= input_count) {
    return hash;
  } else {
    constexpr int32_t count =
      chunk_size < (input_count - Begin) ? chunk_size : (input_count - Begin);
    auto const next = hash_accessor_chunk<CheckNulls, Begin>(
      row, seed, hash, cuda::std::make_integer_sequence<int, count>{});
    return hash_accessor_chunks<CheckNulls, Begin + count>(row, seed, next);
  }
}

template <bool CheckNulls, typename Output, typename Row>
__device__ void hash_accessor_row(Output* output, Row row)
{
  static_assert(Row::size >= 1, "xxhash_32 requires a seed input");
  auto const seed = seed_value(row.template get<Row::size - 1>());
  if constexpr (Row::size == 1) {
    *output = seed;
  } else {
    *output = hash_accessor_chunks<CheckNulls>(row, seed, seed);
  }
}

template <typename Output, typename Row>
__device__ void xxhash_32_row(Output* output, Row row)
{
  hash_accessor_row<false>(output, row);
}

template <typename Output, typename Row>
__device__ void xxhash_32_nullable_row(Output* output, Row row)
{
  hash_accessor_row<true>(output, row);
}

}  // namespace cudf::hashing::jit
