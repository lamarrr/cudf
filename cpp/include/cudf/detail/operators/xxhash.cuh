/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/cmath>
#include <cuda/std/limits>

#include <cstdint>

namespace CUDF_EXPORT cudf {
namespace detail::ops {
namespace xxhash32_detail {

constexpr uint32_t prime1 = 0x9e37'79b1U;
constexpr uint32_t prime2 = 0x85eb'ca77U;
constexpr uint32_t prime3 = 0xc2b2'ae3dU;
constexpr uint32_t prime4 = 0x27d4'eb2fU;
constexpr uint32_t prime5 = 0x1656'67b1U;

__device__ inline uint32_t rotate_left(uint32_t value, int amount)
{
  return (value << amount) | (value >> (32 - amount));
}

__device__ inline uint32_t read32(unsigned char const* data)
{
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

__device__ inline uint32_t round(uint32_t accumulator, uint32_t input)
{
  return rotate_left(accumulator + input * prime2, 13) * prime1;
}

}  // namespace xxhash32_detail

/** @brief Computes xxHash32 over a byte range. */
__device__ inline uint32_t xxhash_32(unsigned char const* data, uint64_t size, uint32_t seed)
{
  using namespace xxhash32_detail;
  auto const* current = data;
  auto remaining      = size;
  uint32_t hash;

  if (size >= 16) {
    uint32_t v1 = seed + prime1 + prime2;
    uint32_t v2 = seed + prime2;
    uint32_t v3 = seed;
    uint32_t v4 = seed - prime1;
    do {
      v1 = round(v1, read32(current));
      current += 4;
      v2 = round(v2, read32(current));
      current += 4;
      v3 = round(v3, read32(current));
      current += 4;
      v4 = round(v4, read32(current));
      current += 4;
      remaining -= 16;
    } while (remaining >= 16);
    hash = rotate_left(v1, 1) + rotate_left(v2, 7) + rotate_left(v3, 12) + rotate_left(v4, 18);
  } else {
    hash = seed + prime5;
  }

  hash += static_cast<uint32_t>(size);
  while (remaining >= 4) {
    hash = rotate_left(hash + read32(current) * prime3, 17) * prime4;
    current += 4;
    remaining -= 4;
  }
  while (remaining > 0) {
    hash = rotate_left(hash + static_cast<uint32_t>(*current) * prime5, 11) * prime1;
    ++current;
    --remaining;
  }
  hash ^= hash >> 15;
  hash *= prime2;
  hash ^= hash >> 13;
  hash *= prime3;
  hash ^= hash >> 16;
  return hash;
}

/** @brief Computes xxHash32 for a scalar value using libcudf hash normalization semantics. */
template <typename T>
__device__ uint32_t xxhash_32(T value, uint32_t seed)
{
  if constexpr (cudf::is_floating_point<T>()) {
    if (value == T{0.0}) { value = T{0.0}; }
    if (cuda::std::isnan(value)) { value = cuda::std::numeric_limits<T>::quiet_NaN(); }
  }
  return xxhash_32(reinterpret_cast<unsigned char const*>(&value), sizeof(value), seed);
}

template <typename Rep>
__device__ uint32_t xxhash_32(numeric::decimal<Rep> value, uint32_t seed)
{
  return xxhash_32(value.value(), seed);
}

__device__ inline uint32_t xxhash_32(bool value, uint32_t seed)
{
  auto const byte = static_cast<uint8_t>(value);
  return xxhash_32(reinterpret_cast<unsigned char const*>(&byte), sizeof(byte), seed);
}

__device__ inline uint32_t xxhash_32(string_view value, uint32_t seed)
{
  return xxhash_32(reinterpret_cast<unsigned char const*>(value.data()), value.size_bytes(), seed);
}

}  // namespace detail::ops
}  // namespace CUDF_EXPORT cudf
