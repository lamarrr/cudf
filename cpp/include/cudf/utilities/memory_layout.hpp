/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>

namespace cudf {
namespace detail {

/**
 * @brief Byte extent and alignment of objects placed sequentially in memory.
 *
 * `append()` inserts padding before an appended object to satisfy its alignment.
 * `aligned()` adds tail padding so the resulting size is a multiple of the largest alignment.
 * Apply it after the last member to obtain the aggregate's storage size and array stride.
 *
 * Complete each nested aggregate with `aligned()` before appending subsequent outer members.
 * For example, a nested pair of `double` and `char`, followed by an outer `int` member:
 *
 * @code{.cpp}
 * auto const inner = memory_layout_of<double>.append(memory_layout_of<char>).aligned();
 * auto const outer = inner.append(memory_layout_of<int>).aligned();
 * @endcode
 *
 * A default layout represents zero bytes with alignment one. Use `memory_layout_of<T>`
 * to describe an actual type, including an empty class that still occupies storage.
 */
struct memory_layout {
  size_t alignment = 1;  ///< Required base-address alignment; must be a nonzero power of two.
  size_t size      = 0;  ///< Byte extent, which may omit aggregate tail padding until `aligned()`.

  /**
   * @brief Returns a layout with another object appended at its next aligned offset.
   *
   * The result has the larger alignment and ends immediately after the appended object.
   * Existing padding in either operand is preserved; result tail padding is added by `aligned()`.
   *
   * @param other Layout of the appended object, including its own tail padding
   * @return Combined layout without additional tail padding
   */
  constexpr memory_layout append(memory_layout const& other) const
  {
    auto const new_alignment = std::max(alignment, other.alignment);
    auto const offset        = (size + other.alignment - 1) & ~(other.alignment - 1);
    auto const new_size      = offset + other.size;
    return {.alignment = new_alignment, .size = new_size};
  }

  /**
   * @brief Returns a layout with its size rounded up to a multiple of its alignment.
   *
   * @return Layout with tail padding included and the same alignment
   */
  constexpr memory_layout aligned() const
  {
    auto const new_size = (size + alignment - 1) & ~(alignment - 1);
    return {.alignment = alignment, .size = new_size};
  }
};

/**
 * @brief Layout of `T`, using its compiler-provided alignment and complete size.
 *
 * The size includes any tail padding in `T`.
 *
 * @tparam T Complete object type whose layout is described
 */
template <typename T>
inline constexpr memory_layout memory_layout_of{.alignment = alignof(T), .size = sizeof(T)};

}  // namespace detail
}  // namespace cudf
