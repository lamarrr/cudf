/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/scalar_column_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @brief A scalar wrapper retaining the representation and accessors of a column device view.
 *
 * The handle returned by create() owns child metadata and must outlive kernel execution.
 * The view itself is copied by value into kernels and borrows the column buffers.
 */
template <typename ColumnView, typename HostView>
class scalar_column_device_view_base : private ColumnView {
 public:
  /// Checked host conversion from an existing device view; borrows its child metadata.
  explicit scalar_column_device_view_base(ColumnView view) : ColumnView(view)
  {
    CUDF_EXPECTS(view.size() == 1, "A scalar device view requires one row.", std::logic_error);
  }
  using ColumnView::child;
  using ColumnView::data;
  using ColumnView::element;
  using ColumnView::head;
  using ColumnView::is_null;
  using ColumnView::is_valid;
  using ColumnView::null_mask;
  using ColumnView::nullable;
  using ColumnView::num_children;
  using ColumnView::offset;
  using ColumnView::size;
  using ColumnView::type;

  /// Access row zero with the same type support as the underlying column view.
  template <typename T>
  [[nodiscard]] __device__ decltype(auto) element() const
  {
    return ColumnView::template element<T>(0);
  }
  /// Read validity directly from the device mask, accounting for offset.
  [[nodiscard]] __device__ bool is_null() const noexcept { return ColumnView::is_null(0); }
  /// Read validity directly from the device mask, accounting for offset.
  [[nodiscard]] __device__ bool is_valid() const noexcept { return ColumnView::is_valid(0); }
  /// Set row zero null. Requires a mutable view with an allocated mask.
  __device__ void set_null() const
    requires(ColumnView::is_mutable)
  {
    ColumnView::set_null(0);
  }
  /// Set row zero valid. Requires a mutable view with an allocated mask.
  __device__ void set_valid() const
    requires(ColumnView::is_mutable)
  {
    ColumnView::set_valid(0);
  }
  /// Explicit conversion; shares buffers and child metadata.
  [[nodiscard]] __host__ __device__ ColumnView const& as_column_view() const noexcept
  {
    return *this;
  }

  /// Create a view and retain the underlying column device view's child metadata.
  static auto create(HostView source,
                     cuda::stream_ref stream           = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    auto backing = ColumnView::create(source.as_column_view(), stream, mr);
    auto result  = new scalar_column_device_view_base(*backing);
    auto deleter = [backing = std::move(backing)](scalar_column_device_view_base* ptr) {
      delete ptr;
    };
    return std::unique_ptr<scalar_column_device_view_base, decltype(deleter)>(result,
                                                                              std::move(deleter));
  }
};
}  // namespace detail
/// Immutable scalar device view, using column device storage semantics.
using scalar_column_device_view =
  detail::scalar_column_device_view_base<column_device_view, scalar_column_view>;
/// Mutable scalar device view. Update owner null-count metadata after validity writes.
using mutable_scalar_column_device_view =
  detail::scalar_column_device_view_base<mutable_column_device_view, mutable_scalar_column_view>;
}  // namespace CUDF_EXPORT cudf
