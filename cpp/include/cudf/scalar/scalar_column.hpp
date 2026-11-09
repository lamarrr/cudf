/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
/**
 * @brief An owning scalar with the physical representation of a one-row column.
 *
 * Views borrow its buffers. Device mask writes require updating set_null_count().
 * After release() or move, only destruction and move assignment are supported.
 * Conversions share column storage, but do not permit reinterpret_cast between wrappers.
 */
class scalar_column {
 public:
  /// Adopt a non-null column with exactly one row; otherwise throws std::logic_error.
  explicit scalar_column(std::unique_ptr<column> storage) : storage_{std::move(storage)}
  {
    CUDF_EXPECTS(storage_ != nullptr && storage_->size() == 1,
                 "A scalar column must own exactly one row.",
                 std::logic_error);
  }
  /// Deep copy a view, accounting for its offset.
  explicit scalar_column(
    scalar_column_view view,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : scalar_column(std::make_unique<column>(view.as_column_view(), stream, mr))
  {
  }
  /// Deep copy another owner on the supplied stream and resource.
  scalar_column(scalar_column const& other,
                cuda::stream_ref stream           = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : scalar_column(other.view(), stream, mr)
  {
  }
  scalar_column(scalar_column&&) noexcept            = default;  ///< Transfer ownership.
  scalar_column& operator=(scalar_column const&)     = delete;
  scalar_column& operator=(scalar_column&&) noexcept = default;  ///< Transfer ownership.
  ~scalar_column()                                   = default;

  [[nodiscard]] data_type type() const noexcept { return storage_->type(); }  ///< Logical type.
  [[nodiscard]] size_type size() const noexcept { return 1; }  ///< One top-level row.
  [[nodiscard]] size_type null_count() const { return storage_->null_count(); }  ///< Null count.
  [[nodiscard]] bool nullable() const noexcept { return storage_->nullable(); }  ///< Has a mask.
  [[nodiscard]] bool has_nulls() const { return storage_->has_nulls(); }  ///< Has a null row.
  [[nodiscard]] std::size_t alloc_size() const
  {
    return storage_->alloc_size();
  }  ///< Buffer bytes.
  /// Replace the validity mask and update host metadata, following column semantics.
  void set_null_mask(cuda::device_buffer<bitmask_type>&& mask, size_type count)
  {
    storage_->set_null_mask(std::move(mask), count);
  }
  /// Update host metadata after mask mutation. Does not change mask bits.
  void set_null_count(size_type count) { storage_->set_null_count(count); }
  /// Immutable view, borrowing all buffers.
  [[nodiscard]] scalar_column_view view() const { return scalar_column_view{storage_->view()}; }
  /// Mutable view. The caller must maintain null-count metadata after mask writes.
  [[nodiscard]] mutable_scalar_column_view mutable_view()
  {
    return mutable_scalar_column_view{storage_->mutable_view()};
  }
  /// Release the underlying column without copying buffers.
  [[nodiscard]] std::unique_ptr<column> release() noexcept { return std::move(storage_); }

 private:
  std::unique_ptr<column> storage_;
};
}  // namespace CUDF_EXPORT cudf
