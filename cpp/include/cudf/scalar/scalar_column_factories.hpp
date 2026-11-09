/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda_runtime_api.h>

#include <span>
#include <string>

namespace CUDF_EXPORT cudf {
namespace scalar_column_factories {
/// Construct an invalid numeric scalar; fixed-width payload is uninitialized.
std::unique_ptr<scalar_column> make_numeric_scalar(
  data_type type,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct an invalid timestamp scalar; fixed-width payload is uninitialized.
std::unique_ptr<scalar_column> make_timestamp_scalar(
  data_type type,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct an invalid duration scalar; fixed-width payload is uninitialized.
std::unique_ptr<scalar_column> make_duration_scalar(
  data_type type,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct an invalid fixed width scalar; fixed-width payload is uninitialized.
std::unique_ptr<scalar_column> make_fixed_width_scalar(
  data_type type,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct an invalid default constructed scalar with a zero-initialized fixed-width payload.
std::unique_ptr<scalar_column> make_default_constructed_scalar(
  data_type type,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct a column-backed string scalar using the supplied stream and resource.
std::unique_ptr<scalar_column> make_string_scalar(
  std::string const& value,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct a column-backed list scalar using the supplied stream and resource.
std::unique_ptr<scalar_column> make_list_scalar(
  column_view elements,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct a column-backed struct scalar using the supplied stream and resource.
std::unique_ptr<scalar_column> make_struct_scalar(
  table_view const& fields,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct a column-backed struct scalar using the supplied stream and resource.
std::unique_ptr<scalar_column> make_struct_scalar(
  std::span<column_view const> fields,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/// Construct a column-backed empty scalar using the supplied stream and resource.
std::unique_ptr<scalar_column> make_empty_scalar_like(
  column_view input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
namespace detail {
/// Copy a host value into one-row column storage. Completes the host copy before returning.
template <typename T>
std::unique_ptr<scalar_column> from_host(T value,
                                         data_type type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto data = rmm::device_buffer{&value, sizeof(T), stream, mr};
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return std::make_unique<scalar_column>(
    std::make_unique<column>(type, 1, std::move(data), cuda::device_buffer<bitmask_type>{}, 0));
}
}  // namespace detail
/// Construct a valid fixed-width scalar from a host value.
template <typename T>
std::unique_ptr<scalar_column> make_fixed_width_scalar(
  T value,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  static_assert(is_fixed_width<T>() && !is_fixed_point<T>());
  return detail::from_host(value, data_type{type_to_id<T>()}, stream, mr);
}
/// Construct a valid decimal scalar, preserving its representation and scale.
template <typename T>
std::unique_ptr<scalar_column> make_fixed_point_scalar(
  typename T::rep value,
  numeric::scale_type scale,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  static_assert(is_fixed_point<T>());
  return detail::from_host(
    value, data_type{type_to_id<T>(), static_cast<int32_t>(scale)}, stream, mr);
}
}  // namespace cudf::scalar_column_factories
