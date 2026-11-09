/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <limits>

namespace cudf::scalar_column_factories {
std::unique_ptr<scalar_column> make_numeric_scalar(data_type type,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  return std::make_unique<scalar_column>(
    make_numeric_column(type, 1, mask_state::ALL_NULL, stream, mr));
}
std::unique_ptr<scalar_column> make_timestamp_scalar(data_type type,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  return std::make_unique<scalar_column>(
    make_timestamp_column(type, 1, mask_state::ALL_NULL, stream, mr));
}
std::unique_ptr<scalar_column> make_duration_scalar(data_type type,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  return std::make_unique<scalar_column>(
    make_duration_column(type, 1, mask_state::ALL_NULL, stream, mr));
}
std::unique_ptr<scalar_column> make_fixed_width_scalar(data_type type,
                                                       cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  return std::make_unique<scalar_column>(
    make_fixed_width_column(type, 1, mask_state::ALL_NULL, stream, mr));
}
namespace {
std::unique_ptr<column> offsets(size_type end,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  size_type values[] = {0, end};
  auto data          = rmm::device_buffer{values, sizeof(values), stream, mr};
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return std::make_unique<column>(
    data_type{type_id::INT32}, 2, std::move(data), cuda::device_buffer<bitmask_type>{}, 0);
}
void invalidate(scalar_column& value, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto storage = value.release();
  storage->set_null_mask(create_null_mask(1, mask_state::ALL_NULL, stream, mr), 1);
  value = scalar_column{std::move(storage)};
}
}  // namespace
std::unique_ptr<scalar_column> make_string_scalar(std::string const& value,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(value.size() <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "String exceeds scalar size limit.",
               std::overflow_error);
  auto chars = rmm::device_buffer{value.data(), value.size(), stream, mr};
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return std::make_unique<scalar_column>(
    make_strings_column(1,
                        offsets(static_cast<size_type>(value.size()), stream, mr),
                        std::move(chars),
                        0,
                        cuda::device_buffer<bitmask_type>{}));
}
std::unique_ptr<scalar_column> make_list_scalar(column_view elements,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  return std::make_unique<scalar_column>(
    make_lists_column(1,
                      offsets(elements.size(), stream, mr),
                      std::make_unique<column>(elements, stream, mr),
                      0,
                      cuda::device_buffer<bitmask_type>{}));
}
std::unique_ptr<scalar_column> make_struct_scalar(std::span<column_view const> fields,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> children;
  for (auto const& field : fields) {
    CUDF_EXPECTS(field.size() == 1, "Struct scalar fields must contain one row.");
    children.push_back(std::make_unique<column>(field, stream, mr));
  }
  return std::make_unique<scalar_column>(make_structs_column(
    1, std::move(children), 0, cuda::device_buffer<bitmask_type>{}, stream, mr));
}
std::unique_ptr<scalar_column> make_struct_scalar(table_view const& fields,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  return make_struct_scalar(std::span<column_view const>{fields.begin(), fields.end()}, stream, mr);
}
std::unique_ptr<scalar_column> make_default_constructed_scalar(data_type type,
                                                               cuda::stream_ref stream,
                                                               rmm::device_async_resource_ref mr)
{
  if (type.id() == type_id::STRING) {
    auto result = make_string_scalar("", stream, mr);
    invalidate(*result, stream, mr);
    return result;
  }
  auto result = make_fixed_width_scalar(type, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(result->mutable_view().head<void>(), 0, cudf::size_of(type), stream.get()));
  return result;
}
std::unique_ptr<scalar_column> make_empty_scalar_like(column_view input,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  if (input.type().id() == type_id::LIST) {
    auto child  = empty_like(lists_column_view{input}.child());
    auto result = make_list_scalar(child->view(), stream, mr);
    invalidate(*result, stream, mr);
    return result;
  }
  if (input.type().id() == type_id::STRUCT) {
    CUDF_EXPECTS(input.size() > 0, "Cannot extract a struct scalar from an empty column.");
    auto result = std::make_unique<scalar_column>(
      scalar_column_view{slice(input, {0, 1}, stream)[0]}, stream, mr);
    invalidate(*result, stream, mr);
    return result;
  }
  return make_default_constructed_scalar(input.type(), stream, mr);
}
}  // namespace cudf::scalar_column_factories
