/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/experimental/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace experimental {

/** @brief Splits strings from the beginning and returns a table. */
std::unique_ptr<table> split_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type maxsplit        = -1,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @brief Splits strings from the end and returns a table. */
std::unique_ptr<table> rsplit_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type maxsplit        = -1,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @brief Splits strings from the beginning and returns a lists column. */
std::unique_ptr<column> split_record_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type maxsplit        = -1,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @brief Splits strings from the end and returns a lists column. */
std::unique_ptr<column> rsplit_record_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type maxsplit        = -1,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

}  // namespace experimental
}  // namespace CUDF_EXPORT cudf
