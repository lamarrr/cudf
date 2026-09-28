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

/** @brief Extracts every capture group into a separate strings column. */
std::unique_ptr<table> extract(strings_column_view const& input,
                               regex_jit_program const& prog,
                               cuda::stream_ref stream   = cudf::get_default_stream(),
                               cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @brief Extracts all capture-group records into a lists column. */
std::unique_ptr<column> extract_all_record(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @brief Extracts one capture group into a strings column. */
std::unique_ptr<column> extract_single(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type group,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

}  // namespace experimental
}  // namespace CUDF_EXPORT cudf
