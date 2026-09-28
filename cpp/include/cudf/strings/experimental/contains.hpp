/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/experimental/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace experimental {

/** @brief Tests whether each input string contains a match. */
std::unique_ptr<column> contains_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @brief Tests whether a match begins at the start of each input string. */
std::unique_ptr<column> matches_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @brief Counts non-overlapping matches in each input string. */
std::unique_ptr<column> count_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

}  // namespace experimental
}  // namespace CUDF_EXPORT cudf
