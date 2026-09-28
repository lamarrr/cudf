/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/experimental/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>
#include <string_view>

namespace CUDF_EXPORT cudf {
namespace experimental {

/** @brief Replaces matches using a literal replacement string. */
std::unique_ptr<column> replace_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  string_scalar const& replacement           = string_scalar(""),
  std::optional<size_type> max_replace_count = std::nullopt,
  cuda::stream_ref stream                    = cudf::get_default_stream(),
  cudf::memory_resources mr                  = cudf::get_current_device_resource_ref());

/** @brief Replaces matches using a replacement template containing back-references. */
std::unique_ptr<column> replace_with_backrefs(
  strings_column_view const& input,
  regex_jit_program const& prog,
  std::string_view replacement,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

}  // namespace experimental
}  // namespace CUDF_EXPORT cudf
