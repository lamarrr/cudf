/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace rtcx {
struct blob_t;
}  // namespace rtcx

namespace CUDF_EXPORT cudf {

/**
 * @brief Binary Fragment Types for compiled CUDA programs.
 */
enum class lto_binary_type : uint8_t {
  LTO_IR = 0,  ///< LTO intermediate representation
  FATBIN = 1,  ///< Fatbinary containing device code
  PTX    = 2   ///< PTX source
};

/**
 * @brief A pre-compiled LTO binary fragment.
 *
 * UDFs defined in a fragment must be CUDA device functions whose parameter types match
 * the input and output specifications of the calling operation.
 *
 * An error-returning UDF must return `cudf::errc`: return `cudf::errc::SUCCESS` on success,
 * or another error code
 * to report failure. Row results are written through output parameters, rather than
 * returned from the function.
 *
 * Only row-specific UDFs take a leading `cudf::size_type` parameter containing the row index.
 * Other UDFs omit this parameter.
 *
 * After the index, when present, specify all output parameters as pointers to their output
 * value types, followed by all input parameters, passed by value. Preserve the order of
 * the output and input specifications within each group.
 *
 * User-data inputs map to `void*` and appear in their specified input positions.
 *
 * For example, a row-specific UDF with one `INT32` output, two `INT32` inputs, and a user-data
 * input has the following signature:
 *
 * @code{.cpp}
 * #include <cudf/errc.hpp>
 * #include <cudf/types.hpp>
 * #include <cstdint>
 *
 * extern "C" __device__ cudf::errc add(
 *   cudf::size_type row_index,
 *   int32_t* result,
 *   int32_t lhs,
 *   int32_t rhs,
 *   void* user_data);
 * @endcode
 *
 * String parameters use the following value types:
 *
 * - A string input is passed by value as `cudf::string_view`.
 *
 * - A string output with supplied offsets is passed as `cudf::mutable_string_view*`.
 *   The view refers to the row's preallocated character range. Write within that range;
 *   the UDF cannot change its offsets or length.
 *
 * - A string output without supplied offsets is passed as `cudf::string_view*`.
 *   Assign a view of the result's character data for subsequent materialization into the
 *   output column. The referenced storage must remain valid until materialization completes.
 *
 * Nullable inputs and outputs should have their value types wrapped in `cuda::std::optional`.
 *
 * For row-specific UDFs, the leading row index is passed directly, without an optional wrapper.
 *
 * Inputs remain values (`cuda::std::optional<T>`), and outputs remain pointers
 * (`cuda::std::optional<T>*`), with the pointer outside the optional.
 *
 * An empty input optional represents a null input; assign an empty output optional to produce
 * a null output. This also applies to string views.
 *
 * For example, these row-specific UDFs
 * accept a nullable string input and produce a nullable string output:
 *
 * @code{.cpp}
 * #include <cudf/errc.hpp>
 * #include <cudf/strings/mutable_string_view.hpp>
 * #include <cudf/strings/string_view.cuh>
 * #include <cudf/types.hpp>
 * #include <cuda/std/optional>
 *
 * // Output without supplied offsets.
 * extern "C" __device__ cudf::errc string_result(
 *   cudf::size_type row_index,
 *   cuda::std::optional<cudf::string_view>* result,
 *   cuda::std::optional<cudf::string_view> input);
 *
 * // Output with supplied offsets and preallocated character storage.
 * extern "C" __device__ cudf::errc write_string_result(
 *   cudf::size_type row_index,
 *   cuda::std::optional<cudf::mutable_string_view>* result,
 *   cuda::std::optional<cudf::string_view> input);
 * @endcode
 *
 * Declare the UDF with `extern "C"` as recommended in these examples to prevent C++
 * symbol mangling. Use the resulting symbol name when specifying the function to invoke.
 *
 */
struct lto_fragment {
  lto_binary_type type = lto_binary_type::LTO_IR;  ///< Type of the LTO fragment

  std::span<uint8_t const> data{};  ///< Binary data of the fragment
};

/// @brief Pre-compiled LTO binary fragments
using lto_fragments = std::span<lto_fragment const>;

/**
 * @brief Shares ownership of a compiled device-code fragment.
 *
 * Copies share the underlying storage. Byte views do not own storage; keep an owner
 * alive while using a view, including when passing it through an `lto_fragment`.
 */
struct [[nodiscard]] compiled_fragment {
  /**
   * @brief Returns a read-only view of the compiled bytes.
   * @return Compiled fragment bytes, or an empty span if this object has no storage.
   */
  [[nodiscard]] std::span<uint8_t const> view() const noexcept;

  /**
   * @brief Converts to a read-only view of the compiled bytes.
   * @return The same byte view as `view()`.
   */
  [[nodiscard]] operator std::span<uint8_t const>() const noexcept { return view(); }

 private:
  std::shared_ptr<rtcx::blob_t> blob_;

  friend compiled_fragment compile_cuda_to_lto_ir(char const*,
                                                  bool,
                                                  std::span<char const*>,
                                                  std::span<char const*>,
                                                  std::span<char const* const>);
};

/**
 * @brief Compiles CUDA device source to an owning LTO IR binary.
 *
 * @param source_code Null-terminated CUDA source code.
 * @param includes_cudf Whether to make the bundled cuDF headers available during compilation.
 * @param include_names Null-terminated names of additional headers provided for compilation.
 * @param includes Null-terminated contents of the additional headers, in the same order as
 * `include_names`. Both spans must have the same size.
 * @param include_paths Additional directories to search for filesystem headers, in search order
 * before the bundled include directories. Each path must be null-terminated and supplied without
 * an `-I` prefix.
 * @return An object owning the compiled LTO IR bytes.
 * @throws std::invalid_argument if `source_code` is null or the header spans differ in size.
 * @throws std::runtime_error if compilation fails.
 */
compiled_fragment compile_cuda_to_lto_ir(char const* source_code,
                                         bool includes_cudf,
                                         std::span<char const*> include_names,
                                         std::span<char const*> includes,
                                         std::span<char const* const> include_paths);

}  // namespace CUDF_EXPORT cudf
