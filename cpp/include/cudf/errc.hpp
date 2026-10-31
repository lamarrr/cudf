/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

/**
 * @file
 * @brief Error codes for libcudf operations.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief The underlying integer type used for error codes.
 *
 * This type is used to ensure consistent ABI representation with external consumers. e.g. NUMBA's
 * FFI expects an `int` for error codes.
 */
using error_int = int;

/**
 * @brief An enumeration of error codes that can occur during operations.
 */
enum class [[nodiscard]] errc : error_int {
  SUCCESS             = 0,
  ARITHMETIC_OVERFLOW = 1,
  DIVISION_BY_ZERO    = 2,
};

/**
 * @brief Convert an `errc` error code to a human-readable string.
 * @param error The error code to convert
 * @return A C-string representing the error code
 */
[[nodiscard]] constexpr char const* to_string(errc error)
{
  switch (error) {
    case errc::SUCCESS: return "SUCCESS";
    case errc::ARITHMETIC_OVERFLOW: return "ARITHMETIC_OVERFLOW";
    case errc::DIVISION_BY_ZERO: return "DIVISION_BY_ZERO";
    default: return "UNKNOWN_ERROR";
  }
}

/** @} */

}  // namespace CUDF_EXPORT cudf
