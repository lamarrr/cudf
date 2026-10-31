/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/udf.hpp>
#include <cudf/user_data.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

/**
 * @file
 * @brief APIs for computing reductions and scans over columns.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup aggregation_reduction
 * @{
 */

/**
 *  @brief Enum to describe scan operation type
 */
enum class scan_type : bool { INCLUSIVE, EXCLUSIVE };

// clang-format off
/**
 * @brief  Computes the reduction of the values in all rows of a column.
 *
 * This function does not detect overflows in reductions except for the `SUM_OVERFLOW`
 * aggregation. When `output_type` does not match the `col.type()`, their values may be promoted to
 * `int64_t` or `double` for computing aggregations and then cast to `output_type` before returning.
 *
 * The `SUM_OVERFLOW` aggregation is a special case that detects integer
 * overflow during summation of signed integer or decimal values and returns a struct
 * containing both the sum result and an overflow flag. On overflow the sum value
 * is unspecified; the boolean flag is the source of truth.
 *
 * Only `min` and `max` ops are supported for reduction of non-arithmetic
 * types (e.g. timestamp or string).
 *
 * Any null values are skipped for the operation.
 * If the reduction fails, the output scalar returns with `%is_valid()==false`.
 *
 * For empty or all-null input, the result is generally an invalid scalar except for specific
 * aggregations where the aggregation has a well-defined output.
 *
 * If the input column is an arithmetic type, the `output_type` can be any arithmetic
 * type. If the input column is a non-arithmetic type (e.g. timestamp or string)
 * the `output_type` must match the `col.type()`. If the reduction type is `any` or
 * `all`, the `output_type` must be type BOOL8.
 *
 * | Aggregation | Output Type | Init Value | Empty Input | Comments |
 * | :---------: | ----------- | :--------: | ----------- | -------- |
 * | SUM/PRODUCT | output_type | yes | NA | Input accumulated into output_type variable |
 * | SUM_OVERFLOW | STRUCT{col.type,BOOL8} | yes | {null,false} | {sum, overflow_flag}, input must be signed integer or decimal |
 * | SUM_OF_SQUARES | output_type | no | NA | Input accumulated into output_type variable |
 * | MIN/MAX | col.type | yes | NA | Supports arithmetic, timestamp, duration, string types only |
 * | ANY/ALL | BOOL8 | yes | True for ALL only | Checks for non-zero elements |
 * | MEAN/VARIANCE/STD | FLOAT32/FLOAT64 | no | NA | output_type must be a float type |
 * | MEDIAN/QUANTILE | output_type | no | NA | Exact value if output_type is FLOAT64. See @ref cudf::quantile |
 * | NUNIQUE | output_type | no | 1 if all-nulls | May process null rows |
 * | NTH_ELEMENT | col.type | no | NA |  |
 * | BITWISE_AGG | col.type | no | NA | Supports only integral types |
 * | HISTOGRAM/MERGE_HISTOGRAM | LIST of col.type | no | empty list returned |  |
 * | COLLECT_LIST/COLLECT_SET | LIST of col.type | no | empty list returned |  |
 * | TDIGEST/MERGE_TDIGEST | STRUCT | no | empty struct returned | tdigest scalar is returned |
 * | HOST_UDF | output_type | yes | NA | Custom UDF could ignore output_type |
 *
 * The NA in the table indicates an output scalar with `%is_valid()==false`
 *
 * @throw std::invalid_argument if reduction is called for non-arithmetic output
 * type and operator other than `min` and `max`.
 * @throw std::invalid_argument if input column data type is not convertible to `output_type`.
 * @throw std::invalid_argument if `min` or `max` reduction is called and the
 * output type does not match the input column data type.
 * @throw std::invalid_argument if `any` or `all` reduction is called and the output type is not BOOL8.
 * @throw std::invalid_argument if `mean`, `var`, or `std` reduction is called and
 * the `output_type` is not floating point.
 * @throw std::invalid_argument if `sum_overflow` reduction is called and the
 * input column type is not a signed integer or decimal, or the `output_type` is not `STRUCT`.
 *
 * @param col Input column view
 * @param agg Aggregation operator applied by the reduction
 * @param output_type The output scalar type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar with reduce result
 */
std::unique_ptr<scalar> reduce(
  column_view const& col,
  reduce_aggregation const& agg,
  data_type output_type,
  cuda::stream_ref stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
// clang-format on

/**
 * @brief  Computes the reduction of the values in all rows of a column with an initial value
 *
 * Only `sum`, `product`, `min`, `max`, `any`, `all`, and `sum_overflow` reductions are
 * supported. For `sum_overflow`, the initial value is added to the sum and overflow
 * detection is performed throughout the entire computation.
 *
 * @see cudf::reduce(column_view const&,reduce_aggregation
 * const&,data_type,cuda::stream_ref,rmm::device_async_resource_ref) for more details
 *
 * @throw std::invalid_argument if reduction is not `sum`, `product`, `min`, `max`, `any`, `all`,
 * or `sum_overflow` and `init` is specified.
 *
 * @param col Input column view
 * @param agg Aggregation operator applied by the reduction
 * @param output_type The output scalar type
 * @param init The initial value of the reduction
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar with reduce result
 */
std::unique_ptr<scalar> reduce(
  column_view const& col,
  reduce_aggregation const& agg,
  data_type output_type,
  std::optional<std::reference_wrapper<scalar const>> init,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Compute reduction of each segment in the input column
 *
 * This function does not detect overflows in reductions. When `output_type`
 * does not match the `segmented_values.type()`, their values may be promoted to
 * `int64_t` or `double` for computing aggregations and then cast to
 * `output_type` before returning.
 *
 * Null values are treated as identities during reduction.
 *
 * If the segment is empty, the row corresponding to the result of the
 * segment is null.
 *
 * If any index in `offsets` is out of bound of `segmented_values`, the behavior
 * is undefined.
 *
 * If the input column has arithmetic type, `output_type` can be any arithmetic
 * type. If the input column has non-arithmetic type, e.g. timestamp, the same
 * output type must be specified.
 *
 * If input is not empty, the result is always nullable.
 *
 * @throw cudf::logic_error if reduction is called for non-arithmetic output
 * type and operator other than `min` and `max`.
 * @throw cudf::logic_error if input column data type is not convertible to
 * `output_type` type.
 * @throw cudf::logic_error if `min` or `max` reduction is called and the
 * `output_type` does not match the input column data type.
 * @throw cudf::logic_error if `any` or `all` reduction is called and the
 * `output_type` is not BOOL8.
 *
 * @param segmented_values Column view of segmented inputs
 * @param offsets Each segment's offset of `segmented_values`. A list of offsets with size
 * `num_segments + 1`. The size of `i`th segment is `offsets[i+1] - offsets[i]`.
 * @param agg Aggregation operator applied by the reduction
 * @param output_type  The output column type
 * @param null_handling If `INCLUDE`, the reduction is valid if all elements in a segment are valid,
 * otherwise null. If `EXCLUDE`, the reduction is valid if any element in the segment is valid,
 * otherwise null.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output column with results of segmented reduction
 */
std::unique_ptr<column> segmented_reduce(
  column_view const& segmented_values,
  device_span<size_type const> offsets,
  segmented_reduce_aggregation const& agg,
  data_type output_type,
  null_policy null_handling,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Compute reduction of each segment in the input column with an initial value. Only SUM,
 * PRODUCT, MIN, MAX, ANY, and ALL aggregations are supported.
 *
 * @param segmented_values Column view of segmented inputs
 * @param offsets Each segment's offset of `segmented_values`. A list of offsets with size
 * `num_segments + 1`. The size of `i`th segment is `offsets[i+1] - offsets[i]`.
 * @param agg Aggregation operator applied by the reduction
 * @param output_type  The output column type
 * @param null_handling If `INCLUDE`, the reduction is valid if all elements in a segment are valid,
 * otherwise null. If `EXCLUDE`, the reduction is valid if any element in the segment is valid,
 * otherwise null.
 * @param init The initial value of the reduction
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output column with results of segmented reduction.
 */
std::unique_ptr<column> segmented_reduce(
  column_view const& segmented_values,
  device_span<size_type const> offsets,
  segmented_reduce_aggregation const& agg,
  data_type output_type,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Computes the scan of a column.
 *
 * The null values are skipped for the operation, and if an input element
 * at `i` is null, then the output element at `i` will also be null.
 *
 * @throws cudf::logic_error if column datatype is not numeric type.
 *
 * @param[in] input The input column view for the scan
 * @param[in] agg unique_ptr to aggregation operator applied by the scan
 * @param[in] inclusive The flag for applying an inclusive scan if scan_type::INCLUSIVE, an
 * exclusive scan if scan_type::EXCLUSIVE.
 * @param[in] null_handling Exclude null values when computing the result if null_policy::EXCLUDE.
 * Include nulls if null_policy::INCLUDE. Any operation with a null results in a null.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Scanned output column
 */
std::unique_ptr<column> scan(
  column_view const& input,
  scan_aggregation const& agg,
  scan_type inclusive,
  null_policy null_handling         = null_policy::EXCLUDE,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Determines the minimum and maximum values of a column.
 *
 *
 * @param col column to compute minmax
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A std::pair of scalars with the first scalar being the minimum value and the second
 * scalar being the maximum value of the input column.
 */
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  column_view const& col,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reduction namespace
 */
namespace reduction {
/**
 * @brief Indicate if a reduction is supported for a source datatype.
 *
 * @param source The source data type.
 * @param kind The reduction aggregation.
 * @returns true if the reduction is supported.
 */
bool is_valid_aggregation(data_type source, aggregation::Kind kind);
}  // namespace reduction

/**
 * @brief An input to a user-defined reduction.
 *
 * A `column_view` supplies one value per input row. A `scalar_column_view` broadcasts its
 * single value to every row. A `user_data` input passes the same `void*` to every update
 * invocation, at its specified position in the input parameter list.
 * Column and scalar inputs support the same types as `transform`, including dictionary-encoded
 * inputs. Dictionary inputs supply decoded values to the callbacks.
 *
 * These alternatives do not own the referenced device storage. The caller must keep that
 * storage accessible to device code until operations using it have completed on the execution
 * stream.
 */
using reduce_input = std::variant<column_view, scalar_column_view, user_data>;

/**
 * @brief Compiled device functions defining a user-defined reduction.
 *
 * The named functions must be defined in `fragments` and return `cudf::errc`, using
 * `cudf::errc::SUCCESS` to indicate success. Their parameters follow these conventions:
 *
 * - `init` takes pointers to all state fields, in `state_types` order, and initializes every
 *   field to the reduction's identity state.
 * - Optional `filter` takes the input row index, a `bool*` result, and input values in `inputs`
 *   order. It writes whether to include the row. A successful true result invokes `update`;
 *   false or an error skips the update. An empty filter symbol includes every row.
 * - `update` takes the input row index as a leading `cudf::size_type`, followed by pointers
 *   to all state fields and then input values in `inputs` order. It updates the state for
 *   that row.
 * - `merge` takes pointers to the destination state fields, followed by source state fields
 *   passed by value, both in `state_types` order. It combines independently accumulated
 *   partial states. The merge operation must be associative and preserve the identity state;
 *   the order and grouping of updates and merges are unspecified.
 * - `finalize` takes pointers to the output values, in `outputs` order, followed by state
 *   fields passed by value, in `state_types` order. It writes one result row.
 *
 * For `segmented_reduce_lto`, every callback takes an additional leading `cudf::size_type`
 * containing the zero-based segment ID. The update signature therefore starts with
 * `(cudf::size_type segment_id, cudf::size_type row_index, ...)`. The row index is relative
 * to the input columns, rather than the start of the segment. The segment ID is also passed
 * when initializing and finalizing empty segments. Initialization and merging may each run
 * multiple times for the same segment.
 *
 * Only `filter` and `update` take a row index. Both receive the same input values, loaded once
 * per row. The filter's `bool*` result is never wrapped in an optional. Packed selections can
 * be supplied through `user_data` and interpreted by the filter.
 * Inputs use the same decoded value types as `transform`; string inputs use `cudf::string_view`.
 * State and output types may be fixed-width or `STRING`, represented by `cudf::string_view`.
 * String views in state and outputs must refer to device character storage that remains valid
 * until the operation completes. Returned string columns own a copy of those characters.
 *
 * With `null_aware::YES`, column and scalar inputs and all state fields use
 * `cuda::std::optional<T>`. State pointers in `init`, `update`, and `merge`, and output pointers
 * in `finalize`, therefore use `cuda::std::optional<T>*`; source state fields in `merge` and
 * `finalize` are passed as optional values. An empty optional represents a null value; leaving
 * an output optional empty produces a null output. Callbacks define each state's identity and
 * how empty state values combine. User-data inputs remain `void*`. With `null_aware::NO`, inputs,
 * state fields, and outputs use their value types, pointer parameters remain pointers, and rows
 * with any null column or scalar input are skipped before filtering.
 *
 * The fragment span, fragment bytes, and symbol names are non-owning views. Their backing
 * storage must remain valid for the duration of the reduction call. See `lto_fragment` for device
 * function linkage and parameter type conventions.
 */
struct reduce_udf {
  lto_fragments fragments{};    ///< Compiled fragments defining the reduction callbacks
  std::string_view init{};      ///< Symbol for initializing an independent reduction state
  std::string_view update{};    ///< Symbol for incorporating one selected input row into a state
  std::string_view merge{};     ///< Symbol for combining two partial reduction states
  std::string_view finalize{};  ///< Symbol for converting a reduced state into output values
  std::string_view filter{};    ///< Optional symbol for selecting rows before updating state
};

/**
 * @brief Reduces selected input rows using compiled user-defined device functions.
 *
 * Column inputs must have the same number of rows. Scalar and user-data inputs are broadcast
 * to every row. The optional filter callback selects rows before updating the reduction state.
 * Null handling and callback signatures follow `reduce_udf`. With no column inputs, the
 * reduction has zero rows and finalizes the identity state.
 *
 * Input column types follow `transform`, including strings and dictionary-encoded inputs.
 * State fields and output columns support fixed-width types and strings. String state fields
 * and callback outputs use `cudf::string_view`; returned string columns own their characters.
 *
 * The result contains one column per entry in `outputs`, in the same order. Each result column
 * has one row. An empty input, or an input with no selected rows, is finalized from the identity
 * state. The finalization callback determines the output values and, in null-aware mode, their
 * validity.
 *
 * @throws std::invalid_argument if input lengths differ or a type is unsupported
 * @throws cudf::evaluation_error if a callback returns an error code other than `SUCCESS`
 *
 * @param udf Compiled fragments and symbols for initialization, filtering, update, merge, and
 * finalization
 * @param is_null_aware Whether inputs, state fields, and outputs use optional values
 * @param inputs Column, broadcast scalar, and user-data inputs in update parameter order
 * @param state_types Logical types of the state fields in callback parameter order
 * @param outputs Logical types of the returned columns in finalization parameter order
 * @param stream CUDA stream used for the operation
 * @param mr Memory resources used for output and temporary allocations
 * @return One single-row column for each requested output type
 */
std::vector<std::unique_ptr<column>> reduce_lto(
  reduce_udf const& udf,
  null_aware is_null_aware,
  std::span<reduce_input const> inputs,
  std::span<data_type const> state_types,
  std::span<data_type const> outputs,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reduces selected rows of each segment using compiled user-defined device functions.
 *
 * Segment `i` covers input rows `[offsets[i], offsets[i + 1])`. Offsets must contain at least
 * one entry, be non-negative and nondecreasing, and lie within the input columns' row range.
 * Adjacent equal offsets describe an empty segment. The first offset may be nonzero, and the
 * last offset need not include every input row.
 *
 * Column inputs must have the same number of rows. Scalar and user-data inputs are broadcast
 * to every row. The optional filter uses input row indices, including for segments beginning
 * at nonzero offsets. Null handling and callback signatures follow `reduce_udf`, including a
 * leading segment ID on every callback.
 *
 * The result contains `outputs.size() * (offsets.size() - 1)` independent single-row columns,
 * ordered by segment and then by output: `result[segment * outputs.size() + output]` contains
 * that segment's value for the requested output type. Zero segments return an empty vector.
 * Empty segments and segments with no selected rows are finalized from the identity state.
 * The finalization callback determines the output values and, in null-aware mode, their validity.
 *
 * Input, state, and output types have the same requirements as the scalar UDF reduction.
 * When there are no column inputs, offsets specify the input row range. A supplied maximum
 * segment size must be nonnegative and at least as large as every segment.
 *
 * @throws std::invalid_argument if input lengths differ, offsets are empty or contain too many
 * segments, the maximum segment size is negative, or a type is unsupported
 * @throws cudf::evaluation_error if a callback returns an error code other than `SUCCESS`
 *
 * @param udf Compiled fragments and symbols for initialization, filtering, update, merge, and
 * finalization
 * @param is_null_aware Whether inputs, state fields, and outputs use optional values
 * @param inputs Column, broadcast scalar, and user-data inputs in update parameter order
 * @param offsets Device-accessible segment boundaries, with `num_segments + 1` entries
 * @param state_types Logical types of the state fields in callback parameter order
 * @param outputs Logical types of the returned columns in finalization parameter order
 * @param max_segment_size Upper bound on segment length, or `std::nullopt` when unknown;
 * supplied as an optimization hint
 * @param stream CUDA stream used for the operation
 * @param mr Memory resources used for output and temporary allocations
 * @return One single-row column per segment and requested output type, ordered by segment
 */
std::vector<std::unique_ptr<column>> segmented_reduce_lto(
  reduce_udf const& udf,
  null_aware is_null_aware,
  std::span<reduce_input const> inputs,
  device_span<size_type const> offsets,
  std::span<data_type const> state_types,
  std::span<data_type const> outputs,
  std::optional<size_type> max_segment_size,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
