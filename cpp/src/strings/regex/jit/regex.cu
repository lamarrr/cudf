/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/cache.hpp"
#include "regex_ir.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/experimental/strings/regex.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/utility>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudf::experimental {

struct retained_kernel {
  kernel value;
  std::uint32_t threads;
};

struct regex_jit_program::regex_jit_program_impl {
  size_type capture_count;
  std::vector<retained_kernel> kernels;
  std::optional<std::string> validation_error;
};

struct regex_jit_program_accessor {
  static retained_kernel const& kernel(regex_jit_program const& prog,
                                       bool offset64,
                                       bool emit            = false,
                                       bool output_offset64 = false);
  static regex_jit_program_options const& options(regex_jit_program const& prog)
  {
    return prog._options;
  }
  static std::optional<std::string> const& validation_error(regex_jit_program const& prog)
  {
    return prog._impl->validation_error;
  }
};

namespace {

using pair_t = cuda::std::pair<char const*, size_type>;

inline constexpr std::string_view KERNEL_ENTRY = "cudf_kernel_entry";

static_assert(sizeof(pair_t) == 16);
static_assert(alignof(pair_t) == alignof(char const*));

constexpr std::size_t required_stack_size = 64U * 1024U;

std::uint32_t block_size(regex_ir::operation_kind operation, std::string_view matcher)
{
  auto literal = matcher.find("; executor: single-byte literal scan") != std::string_view::npos ||
                 matcher.find("; executor: packed ASCII literal scan") != std::string_view::npos;
  // Boolean and direct-literal scans benefit from retaining each thread's sequential input window
  // in L1. Complex ordered span operations are divergence-limited and need smaller blocks for
  // occupancy and scheduling.
  return !literal && (operation == regex_ir::operation_kind::COUNT ||
                      operation == regex_ir::operation_kind::REPLACE ||
                      operation == regex_ir::operation_kind::SPLIT)
           ? 256
           : 1024;
}

struct input_data {
  char const* chars;
  void const* offsets;
  bitmask_type const* validity;
  size_type row_offset;
  size_type rows;
  bool offset64;
};

input_data get_input_data(strings_column_view const& input, cuda::stream_ref stream)
{
  auto const offsets = input.offsets();
  CUDF_EXPECTS(offsets.type().id() == type_id::INT32 || offsets.type().id() == type_id::INT64,
               "Unsupported strings offset type");
  return input_data{input.chars_begin(stream),
                    offsets.type().id() == type_id::INT64
                      ? static_cast<void const*>(offsets.data<std::int64_t>())
                      : static_cast<void const*>(offsets.data<std::int32_t>()),
                    input.parent().null_mask(),
                    input.offset(),
                    input.size(),
                    offsets.type().id() == type_id::INT64};
}

std::string normalize_pattern(std::string_view pattern)
{
  std::string result;
  result.reserve(pattern.size());
  for (std::size_t position = 0; position < pattern.size();) {
    if (pattern[position] != '\\' || position + 1 == pattern.size()) {
      result.push_back(pattern[position++]);
      continue;
    }

    auto const escaped = pattern[position + 1];
    if (escaped < '0' || escaped > '7') {
      result.append(pattern.substr(position, 2));
      position += 2;
      continue;
    }

    position += 1;
    std::uint32_t value = 0;
    std::size_t digits  = 0;
    while (position < pattern.size() && digits < 3 && pattern[position] >= '0' &&
           pattern[position] <= '7') {
      value = (value << 3U) | static_cast<std::uint32_t>(pattern[position] - '0');
      ++position;
      ++digits;
    }
    result += value <= 0xff ? std::format("\\x{:02X}", value) : std::format("\\u{:04X}", value);
  }
  return result;
}

regex_ir::compile_options make_compile_options(strings::regex_flags flags)
{
  regex_ir::compile_options options;
  options.case_insensitive = strings::is_ignorecase(flags);
  options.multiline        = strings::is_multiline(flags);
  options.dot_all          = strings::is_dotall(flags);
  options.ascii_classes    = strings::is_ascii(flags);
  options.extended_newline = strings::is_ext_newline(flags);
  return options;
}

regex_ir::operation_kind internal_operation(regex_operation operation)
{
  switch (operation) {
    case regex_operation::CONTAINS:
    case regex_operation::MATCHES: return regex_ir::operation_kind::CONTAINS;
    case regex_operation::COUNT: return regex_ir::operation_kind::COUNT;
    case regex_operation::FIND: return regex_ir::operation_kind::FIND;
    case regex_operation::EXTRACT:
    case regex_operation::EXTRACT_SINGLE:
    case regex_operation::EXTRACT_ALL_RECORD:
    case regex_operation::FINDALL:
    case regex_operation::REPLACE:
    case regex_operation::REPLACE_WITH_BACKREFS: return regex_ir::operation_kind::EXTRACT;
    case regex_operation::SPLIT:
    case regex_operation::RSPLIT:
    case regex_operation::SPLIT_RECORD:
    case regex_operation::RSPLIT_RECORD: return regex_ir::operation_kind::SPLIT;
  }
  CUDF_FAIL("Invalid JIT regex operation");
}

void expect_operation(regex_jit_program const& prog, regex_operation expected, std::string_view api)
{
  CUDF_EXPECTS(prog.operation() == expected,
               std::format("{} requires a regex_jit_program created for that operation", api));
}

void expect_split_parameters(regex_jit_program const& prog,
                             size_type maxsplit,
                             std::string_view api)
{
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  CUDF_EXPECTS(regex_jit_program_accessor::options(prog).maxsplit == maxsplit,
               std::format("{} maxsplit does not match the regex_jit_program maxsplit", api));
}

void ensure_stack_size()
{
  std::size_t stack_size = 0;
  CUDF_CUDA_TRY(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
  if (stack_size < required_stack_size) {
    CUDF_CUDA_TRY(cudaDeviceSetLimit(cudaLimitStackSize, required_stack_size));
  }
}

retained_kernel compile_kernel(std::string const& matcher,
                               regex_ir::operation_kind operation,
                               std::string wrapper,
                               std::string_view name)
{
  auto threads                      = block_size(operation, matcher);
  auto module                       = regex_ir::nvvm::assemble(matcher, std::move(wrapper));
  auto fragment                     = get_nvvm_fragment(std::string{name}, module);
  rtcx::memory_fragment fragments[] = {
    {.data = fragment->view(), .type = rtcx::binary_type::LTO_IR, .name = nullptr}};
  return {get_lto_linked_kernel(std::string{name}, {}, fragments), threads};
}

void launch(retained_kernel const& prepared,
            input_data const& input,
            cuda::stream_ref stream,
            void* output,
            void const* extra = nullptr)
{
  if (input.rows == 0) { return; }
  auto chars      = const_cast<char*>(input.chars);
  auto offsets    = const_cast<void*>(input.offsets);
  auto validity   = const_cast<bitmask_type*>(input.validity);
  auto row_offset = input.row_offset;
  auto rows       = input.rows;
  auto threads    = prepared.threads;
  auto grid =
    static_cast<std::uint32_t>((static_cast<std::uint32_t>(rows) + threads - 1) / threads);
  if (extra == nullptr) {
    prepared.value.launch_with(
      {grid, 1, 1}, {threads, 1, 1}, 0, stream, chars, offsets, validity, row_offset, rows, output);
  } else {
    prepared.value.launch_with({grid, 1, 1},
                               {threads, 1, 1},
                               0,
                               stream,
                               chars,
                               offsets,
                               validity,
                               row_offset,
                               rows,
                               output,
                               extra);
  }
}

std::unique_ptr<column> fixed_result(strings_column_view const& input,
                                     regex_jit_program const& prog,
                                     data_type output_type,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  auto result =
    make_numeric_column(output_type,
                        input.size(),
                        cudf::detail::copy_bitmask(input.parent(), stream, mr.get_output_mr()),
                        input.null_count(),
                        stream,
                        mr.get_output_mr());
  if (input.is_empty()) { return result; }
  auto data      = get_input_data(input, stream);
  auto& prepared = regex_jit_program_accessor::kernel(prog, data.offset64);
  launch(prepared, data, stream, result->mutable_view().head<void>());
  return result;
}

std::unique_ptr<column> make_strings(rmm::device_uvector<pair_t> const& pairs,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  return cudf::make_strings_column(
    device_span<pair_t const>{pairs.data(), pairs.size()}, stream, mr.get_output_mr());
}

std::unique_ptr<table> extract_impl(strings_column_view const& input,
                                    regex_jit_program const& prog,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  auto groups = prog.groups_count();
  CUDF_EXPECTS(groups > 0, "Group indicators not found in regex pattern");

  if (input.is_empty()) {
    std::vector<std::unique_ptr<column>> columns;
    columns.reserve(groups);
    for (size_type group = 0; group < groups; ++group) {
      columns.emplace_back(make_empty_column(type_id::STRING));
    }
    return std::make_unique<table>(std::move(columns));
  }

  auto pair_count = static_cast<std::size_t>(input.size()) * static_cast<std::size_t>(groups);
  rmm::device_uvector<pair_t> pairs(pair_count, stream, mr.get_temporary_mr());
  auto data      = get_input_data(input, stream);
  auto& prepared = regex_jit_program_accessor::kernel(prog, data.offset64);
  launch(prepared, data, stream, pairs.data());

  std::vector<device_span<pair_t const>> spans;
  spans.reserve(groups);
  for (size_type group = 0; group < groups; ++group) {
    spans.emplace_back(pairs.data() + static_cast<std::size_t>(group) * input.size(), input.size());
  }
  auto columns = cudf::make_strings_column_batch(spans, stream, mr.get_output_mr());
  return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<column> extract_single_impl(strings_column_view const& input,
                                            regex_jit_program const& prog,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  auto groups = prog.groups_count();
  CUDF_EXPECTS(groups > 0, "capture groups not found in regex pattern", std::invalid_argument);

  rmm::device_uvector<pair_t> pairs(input.size(), stream, mr.get_temporary_mr());
  auto data      = get_input_data(input, stream);
  auto& prepared = regex_jit_program_accessor::kernel(prog, data.offset64);
  launch(prepared, data, stream, pairs.data());
  return make_strings(pairs, stream, mr);
}

std::unique_ptr<column> enumerate_impl(strings_column_view const& input,
                                       regex_jit_program const& prog,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  auto capture_count = prog.groups_count();
  auto findall       = prog.operation() == regex_operation::FINDALL;
  auto groups        = prog.capture() == strings::capture_groups::EXTRACT ? capture_count : 0;
  if (findall) {
    CUDF_EXPECTS(groups <= 1, "findall does not support more than 1 capture group");
  } else {
    CUDF_EXPECTS(groups > 0, "extract_all requires group indicators in the regex pattern.");
  }
  if (input.is_empty()) { return make_empty_lists_column(data_type{type_id::STRING}); }

  auto counts       = make_numeric_column(data_type{type_id::INT32},
                                    input.size(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr.get_temporary_mr());
  auto validity     = make_numeric_column(data_type{type_id::BOOL8},
                                      input.size(),
                                      mask_state::UNALLOCATED,
                                      stream,
                                      mr.get_temporary_mr());
  auto data         = get_input_data(input, stream);
  auto& size_kernel = regex_jit_program_accessor::kernel(prog, data.offset64);
  launch(size_kernel,
         data,
         stream,
         counts->mutable_view().head<void>(),
         validity->mutable_view().head<void>());

  auto [offsets, total] = cudf::strings::detail::make_offsets_child_column(
    counts->view().begin<size_type>(), counts->view().end<size_type>(), stream, mr);
  rmm::device_uvector<pair_t> pairs(total, stream, mr.get_temporary_mr());
  if (total > 0) {
    auto& emit_kernel = regex_jit_program_accessor::kernel(prog, data.offset64, true);
    launch(emit_kernel, data, stream, pairs.data(), offsets->view().data<size_type>());
  }
  auto strings_output = make_strings(pairs, stream, mr);

  rmm::device_buffer null_mask;
  size_type null_count;
  if (findall) {
    null_mask  = cudf::detail::copy_bitmask(input.parent(), stream, mr.get_output_mr());
    null_count = input.null_count();
  } else {
    auto converted = cudf::bools_to_mask(validity->view(), stream, mr.get_output_mr());
    null_mask      = std::move(*converted.first);
    null_count     = converted.second;
  }
  return make_lists_column(
    input.size(), std::move(offsets), std::move(strings_output), null_count, std::move(null_mask));
}

using replacement_piece = regex_ir::nvvm::replacement_piece;

std::vector<replacement_piece> literal_replacement(std::string_view replacement)
{
  return {{std::string{replacement}, std::nullopt}};
}

std::vector<replacement_piece> parse_backref_replacement(std::string_view replacement,
                                                         size_type group_count)
{
  CUDF_EXPECTS(!replacement.empty(), "Parameter replacement must not be empty");
  auto uses_backslash = false;
  for (std::size_t index = 0; index + 1 < replacement.size(); ++index) {
    if (replacement[index] == '\\' &&
        std::isdigit(static_cast<unsigned char>(replacement[index + 1])) != 0) {
      uses_backslash = true;
      break;
    }
  }

  std::vector<replacement_piece> result;
  std::string literal;
  auto flush_literal = [&] {
    if (!literal.empty()) {
      result.push_back({std::move(literal), std::nullopt});
      literal.clear();
    }
  };
  for (std::size_t position = 0; position < replacement.size();) {
    auto capture_begin = std::string_view::npos;
    auto capture_end   = std::string_view::npos;
    if (uses_backslash && replacement[position] == '\\' && position + 1 < replacement.size() &&
        std::isdigit(static_cast<unsigned char>(replacement[position + 1])) != 0) {
      capture_begin = position + 1;
      capture_end   = capture_begin;
      while (capture_end < replacement.size() &&
             std::isdigit(static_cast<unsigned char>(replacement[capture_end])) != 0) {
        ++capture_end;
      }
    } else if (!uses_backslash && replacement[position] == '$' &&
               position + 3 < replacement.size() && replacement[position + 1] == '{' &&
               std::isdigit(static_cast<unsigned char>(replacement[position + 2])) != 0) {
      capture_begin = position + 2;
      capture_end   = capture_begin;
      while (capture_end < replacement.size() &&
             std::isdigit(static_cast<unsigned char>(replacement[capture_end])) != 0) {
        ++capture_end;
      }
      if (capture_end == replacement.size() || replacement[capture_end] != '}') {
        capture_begin = std::string_view::npos;
      }
    }

    if (capture_begin == std::string_view::npos) {
      literal.push_back(replacement[position++]);
      continue;
    }

    std::uint64_t capture = 0;
    for (auto index = capture_begin; index < capture_end; ++index) {
      capture = capture * 10U + static_cast<std::uint64_t>(replacement[index] - '0');
    }
    CUDF_EXPECTS(capture <= static_cast<std::uint64_t>(std::min(group_count, size_type{99})),
                 "Group index numbers must be in the range 0 to group count");
    flush_literal();
    result.push_back({{}, static_cast<size_type>(capture)});
    position = uses_backslash ? capture_end : capture_end + 1;
  }
  flush_literal();
  return result;
}

std::unique_ptr<column> replace_impl(strings_column_view const& input,
                                     regex_jit_program const& prog,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  if (prog.pattern().empty()) {
    return std::make_unique<column>(input.parent(), stream, mr.get_output_mr());
  }
  auto sizes        = make_numeric_column(data_type{type_id::INT32},
                                   input.size(),
                                   mask_state::UNALLOCATED,
                                   stream,
                                   mr.get_temporary_mr());
  auto data         = get_input_data(input, stream);
  auto& size_kernel = regex_jit_program_accessor::kernel(prog, data.offset64);
  launch(size_kernel, data, stream, sizes->mutable_view().head<void>());
  auto [offsets, total] = cudf::strings::detail::make_offsets_child_column(
    sizes->view().begin<size_type>(), sizes->view().end<size_type>(), stream, mr);
  rmm::device_buffer chars(total, stream, mr.get_output_mr());
  if (total > 0) {
    auto& emit_kernel = regex_jit_program_accessor::kernel(
      prog, data.offset64, true, offsets->type().id() == type_id::INT64);
    launch(emit_kernel, data, stream, chars.data(), offsets->view().head<void>());
  }
  return make_strings_column(
    input.size(),
    std::move(offsets),
    std::move(chars),
    input.null_count(),
    cudf::detail::copy_bitmask(input.parent(), stream, mr.get_output_mr()));
}

struct split_result {
  std::unique_ptr<column> lists;
  size_type columns;
};

split_result split_record_impl(strings_column_view const& input,
                               regex_jit_program const& prog,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  if (input.is_empty()) { return {make_empty_lists_column(data_type{type_id::STRING}), 1}; }

  auto effective_counts = make_numeric_column(data_type{type_id::INT32},
                                              input.size(),
                                              mask_state::UNALLOCATED,
                                              stream,
                                              mr.get_temporary_mr());
  auto full_counts      = make_numeric_column(data_type{type_id::INT32},
                                         input.size(),
                                         mask_state::UNALLOCATED,
                                         stream,
                                         mr.get_temporary_mr());
  auto data             = get_input_data(input, stream);
  auto& size_kernel     = regex_jit_program_accessor::kernel(prog, data.offset64);
  launch(size_kernel,
         data,
         stream,
         effective_counts->mutable_view().head<void>(),
         full_counts->mutable_view().head<void>());

  auto maximum = cudf::reduce(effective_counts->view(),
                              *make_max_aggregation<reduce_aggregation>(),
                              data_type{type_id::INT32},
                              stream,
                              mr.get_temporary_mr());
  auto columns = static_cast<numeric_scalar<size_type> const&>(*maximum).value(stream);
  columns      = std::max(columns, size_type{1});

  auto [effective_offsets, effective_total] =
    cudf::strings::detail::make_offsets_child_column(effective_counts->view().begin<size_type>(),
                                                     effective_counts->view().end<size_type>(),
                                                     stream,
                                                     mr);
  auto [full_offsets, full_total] = cudf::strings::detail::make_offsets_child_column(
    full_counts->view().begin<size_type>(),
    full_counts->view().end<size_type>(),
    stream,
    cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()});
  rmm::device_uvector<pair_t> pairs(effective_total, stream, mr.get_temporary_mr());
  rmm::device_uvector<std::int64_t> spans(
    static_cast<std::size_t>(full_total) * 2, stream, mr.get_temporary_mr());
  if (effective_total > 0) {
    auto& emit_kernel = regex_jit_program_accessor::kernel(prog, data.offset64, true);
    auto chars        = const_cast<char*>(data.chars);
    auto offsets      = const_cast<void*>(data.offsets);
    auto validity     = const_cast<bitmask_type*>(data.validity);
    auto row_offset   = data.row_offset;
    auto rows         = data.rows;
    auto threads      = emit_kernel.threads;
    auto grid =
      static_cast<std::uint32_t>((static_cast<std::uint32_t>(rows) + threads - 1) / threads);
    auto effective_data = effective_offsets->view().data<size_type>();
    auto full_data      = full_offsets->view().data<size_type>();
    emit_kernel.value.launch_with({grid, 1, 1},
                                  {threads, 1, 1},
                                  0,
                                  stream,
                                  chars,
                                  offsets,
                                  validity,
                                  row_offset,
                                  rows,
                                  pairs.data(),
                                  effective_data,
                                  full_data,
                                  spans.data());
  }
  auto strings_output = make_strings(pairs, stream, mr);
  auto lists =
    make_lists_column(input.size(),
                      std::move(effective_offsets),
                      std::move(strings_output),
                      input.null_count(),
                      cudf::detail::copy_bitmask(input.parent(), stream, mr.get_output_mr()));
  return {std::move(lists), columns};
}

std::unique_ptr<table> split_table_impl(strings_column_view const& input,
                                        regex_jit_program const& prog,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  if (input.is_empty()) {
    std::vector<std::unique_ptr<column>> columns;
    columns.emplace_back(make_empty_column(type_id::STRING));
    return std::make_unique<table>(std::move(columns));
  }
  auto result = split_record_impl(
    input, prog, stream, cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()});
  lists_column_view lists{result.lists->view()};
  std::vector<std::unique_ptr<column>> columns;
  columns.reserve(result.columns);
  for (size_type index = 0; index < result.columns; ++index) {
    auto column = cudf::lists::extract_list_element(lists, index, stream, mr.get_output_mr());
    if (column->null_count() == 0) { column->set_null_mask(rmm::device_buffer{}, 0); }
    columns.emplace_back(std::move(column));
  }
  return std::make_unique<table>(std::move(columns));
}

}  // namespace

retained_kernel const& regex_jit_program_accessor::kernel(regex_jit_program const& prog,
                                                          bool offset64,
                                                          bool emit,
                                                          bool output_offset64)
{
  auto index =
    emit && (prog.operation() == regex_operation::REPLACE ||
             prog.operation() == regex_operation::REPLACE_WITH_BACKREFS)
      ? 2U + static_cast<std::size_t>(offset64) * 2U + static_cast<std::size_t>(output_offset64)
      : static_cast<std::size_t>(offset64) + (emit ? 2U : 0U);
  return prog._impl->kernels.at(index);
}

std::unique_ptr<regex_jit_program> regex_jit_program::create(
  std::string_view pattern,
  regex_operation operation,
  regex_jit_program_options const& program_options,
  strings::regex_flags flags,
  strings::capture_groups captures)
{
  return std::unique_ptr<regex_jit_program>(
    new regex_jit_program(pattern, operation, program_options, flags, captures));
}

regex_jit_program::regex_jit_program(std::string_view pattern,
                                     regex_operation operation,
                                     regex_jit_program_options const& program_options,
                                     strings::regex_flags flags,
                                     strings::capture_groups captures)
  : _pattern{pattern},
    _operation{operation},
    _flags{flags},
    _captures{captures},
    _options{program_options}
{
  try {
    auto compile_pattern = normalize_pattern(pattern);
    if (operation == regex_operation::MATCHES) {
      compile_pattern = std::format(R"(\A(?:{}))", compile_pattern);
    }

    auto compile_options = make_compile_options(flags);
    auto compiled        = regex_ir::compile(
      compile_pattern, internal_operation(operation), std::nullopt, compile_options);
    auto capture_count    = static_cast<size_type>(compiled.capture_count);
    auto matcher          = std::move(compiled.nvvm_ir);
    auto kernels          = std::vector<retained_kernel>{};
    auto validation_error = std::optional<std::string>{};
    auto internal         = internal_operation(operation);
    auto kernel_operation =
      operation == regex_operation::REPLACE || operation == regex_operation::REPLACE_WITH_BACKREFS
        ? regex_ir::operation_kind::REPLACE
        : internal;

    auto add_pass = [&](auto&& make_wrapper, std::string_view name) {
      for (auto offset64 : {false, true}) {
        kernels.push_back(compile_kernel(matcher, kernel_operation, make_wrapper(offset64), name));
      }
    };
    auto add_output_offset_pass = [&](auto&& make_wrapper, std::string_view name) {
      for (auto offset64 : {false, true}) {
        for (auto output_offset64 : {false, true}) {
          kernels.push_back(compile_kernel(
            matcher, kernel_operation, make_wrapper(offset64, output_offset64), name));
        }
      }
    };
    ensure_stack_size();
    switch (operation) {
      case regex_operation::CONTAINS:
      case regex_operation::MATCHES:
      case regex_operation::COUNT:
      case regex_operation::FIND:
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_fixed_kernel(offset64, internal, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.fixed");
        break;
      case regex_operation::EXTRACT:
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_capture_kernel(
              offset64, (capture_count + 1) * 2, 0, capture_count, true, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.extract");
        break;
      case regex_operation::EXTRACT_SINGLE: {
        CUDF_EXPECTS(program_options.group.has_value(),
                     "extract_single requires a group in regex_jit_program_options");
        auto group = *program_options.group;
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_capture_kernel(
              offset64, (capture_count + 1) * 2, group, 1, false, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.extract_single");
        break;
      }
      case regex_operation::EXTRACT_ALL_RECORD:
      case regex_operation::FINDALL: {
        auto findall = operation == regex_operation::FINDALL;
        auto groups  = captures == strings::capture_groups::EXTRACT ? capture_count : 0;
        auto slots   = (capture_count + 1) * 2;
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_enumeration_size_kernel(
              offset64, slots, findall ? 1 : groups, !findall, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.enumerate_size");
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_enumeration_emit_kernel(
              offset64, slots, groups, findall, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.enumerate_emit");
        break;
      }
      case regex_operation::REPLACE:
      case regex_operation::REPLACE_WITH_BACKREFS: {
        auto replacement_text = program_options.replacement.value_or("");
        if (operation == regex_operation::REPLACE_WITH_BACKREFS) {
          CUDF_EXPECTS(program_options.replacement.has_value(),
                       "replace_with_backrefs requires a replacement in regex_jit_program_options");
        }
        auto replacement = literal_replacement(replacement_text);
        if (operation == regex_operation::REPLACE_WITH_BACKREFS) {
          try {
            replacement = parse_backref_replacement(replacement_text, capture_count);
          } catch (cudf::logic_error const& error) {
            validation_error = error.what();
            replacement      = literal_replacement("");
          }
        }
        auto limit =
          program_options.max_replace_count.has_value() && *program_options.max_replace_count > 0
            ? *program_options.max_replace_count
          : program_options.max_replace_count == 0 ? 0
                                                   : std::numeric_limits<size_type>::max();
        auto slots = (capture_count + 1) * 2;
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_limited_replace_kernel(
              offset64, false, false, replacement, slots, limit, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.replace_size");
        add_output_offset_pass(
          [&](bool offset64, bool output_offset64) {
            return regex_ir::nvvm::make_limited_replace_kernel(
              offset64, true, output_offset64, replacement, slots, limit, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.replace_emit");
        break;
      }
      case regex_operation::SPLIT:
      case regex_operation::RSPLIT:
      case regex_operation::SPLIT_RECORD:
      case regex_operation::RSPLIT_RECORD: {
        auto reverse =
          operation == regex_operation::RSPLIT || operation == regex_operation::RSPLIT_RECORD;
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_split_size_kernel(
              offset64, program_options.maxsplit, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.split_size");
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_split_emit_kernel(offset64, reverse, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.split_emit");
        break;
      }
    }
    _impl = std::make_unique<regex_jit_program_impl>(
      regex_jit_program_impl{capture_count, std::move(kernels), std::move(validation_error)});
  } catch (std::invalid_argument const& error) {
    CUDF_FAIL(error.what());
  }
}

regex_jit_program::~regex_jit_program()                                       = default;
regex_jit_program::regex_jit_program(regex_jit_program&& other) noexcept      = default;
regex_jit_program& regex_jit_program::operator=(regex_jit_program&&) noexcept = default;

std::string regex_jit_program::pattern() const { return _pattern; }

strings::regex_flags regex_jit_program::flags() const { return _flags; }

strings::capture_groups regex_jit_program::capture() const { return _captures; }

regex_operation regex_jit_program::operation() const { return _operation; }

size_type regex_jit_program::groups_count() const { return _impl->capture_count; }

std::unique_ptr<column> contains_re(strings_column_view const& input,
                                    regex_jit_program const& prog,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::CONTAINS, "contains_re");
  return fixed_result(input, prog, data_type{type_id::BOOL8}, stream, mr);
}

std::unique_ptr<column> matches_re(strings_column_view const& input,
                                   regex_jit_program const& prog,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::MATCHES, "matches_re");
  return fixed_result(input, prog, data_type{type_id::BOOL8}, stream, mr);
}

std::unique_ptr<column> count_re(strings_column_view const& input,
                                 regex_jit_program const& prog,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::COUNT, "count_re");
  return fixed_result(input, prog, data_type{type_id::INT32}, stream, mr);
}

std::unique_ptr<column> find_re(strings_column_view const& input,
                                regex_jit_program const& prog,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::FIND, "find_re");
  return fixed_result(input, prog, data_type{type_id::INT32}, stream, mr);
}

std::unique_ptr<table> extract(strings_column_view const& input,
                               regex_jit_program const& prog,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::EXTRACT, "extract");
  return extract_impl(input, prog, stream, mr);
}

std::unique_ptr<column> extract_single(strings_column_view const& input,
                                       regex_jit_program const& prog,
                                       size_type group,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::EXTRACT_SINGLE, "extract_single");
  auto& options = regex_jit_program_accessor::options(prog);
  CUDF_EXPECTS(options.group == group,
               "extract_single group does not match the regex_jit_program group");
  CUDF_EXPECTS(group >= 0 && group < prog.groups_count(),
               "group parameter outside the range of capture groups found in the regex pattern",
               std::invalid_argument);
  return extract_single_impl(input, prog, stream, mr);
}

std::unique_ptr<column> extract_all_record(strings_column_view const& input,
                                           regex_jit_program const& prog,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::EXTRACT_ALL_RECORD, "extract_all_record");
  return enumerate_impl(input, prog, stream, mr);
}

std::unique_ptr<column> findall(strings_column_view const& input,
                                regex_jit_program const& prog,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::FINDALL, "findall");
  return enumerate_impl(input, prog, stream, mr);
}

std::unique_ptr<column> replace_re(strings_column_view const& input,
                                   regex_jit_program const& prog,
                                   string_scalar const& replacement,
                                   std::optional<size_type> max_replace_count,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::REPLACE, "replace_re");
  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  auto& options = regex_jit_program_accessor::options(prog);
  CUDF_EXPECTS(options.replacement.value_or("") == replacement.to_string(stream),
               "replace_re replacement does not match the regex_jit_program replacement");
  CUDF_EXPECTS(options.max_replace_count == max_replace_count,
               "replace_re max_replace_count does not match the regex_jit_program limit");
  return replace_impl(input, prog, stream, mr);
}

std::unique_ptr<column> replace_with_backrefs(strings_column_view const& input,
                                              regex_jit_program const& prog,
                                              std::string_view replacement,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::REPLACE_WITH_BACKREFS, "replace_with_backrefs");
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  auto& options = regex_jit_program_accessor::options(prog);
  CUDF_EXPECTS(
    options.replacement == replacement,
    "replace_with_backrefs replacement does not match the regex_jit_program replacement");
  auto& validation_error = regex_jit_program_accessor::validation_error(prog);
  CUDF_EXPECTS(!validation_error.has_value(), validation_error.value_or(""));
  return replace_impl(input, prog, stream, mr);
}

std::unique_ptr<table> split_re(strings_column_view const& input,
                                regex_jit_program const& prog,
                                size_type maxsplit,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::SPLIT, "split_re");
  expect_split_parameters(prog, maxsplit, "split_re");
  return split_table_impl(input, prog, stream, mr);
}

std::unique_ptr<table> rsplit_re(strings_column_view const& input,
                                 regex_jit_program const& prog,
                                 size_type maxsplit,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::RSPLIT, "rsplit_re");
  expect_split_parameters(prog, maxsplit, "rsplit_re");
  return split_table_impl(input, prog, stream, mr);
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        regex_jit_program const& prog,
                                        size_type maxsplit,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::SPLIT_RECORD, "split_record_re");
  expect_split_parameters(prog, maxsplit, "split_record_re");
  return split_record_impl(input, prog, stream, mr).lists;
}

std::unique_ptr<column> rsplit_record_re(strings_column_view const& input,
                                         regex_jit_program const& prog,
                                         size_type maxsplit,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::RSPLIT_RECORD, "rsplit_record_re");
  expect_split_parameters(prog, maxsplit, "rsplit_record_re");
  return split_record_impl(input, prog, stream, mr).lists;
}

}  // namespace cudf::experimental
