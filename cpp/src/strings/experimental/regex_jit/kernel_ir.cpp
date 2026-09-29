/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_ir.hpp"

#include <nvvm_templates.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <format>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudf::experimental::detail::regex_jit {
namespace {

enum class matcher_abi { NONE, BOOLEAN, COUNT, FIND, CAPTURES, REPLACE, SPLIT };

void replace_all(std::string& text, std::string_view from, std::string_view to)
{
  for (auto position = text.find(from); position != std::string::npos;
       position      = text.find(from, position + to.size())) {
    text.replace(position, from.size(), to);
  }
}

std::string common_nvvm(bool offset64, bool pairs, matcher_abi abi)
{
  auto result =
    std::string{regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "core")};
  replace_all(result, "@INPUT_OFFSET64@", offset64 ? "true" : "false");
  if (pairs) {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "write_pair");
  }
  switch (abi) {
    case matcher_abi::NONE: break;
    case matcher_abi::BOOLEAN:
      result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions,
                                                "matcher_boolean_declaration");
      break;
    case matcher_abi::COUNT:
      result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions,
                                                "matcher_count_declaration");
      break;
    case matcher_abi::FIND:
      result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions,
                                                "matcher_find_declaration");
      break;
    case matcher_abi::CAPTURES:
      result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions,
                                                "matcher_captures_declaration");
      break;
    case matcher_abi::REPLACE:
      result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions,
                                                "matcher_replace_declaration");
      break;
    case matcher_abi::SPLIT:
      result +=
        regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "split_execute");
      break;
  }
  return result;
}

void require_kernel_name(std::string_view kernel_name)
{
  auto first_is_valid = [](unsigned char character) {
    return std::isalpha(character) != 0 || character == '_';
  };
  auto rest_is_valid = [&](unsigned char character) {
    return first_is_valid(character) || std::isdigit(character) != 0;
  };
  if (kernel_name.empty() || !first_is_valid(static_cast<unsigned char>(kernel_name.front())) ||
      !std::all_of(kernel_name.begin() + 1, kernel_name.end(), [&](char character) {
        return rest_is_valid(static_cast<unsigned char>(character));
      })) {
    throw std::invalid_argument("kernel_name must be a valid source identifier");
  }
  if (kernel_name.starts_with("llvm.") || kernel_name.starts_with("nvvm.")) {
    throw std::invalid_argument("kernel_name uses a reserved identifier");
  }
}

std::string annotate_kernel(std::string module,
                            std::string_view signature,
                            std::string_view kernel_name)
{
  require_kernel_name(kernel_name);
  replace_all(module, "@KERNEL_ENTRY@", std::format("@{}", kernel_name));
  module += regex_ir::format_nvvm_template_section(
    regex_ir_nvvm_templates::kernel_functions, "kernel_annotation", signature, kernel_name);
  return module;
}

}  // namespace

std::string make_module(std::string wrapper)
{
  return regex_ir::render_nvvm_template(regex_ir_nvvm_templates::kernel_module,
                                        {{"@WRAPPER@", wrapper}});
}

std::string make_warp_literal_contains_kernel(bool offset64,
                                              std::string_view literal,
                                              std::string_view kernel_name)
{
  if (literal.empty()) { throw std::invalid_argument("literal must not be empty"); }
  auto comparisons = std::string{};
  auto matched     = std::string{};
  auto offset      = std::size_t{1};
  auto index       = std::size_t{0};
  while (offset < literal.size()) {
    auto remaining      = literal.size() - offset;
    auto width          = remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
    std::uint64_t value = 0;
    for (std::size_t byte = 0; byte < width; ++byte) {
      value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(literal[offset + byte]))
               << (byte * 8U);
    }
    auto bits = width * 8U;
    std::format_to(std::back_inserter(comparisons),
                   "  %literal_ptr_{0} = getelementptr i8, i8* %data, i64 %literal_offset_{0}\n",
                   index);
    if (width == 1U) {
      std::format_to(std::back_inserter(comparisons),
                     "  %literal_chunk_{0} = load i8, i8* %literal_ptr_{0}, align 1\n",
                     index);
    } else {
      std::format_to(std::back_inserter(comparisons),
                     R"NVVM(  %literal_chunk_ptr_{0} = bitcast i8* %literal_ptr_{0} to i{1}*
  %literal_chunk_{0} = load i{1}, i{1}* %literal_chunk_ptr_{0}, align 1
)NVVM",
                     index,
                     bits);
    }
    std::format_to(std::back_inserter(comparisons),
                   "  %literal_equal_{0} = icmp eq i{1} %literal_chunk_{0}, {2}\n",
                   index,
                   bits,
                   value);
    if (matched.empty()) {
      matched = std::format("%literal_equal_{}", index);
    } else {
      std::format_to(std::back_inserter(comparisons),
                     "  %literal_through_{0} = and i1 {1}, %literal_equal_{0}\n",
                     index,
                     matched);
      matched = std::format("%literal_through_{}", index);
    }
    offset += width;
    ++index;
  }

  auto offsets = std::string{};
  offset       = 1;
  for (std::size_t chunk = 0; chunk < index; ++chunk) {
    std::format_to(std::back_inserter(offsets),
                   "  %literal_offset_{0} = add i64 %position, {1}\n",
                   chunk,
                   offset);
    auto remaining = literal.size() - offset;
    offset += remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
  }

  auto verify =
    literal.size() == 1U
      ? std::string{"  br i1 %first_equal, label %local_yes, label %inner_continue\n"}
      : regex_ir::render_nvvm_template_section(
          regex_ir_nvvm_templates::kernel_warp_literal,
          "warp_literal_verify",
          {{"@OFFSETS@", offsets}, {"@COMPARISONS@", comparisons}, {"@MATCHED@", matched}});

  auto result = common_nvvm(offset64, false, matcher_abi::NONE);
  result += regex_ir::render_nvvm_template_section(
    regex_ir_nvvm_templates::kernel_warp_literal,
    "warp_literal_contains",
    {{"@LITERAL_SIZE@", std::to_string(literal.size())},
     {"@FIRST_BYTE@",
      std::to_string(static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())))},
     {"@VERIFY@", verify}});
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i8*", kernel_name);
}

std::string make_fixed_kernel(bool offset64,
                              regex_ir::operation_kind operation,
                              std::string_view kernel_name)
{
  auto abi    = operation == regex_ir::operation_kind::CONTAINS ||
                 operation == regex_ir::operation_kind::MATCHES
                  ? matcher_abi::BOOLEAN
                : operation == regex_ir::operation_kind::COUNT ? matcher_abi::COUNT
                                                               : matcher_abi::FIND;
  auto result = common_nvvm(offset64, false, abi);
  if (operation == regex_ir::operation_kind::CONTAINS ||
      operation == regex_ir::operation_kind::MATCHES) {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_fixed, "fixed_boolean");
  } else if (operation == regex_ir::operation_kind::COUNT) {
    result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_fixed, "fixed_count");
  } else {
    result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_fixed, "fixed_find");
  }
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i8*", kernel_name);
}

std::string make_capture_kernel(bool offset64,
                                std::int32_t capture_slots,
                                std::int32_t first_group,
                                std::int32_t output_groups,
                                bool column_major,
                                std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, true, matcher_abi::CAPTURES);
  result += regex_ir::render_nvvm_template_section(
    regex_ir_nvvm_templates::kernel_enumeration,
    "capture",
    {{"@CAPTURE_SLOTS@", std::to_string(capture_slots)},
     {"@FIRST_GROUP@", std::to_string(first_group)},
     {"@OUTPUT_GROUPS@", std::to_string(output_groups)},
     {"@COLUMN_SETUP@", column_major ? "%rows64 = sext i32 %rows to i64" : ""},
     {"@PAIR_INDEX@",
      column_major ? R"NVVM(%group_base = mul i64 %group64, %rows64
  %row64 = sext i32 %row to i64
  %pair_index = add i64 %group_base, %row64)NVVM"
                   : "%pair_index = sext i32 %row to i64"}});
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i8*", kernel_name);
}

std::string make_enumeration_size_kernel(bool offset64,
                                         std::int32_t capture_slots,
                                         std::int32_t multiplier,
                                         bool require_match,
                                         bool cache,
                                         std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false, matcher_abi::CAPTURES);
  if (cache) {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "capture_cache");
  }
  auto cache_parameters = cache ? ", i8* %cache_buffer, i32 %capacity, i8* %overflow" : "";
  auto cache_match      = cache ? std::format(
                               R"NVVM(%typed_cache = bitcast i8* %cache_buffer to i64*
  %overflow_ptr = getelementptr i8, i8* %overflow, i32 %row
  call void @libregex_ir_cache_captures(i64* %capture_ptr, i64* %typed_cache, i32 %row, i32 %capacity, i32 {}, i64 %count, i8* %overflow_ptr))NVVM",
                               capture_slots)
                                : "";
  result += regex_ir::render_nvvm_template_section(
    regex_ir_nvvm_templates::kernel_enumeration,
    "enumeration_size",
    {{"@CAPTURE_SLOTS@", std::to_string(capture_slots)},
     {"@MULTIPLIER@", std::to_string(multiplier)},
     {"@ROW_VALID@", require_match ? "and i1 %valid, %has_match" : "and i1 %valid, true"},
     {"@CACHE_PARAMETERS@", cache_parameters},
     {"@CACHE_MATCH@", cache_match}});
  return annotate_kernel(std::move(result),
                         cache ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*, i32, i8*"
                               : "i8*, i8*, i32*, i32, i32, i8*, i8*",
                         kernel_name);
}

std::string make_enumeration_emit_kernel(bool offset64,
                                         std::int32_t capture_slots,
                                         std::int32_t groups,
                                         bool findall,
                                         bool overflow_only,
                                         std::string_view kernel_name)
{
  auto result      = common_nvvm(offset64, true, matcher_abi::CAPTURES);
  auto write_match = [&] {
    if (findall && groups == 0) {
      return std::string{regex_ir::nvvm_template_section(
        regex_ir_nvvm_templates::kernel_enumeration, "write_findall_whole")};
    }
    if (findall) {
      return regex_ir::render_nvvm_template_section(
        regex_ir_nvvm_templates::kernel_enumeration,
        "write_findall_capture",
        {{"@CAPTURE_SLOTS@", std::to_string(capture_slots)}});
    }
    return regex_ir::render_nvvm_template_section(
      regex_ir_nvvm_templates::kernel_enumeration,
      "write_extract",
      {{"@CAPTURE_SLOTS@", std::to_string(capture_slots)}, {"@GROUPS@", std::to_string(groups)}});
  }();
  auto overflow_parameter = overflow_only ? ", i8* %overflow" : "";
  auto overflow_guard     = overflow_only
                              ? R"NVVM(%overflow_ptr = getelementptr i8, i8* %overflow, i32 %row
  %overflow_value = load i8, i8* %overflow_ptr, align 1
  %did_overflow = icmp ne i8 %overflow_value, 0
  %selected = and i1 %valid, %did_overflow
  br i1 %selected, label %setup, label %done)NVVM"
                              : "br i1 %valid, label %setup, label %done";
  result +=
    regex_ir::render_nvvm_template_section(regex_ir_nvvm_templates::kernel_enumeration,
                                           "enumeration_emit",
                                           {{"@CAPTURE_SLOTS@", std::to_string(capture_slots)},
                                            {"@OVERFLOW_PARAMETER@", overflow_parameter},
                                            {"@OVERFLOW_GUARD@", overflow_guard},
                                            {"@WRITE_MATCH@", write_match}});
  return annotate_kernel(std::move(result),
                         overflow_only ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*"
                                       : "i8*, i8*, i32*, i32, i32, i8*, i8*",
                         kernel_name);
}

namespace {

std::string llvm_bytes(std::string_view value)
{
  std::string result;
  result.reserve(value.size() * 3);
  for (auto character : value) {
    result += std::format("\\{:02X}", static_cast<unsigned char>(character));
  }
  return result;
}

}  // namespace

std::string make_limited_replace_kernel(bool offset64,
                                        bool emit,
                                        bool output_offset64,
                                        bool cache,
                                        std::span<replacement_piece const> replacement,
                                        std::int32_t capture_slots,
                                        std::int32_t max_replace_count,
                                        std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false, matcher_abi::CAPTURES);
  if (emit) {
    result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions,
                                              "load_offset_general");
  }
  if (cache) {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "capture_cache");
  }
  result +=
    regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "append_range");
  for (std::size_t index = 0; index < replacement.size(); ++index) {
    auto& literal = replacement[index].literal;
    if (!literal.empty()) {
      result +=
        std::format("\n@libregex_ir_replacement_{0} = private constant [{1} x i8] c\"{2}\"\n",
                    index,
                    literal.size(),
                    llvm_bytes(literal));
    }
  }
  std::string steps;
  auto cursor = std::string{"%cursor_unmatched"};
  for (std::size_t index = 0; index < replacement.size(); ++index) {
    auto& piece      = replacement[index];
    auto next_cursor = std::format("%cursor_piece_{}", index);
    if (piece.capture.has_value()) {
      auto slot = static_cast<std::int64_t>(*piece.capture) * 2;
      steps += std::format(
        R"NVVM(  %capture_begin_ptr_{0} = getelementptr i64, i64* %captures, i64 {2}
  %capture_end_ptr_{0} = getelementptr i64, i64* %captures, i64 {3}
  %capture_begin_{0} = load i64, i64* %capture_begin_ptr_{0}, align 8
  %capture_end_{0} = load i64, i64* %capture_end_ptr_{0}, align 8
  %capture_has_begin_{0} = icmp sge i64 %capture_begin_{0}, 0
  %capture_has_end_{0} = icmp sge i64 %capture_end_{0}, 0
  %capture_present_{0} = and i1 %capture_has_begin_{0}, %capture_has_end_{0}
  %capture_selected_begin_{0} = select i1 %capture_present_{0}, i64 %capture_begin_{0}, i64 0
  %capture_selected_end_{0} = select i1 %capture_present_{0}, i64 %capture_end_{0}, i64 0
  {4} = call i64 @libregex_ir_append_range(i8* %data, i64 %capture_selected_begin_{0}, i64 %capture_selected_end_{0}, i8* %output, i64 {5})
)NVVM",
        index,
        capture_slots,
        slot,
        slot + 1,
        next_cursor,
        cursor);
    } else if (!piece.literal.empty()) {
      steps += std::format(
        R"NVVM(  %literal_{0} = getelementptr [{1} x i8], [{1} x i8]* @libregex_ir_replacement_{0}, i32 0, i32 0
  {2} = call i64 @libregex_ir_append_range(i8* %literal_{0}, i64 0, i64 {1}, i8* %output, i64 {3})
)NVVM",
        index,
        piece.literal.size(),
        next_cursor,
        cursor);
    } else {
      continue;
    }
    cursor = std::move(next_cursor);
  }
  steps += std::format("  %replacement_cursor = add i64 {}, 0\n", cursor);

  result += std::string{regex_ir::nvvm_template_section(
    regex_ir_nvvm_templates::kernel_replace,
    cache ? "limited_replace_execute_cached" : "limited_replace_execute")};
  if (cache) {
    result += std::string{regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_replace,
                                                          "limited_replace_cached")};
  }
  replace_all(result, "@REPLACEMENT_STEPS@", steps);
  replace_all(result,
              "@LIMIT_REACHED@",
              std::format("icmp uge i64 %replacement_count, {}", max_replace_count));

  if (cache) {
    result += emit ? std::string{regex_ir::nvvm_template_section(
                       regex_ir_nvvm_templates::kernel_replace, "limited_replace_cached_emit")}
                   : std::string{regex_ir::nvvm_template_section(
                       regex_ir_nvvm_templates::kernel_replace, "limited_replace_cached_size")};
  } else {
    result += emit ? std::string{regex_ir::nvvm_template_section(
                       regex_ir_nvvm_templates::kernel_replace, "limited_replace_emit")}
                   : std::string{regex_ir::nvvm_template_section(
                       regex_ir_nvvm_templates::kernel_replace, "limited_replace_size")};
  }
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@OUTPUT_OFFSET64@", output_offset64 ? "true" : "false");
  return annotate_kernel(
    std::move(result),
    cache ? (emit ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*, i32, i8*, i8*"
                  : "i8*, i8*, i32*, i32, i32, i8*, i8*, i32, i8*, i8*, i32*")
          : (emit ? "i8*, i8*, i32*, i32, i32, i8*, i8*" : "i8*, i8*, i32*, i32, i32, i8*, i32*"),
    kernel_name);
}

std::string encode_replacement(std::span<replacement_piece const> replacement)
{
  std::string result;
  for (auto& piece : replacement) {
    if (piece.capture.has_value()) {
      result += std::format("${{{}}}", *piece.capture);
      continue;
    }
    for (auto character : piece.literal) {
      result.push_back(character);
      if (character == '$') { result.push_back('$'); }
    }
  }
  return result;
}

std::string make_replace_kernel(bool offset64,
                                bool emit,
                                bool output_offset64,
                                std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false, matcher_abi::REPLACE);
  if (emit) {
    result += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions,
                                              "load_offset_general");
    result += regex_ir::render_nvvm_template_section(
      regex_ir_nvvm_templates::kernel_replace,
      "replace_emit",
      {{"@OUTPUT_OFFSET64@", output_offset64 ? "true" : "false"}});
  } else {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_replace, "replace_size");
  }
  return annotate_kernel(
    std::move(result),
    emit ? "i8*, i8*, i32*, i32, i32, i8*, i8*" : "i8*, i8*, i32*, i32, i32, i8*",
    kernel_name);
}

std::string make_split_size_kernel(bool offset64,
                                   std::int32_t maxsplit,
                                   bool cache,
                                   std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false, matcher_abi::SPLIT);
  if (cache) {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "split_cached");
  }
  auto cache_parameters = cache ? ", i8* %cache_buffer, i32 %capacity, i8* %overflow" : "";
  auto split_call =
    cache
      ? R"NVVM(  %count64 = call i64 @libregex_ir_split_cached(i8* %data, i64 %size, i8* %cache_buffer, i32 %row, i32 %capacity, i8* %overflow, i64 @MAXSPLIT@))NVVM"
      : R"NVVM(  %count64 = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* null, i64 @MAXSPLIT@, i64 -1, i8* null))NVVM";
  result += regex_ir::render_nvvm_template_section(regex_ir_nvvm_templates::kernel_split,
                                                   "split_size",
                                                   {{"@CACHE_PARAMETERS@", cache_parameters},
                                                    {"@SPLIT_CALL@", split_call},
                                                    {"@MAXSPLIT@", std::to_string(maxsplit)}});
  return annotate_kernel(
    std::move(result),
    cache ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i32, i8*" : "i8*, i8*, i32*, i32, i32, i8*",
    kernel_name);
}

std::string make_split_emit_kernel(bool offset64,
                                   bool reverse,
                                   std::int32_t maxsplit,
                                   bool overflow_only,
                                   std::string_view kernel_name)
{
  auto result             = common_nvvm(offset64, true, matcher_abi::SPLIT);
  auto overflow_parameter = overflow_only ? ", i8* %overflow" : "";
  if (overflow_only) {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "split_overflow");
  }
  auto overflow_guard =
    overflow_only
      ? R"NVVM(  %selected = call i1 @libregex_ir_split_row_selected(i1 %valid, i8* %overflow, i32 %row)
  br i1 %selected, label %setup, label %done)NVVM"
      : "  br i1 %valid, label %setup, label %done";
  auto source_index = std::string{};
  auto select_span  = std::string{};
  if (reverse) {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "split_reverse");
    source_index =
      R"NVVM(%source = call i32 @libregex_ir_reverse_split_source(i32 %full_count, i32 %effective_count, i32 %token, i1 %truncated))NVVM";
    select_span = R"NVVM(%selected_begin = add i64 %source_begin, 0
  %selected_end = call i64 @libregex_ir_reverse_split_end(i64* %row_spans, i32 %full_count, i32 %effective_count, i32 %token, i1 %truncated, i64 %source_end))NVVM";
  } else {
    result +=
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_functions, "split_forward");
    source_index = "%source = add i32 %token, 0";
    select_span  = R"NVVM(%selected_begin = add i64 %source_begin, 0
  %selected_end = call i64 @libregex_ir_forward_split_end(i32 %effective_count, i32 %token, i1 %truncated, i64 %size, i64 %source_end))NVVM";
  }
  result +=
    regex_ir::render_nvvm_template_section(regex_ir_nvvm_templates::kernel_split,
                                           "split_emit",
                                           {{"@OVERFLOW_PARAMETER@", overflow_parameter},
                                            {"@OVERFLOW_GUARD@", overflow_guard},
                                            {"@MAXSPLIT@", std::to_string(reverse ? -1 : maxsplit)},
                                            {"@SOURCE_INDEX@", source_index},
                                            {"@SELECT_SPAN@", select_span}});
  return annotate_kernel(std::move(result),
                         overflow_only ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*, i64*, i8*"
                                       : "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*, i64*",
                         kernel_name);
}

std::string make_span_cache_sample_kernel(bool offset64,
                                          bool split,
                                          std::int32_t capture_slots,
                                          std::int32_t match_limit,
                                          std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false, split ? matcher_abi::SPLIT : matcher_abi::CAPTURES);
  result += split
              ? std::string{regex_ir::nvvm_template_section(regex_ir_nvvm_templates::kernel_split,
                                                            "span_cache_sample_split")}
              : std::string{regex_ir::nvvm_template_section(
                  regex_ir_nvvm_templates::kernel_enumeration, "span_cache_sample_enumeration")};
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(std::max(capture_slots, 2)));
  replace_all(result, "@MATCH_LIMIT@", std::to_string(match_limit));
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i32, i32, i8*", kernel_name);
}

}  // namespace cudf::experimental::detail::regex_jit
