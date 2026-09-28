/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_ir.hpp"

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
  std::string result = R"NVVM(
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

define internal i32 @libregex_ir_row_index() alwaysinline nounwind readnone {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %width
  %row = add i32 %base, %thread
  ret i32 %row
}

define internal i1 @libregex_ir_is_valid(i32* %mask, i32 %row) alwaysinline nounwind readonly {
entry:
  %all_valid = icmp eq i32* %mask, null
  br i1 %all_valid, label %yes, label %check
check:
  %word_index = lshr i32 %row, 5
  %word_ptr = getelementptr i32, i32* %mask, i32 %word_index
  %word = load i32, i32* %word_ptr, align 4
  %bit_index = and i32 %row, 31
  %shifted = lshr i32 %word, %bit_index
  %bit = and i32 %shifted, 1
  %valid = icmp ne i32 %bit, 0
  ret i1 %valid
yes:
  ret i1 true
}

define internal i64 @libregex_ir_load_offset(i8* %offsets, i32 %index) alwaysinline nounwind readonly {
entry:
  %typed = bitcast i8* %offsets to @OFFSET_TYPE@*
  %ptr = getelementptr @OFFSET_TYPE@, @OFFSET_TYPE@* %typed, i32 %index
  %raw = load @OFFSET_TYPE@, @OFFSET_TYPE@* %ptr, align @OFFSET_ALIGN@
  %value = @OFFSET_EXTEND@ @OFFSET_TYPE@ %raw to i64
  ret i64 %value
}

define internal i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %position) alwaysinline nounwind readonly {
entry:
  %at_end = icmp uge i64 %position, %size
  br i1 %at_end, label %done, label %read
read:
  %ptr = getelementptr i8, i8* %data, i64 %position
  %byte = load i8, i8* %ptr, align 1
  %unsigned = zext i8 %byte to i32
  %ascii = icmp ult i32 %unsigned, 128
  %two_test = and i32 %unsigned, 224
  %is_two = icmp eq i32 %two_test, 192
  %three_test = and i32 %unsigned, 240
  %is_three = icmp eq i32 %three_test, 224
  %wide = select i1 %is_three, i64 3, i64 4
  %non_ascii = select i1 %is_two, i64 2, i64 %wide
  %width = select i1 %ascii, i64 1, i64 %non_ascii
  %advanced = add i64 %position, %width
  %clamped_test = icmp ult i64 %advanced, %size
  %clamped = select i1 %clamped_test, i64 %advanced, i64 %size
  ret i64 %clamped
done:
  ret i64 %size
}
)NVVM";
  switch (abi) {
    case matcher_abi::NONE: break;
    case matcher_abi::BOOLEAN:
      result += "\ndeclare i1 @regex_ir_execute(i8*, i64) nounwind readonly\n";
      break;
    case matcher_abi::COUNT:
      result += "\ndeclare i64 @regex_ir_execute(i8*, i64) nounwind readonly\n";
      break;
    case matcher_abi::FIND:
      result += "\ndeclare i1 @regex_ir_execute(i8*, i64, i64*) nounwind\n";
      break;
    case matcher_abi::CAPTURES:
      result += "\ndeclare i1 @regex_ir_execute(i8*, i64, i64, i64*) nounwind\n";
      break;
    case matcher_abi::REPLACE:
      result += "\ndeclare i64 @regex_ir_execute(i8*, i64, i8*) nounwind\n";
      break;
    case matcher_abi::SPLIT:
      result +=
        "\ndeclare i64 @libregex_ir_split_execute_limited(i8*, i64, i64*, i64, i64, i8*) "
        "nounwind\n";
      break;
  }
  replace_all(result, "@OFFSET_TYPE@", offset64 ? "i64" : "i32");
  replace_all(result, "@OFFSET_ALIGN@", offset64 ? "8" : "4");
  replace_all(result, "@OFFSET_EXTEND@", offset64 ? "add i64 0," : "sext");
  if (offset64) {
    replace_all(result, "%value = add i64 0, i64 %raw to i64", "%value = add i64 %raw, 0");
  }
  if (pairs) {
    result += R"NVVM(
%libregex_ir_pair = type { i8*, i32 }

define internal void @libregex_ir_write_pair(i8* %output, i64 %index, i8* %data, i8* %empty, i64 %begin, i64 %end, i1 %present) alwaysinline nounwind {
entry:
  %size64 = sub i64 %end, %begin
  %size = trunc i64 %size64 to i32
  %data_ptr = getelementptr i8, i8* %data, i64 %begin
  %is_empty = icmp eq i64 %size64, 0
  %empty_ptr = select i1 %is_empty, i8* %empty, i8* %data_ptr
  %pointer = select i1 %present, i8* %empty_ptr, i8* null
  %stored_size = select i1 %present, i32 %size, i32 0
  %typed_output = bitcast i8* %output to %libregex_ir_pair*
  %pair_ptr = getelementptr %libregex_ir_pair, %libregex_ir_pair* %typed_output, i64 %index
  %pointer_ptr = getelementptr %libregex_ir_pair, %libregex_ir_pair* %pair_ptr, i32 0, i32 0
  %size_ptr = getelementptr %libregex_ir_pair, %libregex_ir_pair* %pair_ptr, i32 0, i32 1
  store i8* %pointer, i8** %pointer_ptr, align 8
  store i32 %stored_size, i32* %size_ptr, align 4
  ret void
}
)NVVM";
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
  module += std::format(
    "\n@llvm.used = appending global [1 x i8*] [i8* bitcast (void ({})* "
    "@{} to i8*)], section \"llvm.metadata\"\n",
    signature,
    kernel_name);
  module += "\n!nvvm.annotations = !{!900000}\n";
  module +=
    std::format("!900000 = !{{void ({})* @{}, !\"kernel\", i32 1}}\n", signature, kernel_name);
  return module;
}

}  // namespace

std::string make_module(std::string wrapper)
{
  return std::format(
    R"NVVM(; cuDF regex JIT kernel wrapper
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
{}
!nvvmir.version = !{{!0}}
!0 = !{{i32 2, i32 0}}
)NVVM",
    wrapper);
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
                     "  %literal_chunk_ptr_{0} = bitcast i8* %literal_ptr_{0} to i{1}*\n"
                     "  %literal_chunk_{0} = load i{1}, i{1}* %literal_chunk_ptr_{0}, align 1\n",
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

  auto verify = literal.size() == 1U
                  ? std::string{"  br i1 %first_equal, label %local_yes, label %inner_continue\n"}
                  : std::format(
                      R"NVVM(  br i1 %first_equal, label %verify, label %inner_continue
verify:
{0}{1}  br i1 {2}, label %local_yes, label %inner_continue
)NVVM",
                      offsets,
                      comparisons,
                      matched);

  auto result = common_nvvm(offset64, false, matcher_abi::NONE);
  result += std::format(
    R"NVVM(
declare i1 @llvm.nvvm.vote.any.sync(i32, i1) nounwind convergent

define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {{
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %width
  %global = add i32 %base, %thread
  %lane = and i32 %thread, 31
  %row = lshr i32 %global, 5
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %lane64 = zext i32 %lane to i64
  %lane_base = mul nuw i64 %lane64, 4
  br label %search
search:
  %round = phi i64 [ 0, %work ], [ %next_round, %continue ]
  %first_round = icmp eq i64 %round, 0
  %round_lane = select i1 %first_round, i64 %lane64, i64 %lane_base
  %candidate_limit = select i1 %first_round, i64 1, i64 4
  %lane_position = add i64 %round, %round_lane
  br label %inner
inner:
  %candidate_offset = phi i64 [ 0, %search ], [ %next_candidate, %inner_continue ]
  %position = add i64 %lane_position, %candidate_offset
  %candidate_end = add i64 %position, {0}
  %candidate = icmp ule i64 %candidate_end, %size
  br i1 %candidate, label %compare, label %inner_done
compare:
  %first_ptr = getelementptr i8, i8* %data, i64 %position
  %first = load i8, i8* %first_ptr, align 1
  %first_equal = icmp eq i8 %first, {1}
{2}inner_continue:
  %next_candidate = add nuw i64 %candidate_offset, 1
  %more_candidates = icmp ult i64 %next_candidate, %candidate_limit
  br i1 %more_candidates, label %inner, label %inner_done
local_yes:
  br label %inner_done
inner_done:
  %local_match = phi i1 [ true, %local_yes ], [ false, %inner ], [ false, %inner_continue ]
  br label %vote
vote:
  %warp_match = call i1 @llvm.nvvm.vote.any.sync(i32 -1, i1 %local_match)
  br i1 %warp_match, label %store_true, label %continue
continue:
  %later_round = add i64 %round, 128
  %next_round = select i1 %first_round, i64 32, i64 %later_round
  %round_possible = icmp ult i64 %next_round, %size
  br i1 %round_possible, label %search, label %store_false
store_true:
  br label %store
store_false:
  br label %store
store:
  %value = phi i8 [ 1, %store_true ], [ 0, %store_false ]
  %lane_zero = icmp eq i32 %lane, 0
  %write = and i1 %lane_zero, %valid
  br i1 %write, label %write_value, label %done
write_value:
  %out = getelementptr i8, i8* %output, i32 %row
  store i8 %value, i8* %out, align 1
  br label %done
done:
  ret void
}}
)NVVM",
    literal.size(),
    static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())),
    verify);
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
    result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %match, label %store_false
match:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size)
  %value = zext i1 %matched to i8
  br label %store
store_false:
  br label %store
store:
  %result = phi i8 [ %value, %match ], [ 0, %store_false ]
  %out = getelementptr i8, i8* %output, i32 %row
  store i8 %result, i8* %out, align 1
  br label %done
done:
  ret void
}
)NVVM";
  } else if (operation == regex_ir::operation_kind::COUNT) {
    result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %count, label %store_zero
count:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %count64 = call i64 @regex_ir_execute(i8* %data, i64 %size)
  %value = trunc i64 %count64 to i32
  br label %store
store_zero:
  br label %store
store:
  %result = phi i32 [ %value, %count ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %result, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  } else {
    result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %span = alloca [2 x i64], align 8
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %find, label %store_missing
find:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %span_ptr = getelementptr [2 x i64], [2 x i64]* %span, i32 0, i32 0
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64* %span_ptr)
  br i1 %matched, label %convert, label %store_missing
convert:
  %match_begin = load i64, i64* %span_ptr, align 8
  br label %utf8_loop
utf8_loop:
  %position = phi i64 [ 0, %convert ], [ %advanced, %utf8_loop ]
  %characters = phi i32 [ 0, %convert ], [ %next_characters, %utf8_loop ]
  %at_match = icmp uge i64 %position, %match_begin
  %advanced = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %position)
  %next_characters = add i32 %characters, 1
  br i1 %at_match, label %store_found, label %utf8_loop
store_found:
  br label %store
store_missing:
  br label %store
store:
  %result = phi i32 [ %characters, %store_found ], [ -1, %store_missing ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %result, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
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
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %captures = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %match, label %output_begin
match:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %capture_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 0, i64* %capture_ptr)
  br label %output_begin
output_begin:
  %row_data = phi i8* [ %data, %match ], [ %chars, %work ]
  %row_matched = phi i1 [ %matched, %match ], [ false, %work ]
  %typed_output = getelementptr i8, i8* %output, i64 0
  br label %group_loop
group_loop:
  %group = phi i32 [ 0, %output_begin ], [ %next_group, %group_done ]
  br i1 %row_matched, label %group_found, label %group_missing
group_found:
  %capture_group = add i32 %group, @FIRST_GROUP@
  %capture_plus_whole = add i32 %capture_group, 1
  %capture_slot = shl i32 %capture_plus_whole, 1
  %capture_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 %capture_slot
  %capture_end_slot = add i32 %capture_slot, 1
  %capture_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 %capture_end_slot
  %capture_begin = load i64, i64* %capture_begin_ptr, align 8
  %capture_end = load i64, i64* %capture_end_ptr, align 8
  %has_begin = icmp sge i64 %capture_begin, 0
  %has_end = icmp sge i64 %capture_end, 0
  %present = and i1 %has_begin, %has_end
  br label %group_write
group_missing:
  br label %group_write
group_write:
  %stored_begin = phi i64 [ %capture_begin, %group_found ], [ 0, %group_missing ]
  %stored_end = phi i64 [ %capture_end, %group_found ], [ 0, %group_missing ]
  %stored_present = phi i1 [ %present, %group_found ], [ false, %group_missing ]
  %group64 = sext i32 %group to i64
  @PAIR_INDEX@
  call void @libregex_ir_write_pair(i8* %typed_output, i64 %pair_index, i8* %row_data, i8* %offsets, i64 %stored_begin, i64 %stored_end, i1 %stored_present)
  br label %group_done
group_done:
  %next_group = add i32 %group, 1
  %finished = icmp eq i32 %next_group, @OUTPUT_GROUPS@
  br i1 %finished, label %done, label %group_loop
done:
  ret void
}
)NVVM";
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@FIRST_GROUP@", std::to_string(first_group));
  replace_all(result, "@OUTPUT_GROUPS@", std::to_string(output_groups));
  replace_all(result,
              "@PAIR_INDEX@",
              column_major ? "%group_base = mul i64 %group64, %rows64\n  %row64 = sext i32 %row to "
                             "i64\n  %pair_index = add i64 %group_base, %row64"
                           : "%pair_index = sext i32 %row to i64");
  if (column_major) {
    replace_all(result,
                "%typed_output = getelementptr i8, i8* %output, i64 0",
                "%typed_output = getelementptr i8, i8* %output, i64 0\n  %rows64 = sext i32 "
                "%rows to i64");
  }
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i8*", kernel_name);
}

std::string capture_cache_helpers();

std::string make_enumeration_size_kernel(bool offset64,
                                         std::int32_t capture_slots,
                                         std::int32_t multiplier,
                                         bool require_match,
                                         bool cache,
                                         std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false, matcher_abi::CAPTURES);
  if (cache) { result += capture_cache_helpers(); }
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_validity) nounwind {
entry:
  %captures = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %setup, label %store_invalid
setup:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %capture_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  br label %match_loop
match_loop:
  %search = phi i64 [ 0, %setup ], [ %match_end, %continue_nonempty ], [ %advanced, %continue_empty ]
  %count = phi i64 [ 0, %setup ], [ %next_count, %continue_nonempty ], [ %next_count, %continue_empty ]
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search, i64* %capture_ptr)
  br i1 %matched, label %found, label %store_count
found:
  %match_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %match_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  %next_count = add i64 %count, 1
  %nonempty = icmp ne i64 %match_begin, %match_end
  br i1 %nonempty, label %continue_nonempty, label %empty
continue_nonempty:
  br label %match_loop
empty:
  %empty_at_end = icmp eq i64 %match_end, %size
  br i1 %empty_at_end, label %store_after_match, label %continue_empty
continue_empty:
  %advanced = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %match_end)
  br label %match_loop
store_after_match:
  br label %store
store_count:
  br label %store
store_invalid:
  br label %store
store:
  %matches = phi i64 [ %next_count, %store_after_match ], [ %count, %store_count ], [ 0, %store_invalid ]
  %scaled = mul i64 %matches, @MULTIPLIER@
  %stored_count = trunc i64 %scaled to i32
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %stored_count, i32* %out, align 4
  %has_match = icmp ne i64 %matches, 0
  %row_valid = @ROW_VALID@
  %valid_byte = zext i1 %row_valid to i8
  %valid_out = getelementptr i8, i8* %output_validity, i32 %row
  store i8 %valid_byte, i8* %valid_out, align 1
  br label %done
done:
  ret void
}
)NVVM";
  if (cache) {
    replace_all(result,
                "i8* %output, i8* %output_validity) nounwind {",
                "i8* %output, i8* %output_validity, i8* %cache_buffer, i32 %capacity, i8* "
                "%overflow) nounwind {");
    replace_all(
      result,
      "  %match_end = load i64, i64* %match_end_ptr, align 8\n  %next_count =",
      std::format(
        "  %match_end = load i64, i64* %match_end_ptr, align 8\n  %typed_cache = bitcast i8* "
        "%cache_buffer to i64*\n  %overflow_ptr = getelementptr i8, i8* %overflow, i32 %row\n  "
        "call void @libregex_ir_cache_captures(i64* %capture_ptr, i64* %typed_cache, i32 %row, i32 "
        "%capacity, i32 {}, i64 %count, i8* %overflow_ptr)\n  %next_count =",
        capture_slots));
  }
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@MULTIPLIER@", std::to_string(multiplier));
  replace_all(
    result, "@ROW_VALID@", require_match ? "and i1 %valid, %has_match" : "and i1 %valid, true");
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
  auto result = common_nvvm(offset64, true, matcher_abi::CAPTURES);
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_offsets) nounwind {
entry:
  %captures = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %setup, label %done
setup:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %capture_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %typed_offsets = bitcast i8* %output_offsets to i32*
  %row_output_ptr = getelementptr i32, i32* %typed_offsets, i32 %row
  %row_output = load i32, i32* %row_output_ptr, align 4
  %row_output64 = sext i32 %row_output to i64
  %typed_output = getelementptr i8, i8* %output, i64 0
  br label %match_loop
match_loop:
  %search = phi i64 [ 0, %setup ], [ %match_end, %continue_nonempty ], [ %advanced, %continue_empty ]
  %output_index = phi i64 [ 0, %setup ], [ %next_output_index, %continue_nonempty ], [ %next_output_index, %continue_empty ]
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search, i64* %capture_ptr)
  br i1 %matched, label %found, label %done
found:
  %match_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %match_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  @WRITE_MATCH@
after_write:
  %nonempty = icmp ne i64 %match_begin, %match_end
  br i1 %nonempty, label %continue_nonempty, label %empty
continue_nonempty:
  br label %match_loop
empty:
  %empty_at_end = icmp eq i64 %match_end, %size
  br i1 %empty_at_end, label %done, label %continue_empty
continue_empty:
  %advanced = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %match_end)
  br label %match_loop
done:
  ret void
}
)NVVM";
  if (overflow_only) {
    replace_all(result,
                "i8* %output, i8* %output_offsets) nounwind {",
                "i8* %output, i8* %output_offsets, i8* %overflow) nounwind {");
    replace_all(
      result,
      "  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)\n  br i1 %valid, "
      "label %setup, label %done",
      "  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)\n  %overflow_ptr = "
      "getelementptr i8, i8* %overflow, i32 %row\n  %overflow_value = load i8, i8* %overflow_ptr, "
      "align 1\n  %did_overflow = icmp ne i8 %overflow_value, 0\n  %selected = and i1 %valid, "
      "%did_overflow\n  br i1 %selected, label %setup, label %done");
  }
  auto write_findall =
    groups == 0
      ? R"NVVM(%selected_begin = add i64 %match_begin, 0
  %selected_end = add i64 %match_end, 0
  %present = icmp sge i64 %selected_begin, 0
  %pair_index = add i64 %row_output64, %output_index
  call void @libregex_ir_write_pair(i8* %typed_output, i64 %pair_index, i8* %data, i8* %offsets, i64 %selected_begin, i64 %selected_end, i1 %present)
  %next_output_index = add i64 %output_index, 1
  br label %after_write)NVVM"
      : R"NVVM(%selected_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 2
  %selected_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 3
  %selected_begin = load i64, i64* %selected_begin_ptr, align 8
  %selected_end = load i64, i64* %selected_end_ptr, align 8
  %has_begin = icmp sge i64 %selected_begin, 0
  %has_end = icmp sge i64 %selected_end, 0
  %present = and i1 %has_begin, %has_end
  %pair_index = add i64 %row_output64, %output_index
  call void @libregex_ir_write_pair(i8* %typed_output, i64 %pair_index, i8* %data, i8* %offsets, i64 %selected_begin, i64 %selected_end, i1 %present)
  %next_output_index = add i64 %output_index, 1
  br label %after_write)NVVM";
  auto write_extract = R"NVVM(br label %group_loop
group_loop:
  %group = phi i32 [ 0, %found ], [ %next_group, %group_write ]
  %capture_plus_whole = add i32 %group, 1
  %capture_slot = shl i32 %capture_plus_whole, 1
  %capture_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 %capture_slot
  %capture_end_slot = add i32 %capture_slot, 1
  %capture_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 %capture_end_slot
  %capture_begin = load i64, i64* %capture_begin_ptr, align 8
  %capture_end = load i64, i64* %capture_end_ptr, align 8
  %has_begin = icmp sge i64 %capture_begin, 0
  %has_end = icmp sge i64 %capture_end, 0
  %present = and i1 %has_begin, %has_end
  %group64 = sext i32 %group to i64
  %group_output = add i64 %output_index, %group64
  %pair_index = add i64 %row_output64, %group_output
  call void @libregex_ir_write_pair(i8* %typed_output, i64 %pair_index, i8* %data, i8* %offsets, i64 %capture_begin, i64 %capture_end, i1 %present)
  br label %group_write
group_write:
  %next_group = add i32 %group, 1
  %groups_done = icmp eq i32 %next_group, @GROUPS@
  br i1 %groups_done, label %groups_finished, label %group_loop
groups_finished:
  %next_output_index = add i64 %output_index, @GROUPS@
  br label %after_write)NVVM";
  replace_all(result, "@WRITE_MATCH@", findall ? write_findall : write_extract);
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@GROUPS@", std::to_string(groups));
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

std::string capture_cache_helpers()
{
  return R"NVVM(
define internal void @libregex_ir_cache_captures(i64* %captures, i64* %cache, i32 %row, i32 %capacity, i32 %values, i64 %match, i8* %overflow) nounwind {
entry:
  %missing = icmp eq i64* %cache, null
  br i1 %missing, label %done, label %check_capacity
check_capacity:
  %capacity64 = zext i32 %capacity to i64
  %fits = icmp ult i64 %match, %capacity64
  br i1 %fits, label %copy_setup, label %mark_overflow
mark_overflow:
  store i8 1, i8* %overflow, align 1
  br label %done
copy_setup:
  %row64 = zext i32 %row to i64
  %values64 = zext i32 %values to i64
  %row_matches = mul i64 %row64, %capacity64
  %record = add i64 %row_matches, %match
  %base = mul i64 %record, %values64
  br label %copy_loop
copy_loop:
  %slot = phi i32 [ 0, %copy_setup ], [ %next_slot, %copy_loop ]
  %slot64 = zext i32 %slot to i64
  %source = getelementptr i64, i64* %captures, i64 %slot64
  %value = load i64, i64* %source, align 8
  %index = add i64 %base, %slot64
  %destination = getelementptr i64, i64* %cache, i64 %index
  store i64 %value, i64* %destination, align 8
  %next_slot = add i32 %slot, 1
  %finished = icmp eq i32 %next_slot, %values
  br i1 %finished, label %done, label %copy_loop
done:
  ret void
}
)NVVM";
}

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
  if (cache) { result += capture_cache_helpers(); }
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
  result += R"NVVM(
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

define internal i64 @libregex_ir_append_range(i8* %source, i64 %begin, i64 %end, i8* %output, i64 %cursor) alwaysinline nounwind {
entry:
  %length = sub i64 %end, %begin
  %next_cursor = add i64 %cursor, %length
  %output_missing = icmp eq i8* %output, null
  %empty = icmp eq i64 %length, 0
  %skip = or i1 %output_missing, %empty
  br i1 %skip, label %done, label %copy
copy:
  %source_ptr = getelementptr i8, i8* %source, i64 %begin
  %output_ptr = getelementptr i8, i8* %output, i64 %cursor
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %output_ptr, i8* align 1 %source_ptr, i64 %length, i1 false)
  br label %done
done:
  ret i64 %next_cursor
}
)NVVM";

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

  result += R"NVVM(
define internal i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output) nounwind {
entry:
  %capture_array = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %captures = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %capture_array, i32 0, i32 0
  br label %loop
loop:
  %search_start = phi i64 [ 0, %entry ], [ %match_end, %continue_nonempty ], [ %advanced_start, %continue_empty ]
  %copied = phi i64 [ 0, %entry ], [ %match_end, %continue_nonempty ], [ %match_end, %continue_empty ]
  %cursor = phi i64 [ 0, %entry ], [ %replacement_cursor, %continue_nonempty ], [ %replacement_cursor, %continue_empty ]
  %replacement_count = phi i64 [ 0, %entry ], [ %next_replacement_count, %continue_nonempty ], [ %next_replacement_count, %continue_empty ]
  %limit_reached = @LIMIT_REACHED@
  br i1 %limit_reached, label %finish_limit, label %search
search:
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search_start, i64* %captures)
  br i1 %matched, label %found, label %no_match
found:
  %match_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %capture_array, i32 0, i32 0
  %match_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %capture_array, i32 0, i32 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  %cursor_unmatched = call i64 @libregex_ir_append_range(i8* %data, i64 %copied, i64 %match_begin, i8* %output, i64 %cursor)
@REPLACEMENT_STEPS@  %next_replacement_count = add i64 %replacement_count, 1
  %nonempty = icmp ne i64 %match_begin, %match_end
  br i1 %nonempty, label %continue_nonempty, label %empty
continue_nonempty:
  br label %loop
empty:
  %empty_at_end = icmp eq i64 %match_end, %size
  br i1 %empty_at_end, label %finish_empty, label %continue_empty
continue_empty:
  %advanced_start = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %match_end)
  br label %loop
finish_limit:
  br label %finish
no_match:
  br label %finish
finish_empty:
  br label %finish
finish:
  %tail_begin = phi i64 [ %copied, %finish_limit ], [ %copied, %no_match ], [ %match_end, %finish_empty ]
  %tail_cursor = phi i64 [ %cursor, %finish_limit ], [ %cursor, %no_match ], [ %replacement_cursor, %finish_empty ]
  %final_cursor = call i64 @libregex_ir_append_range(i8* %data, i64 %tail_begin, i64 %size, i8* %output, i64 %tail_cursor)
  ret i64 %final_cursor
}
)NVVM";
  if (cache) {
    replace_all(
      result,
      "define internal i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output) "
      "nounwind {",
      "define internal i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output, i64* "
      "%cache, i32 %row, i32 %capacity, i8* %overflow, i32* %count_out) nounwind {");
    replace_all(
      result,
      "  %match_end = load i64, i64* %match_end_ptr, align 8\n  %cursor_unmatched =",
      std::format(
        "  %match_end = load i64, i64* %match_end_ptr, align 8\n  call void "
        "@libregex_ir_cache_captures(i64* %captures, i64* %cache, i32 %row, i32 %capacity, i32 {}, "
        "i64 %replacement_count, i8* %overflow)\n  %cursor_unmatched =",
        capture_slots));
    replace_all(
      result,
      "  %final_cursor = call i64 @libregex_ir_append_range(i8* %data, i64 %tail_begin, i64 %size, "
      "i8* %output, i64 %tail_cursor)\n  ret i64 %final_cursor",
      "  %final_count = phi i64 [ %replacement_count, %finish_limit ], [ %replacement_count, "
      "%no_match ], [ %next_replacement_count, %finish_empty ]\n  %stored_count = trunc i64 "
      "%final_count to i32\n  store i32 %stored_count, i32* %count_out, align 4\n  %final_cursor = "
      "call i64 @libregex_ir_append_range(i8* %data, i64 %tail_begin, i64 %size, i8* %output, i64 "
      "%tail_cursor)\n  ret i64 %final_cursor");
    result += R"NVVM(
define internal i64 @libregex_ir_replace_cached(i8* %data, i64 %size, i8* %output, i64* %cache, i32 %row, i32 %capacity, i32 %match_count) nounwind {
entry:
  %row64 = zext i32 %row to i64
  %capacity64 = zext i32 %capacity to i64
  %slots64 = zext i32 @CAPTURE_SLOTS@ to i64
  %row_records = mul i64 %row64, %capacity64
  %row_base = mul i64 %row_records, %slots64
  br label %loop
loop:
  %match = phi i32 [ 0, %entry ], [ %next_match, %copy_match ]
  %copied = phi i64 [ 0, %entry ], [ %match_end, %copy_match ]
  %cursor = phi i64 [ 0, %entry ], [ %replacement_cursor, %copy_match ]
  %finished = icmp eq i32 %match, %match_count
  br i1 %finished, label %finish, label %copy_match
copy_match:
  %match64 = zext i32 %match to i64
  %record_offset = mul i64 %match64, %slots64
  %capture_offset = add i64 %row_base, %record_offset
  %captures = getelementptr i64, i64* %cache, i64 %capture_offset
  %match_begin_ptr = getelementptr i64, i64* %captures, i64 0
  %match_end_ptr = getelementptr i64, i64* %captures, i64 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  %cursor_unmatched = call i64 @libregex_ir_append_range(i8* %data, i64 %copied, i64 %match_begin, i8* %output, i64 %cursor)
@REPLACEMENT_STEPS@  %next_match = add i32 %match, 1
  br label %loop
finish:
  %final_cursor = call i64 @libregex_ir_append_range(i8* %data, i64 %copied, i64 %size, i8* %output, i64 %cursor)
  ret i64 %final_cursor
}
)NVVM";
  }
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@REPLACEMENT_STEPS@", steps);
  replace_all(result,
              "@LIMIT_REACHED@",
              std::format("icmp uge i64 %replacement_count, {}", max_replace_count));

  auto output_offset_type   = output_offset64 ? std::string{"i64"} : std::string{"i32"};
  auto output_offset_align  = output_offset64 ? std::string{"8"} : std::string{"4"};
  auto extend_output_offset = output_offset64
                                ? std::string{"%output_offset64 = add i64 %output_offset, 0"}
                                : std::string{"%output_offset64 = sext i32 %output_offset to i64"};

  result += emit ? R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_offsets) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %done
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_output_offsets = bitcast i8* %output_offsets to @OUTPUT_OFFSET_TYPE@*
  %output_offset_ptr = getelementptr @OUTPUT_OFFSET_TYPE@, @OUTPUT_OFFSET_TYPE@* %typed_output_offsets, i32 %row
  %output_offset = load @OUTPUT_OFFSET_TYPE@, @OUTPUT_OFFSET_TYPE@* %output_offset_ptr, align @OUTPUT_OFFSET_ALIGN@
  @EXTEND_OUTPUT_OFFSET@
  %output_ptr = getelementptr i8, i8* %output, i64 %output_offset64
  %written = call i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output_ptr)
  br label %done
done:
  ret void
}
)NVVM"
                 : R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i32* %size_overflow) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %store_zero
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %result_size = call i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* null)
  %too_large = icmp ugt i64 %result_size, 2147483647
  br i1 %too_large, label %mark_overflow, label %size_fits
mark_overflow:
  %overflow_old = atomicrmw max i32* %size_overflow, i32 1 monotonic
  br label %size_fits
size_fits:
  %stored_size = trunc i64 %result_size to i32
  br label %store
store_zero:
  br label %store
store:
  %value = phi i32 [ %stored_size, %size_fits ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %value, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  if (cache) {
    auto kernel_position = result.rfind("define void @KERNEL_ENTRY@");
    result.resize(kernel_position);
    result +=
      emit
        ? R"NVVM(define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_offsets, i8* %cache_buffer, i32 %capacity, i8* %overflow, i8* %match_counts) nounwind {
entry:
  %dummy_count = alloca i32, align 4
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %done
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_output_offsets = bitcast i8* %output_offsets to @OUTPUT_OFFSET_TYPE@*
  %output_offset_ptr = getelementptr @OUTPUT_OFFSET_TYPE@, @OUTPUT_OFFSET_TYPE@* %typed_output_offsets, i32 %row
  %output_offset = load @OUTPUT_OFFSET_TYPE@, @OUTPUT_OFFSET_TYPE@* %output_offset_ptr, align @OUTPUT_OFFSET_ALIGN@
  @EXTEND_OUTPUT_OFFSET@
  %output_ptr = getelementptr i8, i8* %output, i64 %output_offset64
  %overflow_ptr = getelementptr i8, i8* %overflow, i32 %row
  %row_overflow = load i8, i8* %overflow_ptr, align 1
  %did_overflow = icmp ne i8 %row_overflow, 0
  br i1 %did_overflow, label %rematch, label %cached
rematch:
  %rematched = call i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output_ptr, i64* null, i32 %row, i32 %capacity, i8* %overflow_ptr, i32* %dummy_count)
  br label %done
cached:
  %typed_cache = bitcast i8* %cache_buffer to i64*
  %typed_counts = bitcast i8* %match_counts to i32*
  %count_ptr = getelementptr i32, i32* %typed_counts, i32 %row
  %match_count = load i32, i32* %count_ptr, align 4
  %written = call i64 @libregex_ir_replace_cached(i8* %data, i64 %size, i8* %output_ptr, i64* %typed_cache, i32 %row, i32 %capacity, i32 %match_count)
  br label %done
done:
  ret void
}
)NVVM"
        : R"NVVM(define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %cache_buffer, i32 %capacity, i8* %overflow, i8* %match_counts, i32* %size_overflow) nounwind {
entry:
  %typed_counts = bitcast i8* %match_counts to i32*
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %store_zero
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_cache = bitcast i8* %cache_buffer to i64*
  %overflow_ptr = getelementptr i8, i8* %overflow, i32 %row
  %count_ptr = getelementptr i32, i32* %typed_counts, i32 %row
  %result_size = call i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* null, i64* %typed_cache, i32 %row, i32 %capacity, i8* %overflow_ptr, i32* %count_ptr)
  %too_large = icmp ugt i64 %result_size, 2147483647
  br i1 %too_large, label %mark_overflow, label %size_fits
mark_overflow:
  %overflow_old = atomicrmw max i32* %size_overflow, i32 1 monotonic
  br label %size_fits
size_fits:
  %stored_size = trunc i64 %result_size to i32
  br label %store
store_zero:
  %invalid_count_ptr = getelementptr i32, i32* %typed_counts, i32 %row
  store i32 0, i32* %invalid_count_ptr, align 4
  br label %store
store:
  %value = phi i32 [ %stored_size, %size_fits ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %value, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  }
  replace_all(result, "@OUTPUT_OFFSET_TYPE@", output_offset_type);
  replace_all(result, "@OUTPUT_OFFSET_ALIGN@", output_offset_align);
  replace_all(result, "@EXTEND_OUTPUT_OFFSET@", extend_output_offset);
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
  result += emit ? R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_offsets) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %done
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_output_offsets = bitcast i8* %output_offsets to @OUTPUT_TYPE@*
  %output_offset_ptr = getelementptr @OUTPUT_TYPE@, @OUTPUT_TYPE@* %typed_output_offsets, i32 %row
  %raw_output_offset = load @OUTPUT_TYPE@, @OUTPUT_TYPE@* %output_offset_ptr, align @OUTPUT_ALIGN@
  %output_offset = @OUTPUT_EXTEND@ @OUTPUT_TYPE@ %raw_output_offset to i64
  %output_ptr = getelementptr i8, i8* %output, i64 %output_offset
  %written = call i64 @regex_ir_execute(i8* %data, i64 %size, i8* %output_ptr)
  br label %done
done:
  ret void
}
)NVVM"
                 : R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %store_zero
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %result_size = call i64 @regex_ir_execute(i8* %data, i64 %size, i8* null)
  %stored_size = trunc i64 %result_size to i32
  br label %store
store_zero:
  br label %store
store:
  %value = phi i32 [ %stored_size, %replace ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %value, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  if (emit) {
    replace_all(result, "@OUTPUT_TYPE@", output_offset64 ? "i64" : "i32");
    replace_all(result, "@OUTPUT_ALIGN@", output_offset64 ? "8" : "4");
    replace_all(result, "@OUTPUT_EXTEND@", output_offset64 ? "add i64 0," : "sext");
    if (output_offset64) {
      replace_all(result,
                  "%output_offset = add i64 0, i64 %raw_output_offset to i64",
                  "%output_offset = add i64 %raw_output_offset, 0");
    }
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
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %split, label %store_zero
split:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %count64 = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* null, i64 @MAXSPLIT@, i64 -1, i8* null)
  %count = trunc i64 %count64 to i32
  br label %store
store_zero:
  br label %store
store:
  %stored_count = phi i32 [ %count, %split ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %stored_count, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  if (cache) {
    replace_all(result,
                "i8* %output) nounwind {",
                "i8* %output, i8* %cache_buffer, i32 %capacity, i8* %overflow) nounwind {");
    replace_all(
      result,
      "  %count64 = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* null, "
      "i64 @MAXSPLIT@, i64 -1, i8* null)",
      "  %typed_cache = bitcast i8* %cache_buffer to i64*\n  %row64 = zext i32 %row to i64\n  "
      "%fields_per_row = add i32 %capacity, 1\n  %fields_per_row64 = zext i32 %fields_per_row "
      "to i64\n  %row_fields = mul i64 %row64, %fields_per_row64\n  %row_base = shl i64 "
      "%row_fields, 1\n  %row_spans = getelementptr i64, i64* %typed_cache, i64 %row_base\n  "
      "%overflow_ptr = getelementptr i8, i8* %overflow, i32 %row\n  %count64 = call i64 "
      "@libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* %row_spans, i64 "
      "@MAXSPLIT@, i64 %fields_per_row64, i8* %overflow_ptr)");
  }
  replace_all(result, "@MAXSPLIT@", std::to_string(maxsplit));
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
  auto result = common_nvvm(offset64, true, matcher_abi::SPLIT);
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %effective_offsets, i8* %full_offsets, i64* %spans) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %setup, label %done
setup:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_effective_offsets = bitcast i8* %effective_offsets to i32*
  %effective_begin_ptr = getelementptr i32, i32* %typed_effective_offsets, i32 %row
  %effective_next = add i32 %row, 1
  %effective_end_ptr = getelementptr i32, i32* %typed_effective_offsets, i32 %effective_next
  %effective_begin = load i32, i32* %effective_begin_ptr, align 4
  %effective_end = load i32, i32* %effective_end_ptr, align 4
  %effective_count = sub i32 %effective_end, %effective_begin
  %typed_full_offsets = bitcast i8* %full_offsets to i32*
  %full_begin_ptr = getelementptr i32, i32* %typed_full_offsets, i32 %row
  %full_next = add i32 %row, 1
  %full_end_ptr = getelementptr i32, i32* %typed_full_offsets, i32 %full_next
  %full_begin = load i32, i32* %full_begin_ptr, align 4
  %full_end = load i32, i32* %full_end_ptr, align 4
  %full_count = sub i32 %full_end, %full_begin
  %full_begin64 = sext i32 %full_begin to i64
  %span_base_index = shl i64 %full_begin64, 1
  %row_spans = getelementptr i64, i64* %spans, i64 %span_base_index
  %written_fields = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* %row_spans, i64 @MAXSPLIT@, i64 -1, i8* null)
  %typed_output = getelementptr i8, i8* %output, i64 0
  %effective_begin64 = sext i32 %effective_begin to i64
  %truncated = icmp sgt i32 %full_count, %effective_count
  br label %token_loop
token_loop:
  %token = phi i32 [ 0, %setup ], [ %next_token, %token_loop ]
  @SOURCE_INDEX@
  %source64 = sext i32 %source to i64
  %source_base = shl i64 %source64, 1
  %source_end_index = add i64 %source_base, 1
  %source_begin_ptr = getelementptr i64, i64* %row_spans, i64 %source_base
  %source_end_ptr = getelementptr i64, i64* %row_spans, i64 %source_end_index
  %source_begin = load i64, i64* %source_begin_ptr, align 8
  %source_end = load i64, i64* %source_end_ptr, align 8
  @SELECT_SPAN@
  %token64 = sext i32 %token to i64
  %pair_index = add i64 %effective_begin64, %token64
  call void @libregex_ir_write_pair(i8* %typed_output, i64 %pair_index, i8* %data, i8* %offsets, i64 %selected_begin, i64 %selected_end, i1 true)
  %next_token = add i32 %token, 1
  %finished = icmp eq i32 %next_token, %effective_count
  br i1 %finished, label %done, label %token_loop
done:
  ret void
}
)NVVM";
  if (overflow_only) {
    replace_all(
      result,
      "i8* %effective_offsets, i8* %full_offsets, i64* %spans) nounwind {",
      "i8* %effective_offsets, i8* %full_offsets, i64* %spans, i8* %overflow) nounwind {");
    replace_all(
      result,
      "  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)\n  br i1 %valid, "
      "label %setup, label %done",
      "  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)\n  %overflow_ptr = "
      "getelementptr i8, i8* %overflow, i32 %row\n  %overflow_value = load i8, i8* "
      "%overflow_ptr, align 1\n  %did_overflow = icmp ne i8 %overflow_value, 0\n  %selected = "
      "and i1 %valid, %did_overflow\n  br i1 %selected, label %setup, label %done");
  }
  replace_all(result, "@MAXSPLIT@", std::to_string(reverse ? -1 : maxsplit));
  if (reverse) {
    replace_all(result,
                "@SOURCE_INDEX@",
                R"NVVM(%removed = sub i32 %full_count, %effective_count
  %is_first = icmp eq i32 %token, 0
  %shifted = add i32 %removed, %token
  %truncated_source = select i1 %is_first, i32 0, i32 %shifted
  %source = select i1 %truncated, i32 %truncated_source, i32 %token)NVVM");
    replace_all(result,
                "@SELECT_SPAN@",
                R"NVVM(%merged_field = sub i32 %full_count, %effective_count
  %merged_field64 = sext i32 %merged_field to i64
  %merged_field_base = shl i64 %merged_field64, 1
  %merged_end_index = add i64 %merged_field_base, 1
  %merged_end_ptr = getelementptr i64, i64* %row_spans, i64 %merged_end_index
  %merged_end = load i64, i64* %merged_end_ptr, align 8
  %is_first_selected = and i1 %truncated, %is_first
  %selected_begin = add i64 %source_begin, 0
  %selected_end = select i1 %is_first_selected, i64 %merged_end, i64 %source_end)NVVM");
  } else {
    replace_all(result, "@SOURCE_INDEX@", "%source = add i32 %token, 0");
    replace_all(result,
                "@SELECT_SPAN@",
                R"NVVM(%last_token = sub i32 %effective_count, 1
  %is_last = icmp eq i32 %token, %last_token
  %merge_tail = and i1 %truncated, %is_last
  %selected_begin = add i64 %source_begin, 0
  %selected_end = select i1 %merge_tail, i64 %size, i64 %source_end)NVVM");
  }
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
  result += split ? R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i32 %samples, i32 %capacity, i8* %statistics) nounwind {
entry:
  %sample = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %sample, %samples
  br i1 %in_bounds, label %select, label %done
select:
  %sample64 = zext i32 %sample to i64
  %rows64 = zext i32 %rows to i64
  %samples64 = zext i32 %samples to i64
  %scaled = mul i64 %sample64, %rows64
  %selected64 = udiv i64 %scaled, %samples64
  %selected = trunc i64 %selected64 to i32
  %physical = add i32 %row_offset, %selected
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %sample_row, label %done
sample_row:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %fields = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* null, i64 @MATCH_LIMIT@, i64 -1, i8* null)
  %matches = sub i64 %fields, 1
  br label %record
record:
  %capacity64 = zext i32 %capacity to i64
  %overflow = icmp ugt i64 %matches, %capacity64
  %cap_plus_one = add i64 %capacity64, 1
  %sampled_matches = select i1 %overflow, i64 %cap_plus_one, i64 %matches
  %stats = bitcast i8* %statistics to i64*
  %valid_ptr = getelementptr i64, i64* %stats, i64 0
  %bytes_ptr = getelementptr i64, i64* %stats, i64 1
  %matches_ptr = getelementptr i64, i64* %stats, i64 2
  %overflow_ptr = getelementptr i64, i64* %stats, i64 3
  %valid_old = atomicrmw add i64* %valid_ptr, i64 1 monotonic
  %bytes_old = atomicrmw add i64* %bytes_ptr, i64 %size monotonic
  %matches_old = atomicrmw add i64* %matches_ptr, i64 %sampled_matches monotonic
  %overflow64 = zext i1 %overflow to i64
  %overflow_old = atomicrmw add i64* %overflow_ptr, i64 %overflow64 monotonic
  br label %done
done:
  ret void
}
)NVVM"
                  : R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i32 %samples, i32 %capacity, i8* %statistics) nounwind {
entry:
  %captures = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %sample = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %sample, %samples
  br i1 %in_bounds, label %select, label %done
select:
  %sample64 = zext i32 %sample to i64
  %rows64 = zext i32 %rows to i64
  %samples64 = zext i32 %samples to i64
  %scaled = mul i64 %sample64, %rows64
  %selected64 = udiv i64 %scaled, %samples64
  %selected = trunc i64 %selected64 to i32
  %physical = add i32 %row_offset, %selected
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %sample_row, label %done
sample_row:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %capture_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  br label %match_loop
match_loop:
  %search = phi i64 [ 0, %sample_row ], [ %match_end, %continue_nonempty ], [ %advanced, %continue_empty ]
  %count = phi i64 [ 0, %sample_row ], [ %next_count, %continue_nonempty ], [ %next_count, %continue_empty ]
  %semantic_limited = icmp sge i64 @MATCH_LIMIT@, 0
  %at_semantic_limit = icmp eq i64 %count, @MATCH_LIMIT@
  %semantic_done = and i1 %semantic_limited, %at_semantic_limit
  br i1 %semantic_done, label %record, label %search_match
search_match:
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search, i64* %capture_ptr)
  br i1 %matched, label %found, label %record
found:
  %match_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %match_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  %next_count = add i64 %count, 1
  %capacity64_found = zext i32 %capacity to i64
  %overflow_found = icmp ugt i64 %next_count, %capacity64_found
  br i1 %overflow_found, label %record_overflow, label %check_empty
check_empty:
  %nonempty = icmp ne i64 %match_begin, %match_end
  br i1 %nonempty, label %continue_nonempty, label %empty
continue_nonempty:
  br label %match_loop
empty:
  %empty_at_end = icmp eq i64 %match_end, %size
  br i1 %empty_at_end, label %record_after_match, label %continue_empty
continue_empty:
  %advanced = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %match_end)
  br label %match_loop
record_after_match:
  br label %record
record_overflow:
  br label %record
record:
  %matches = phi i64 [ %count, %match_loop ], [ %count, %search_match ], [ %next_count, %record_after_match ], [ %next_count, %record_overflow ]
  %capacity64 = zext i32 %capacity to i64
  %overflow = icmp ugt i64 %matches, %capacity64
  %stats = bitcast i8* %statistics to i64*
  %valid_ptr = getelementptr i64, i64* %stats, i64 0
  %bytes_ptr = getelementptr i64, i64* %stats, i64 1
  %matches_ptr = getelementptr i64, i64* %stats, i64 2
  %overflow_ptr = getelementptr i64, i64* %stats, i64 3
  %valid_old = atomicrmw add i64* %valid_ptr, i64 1 monotonic
  %bytes_old = atomicrmw add i64* %bytes_ptr, i64 %size monotonic
  %matches_old = atomicrmw add i64* %matches_ptr, i64 %matches monotonic
  %overflow64 = zext i1 %overflow to i64
  %overflow_old = atomicrmw add i64* %overflow_ptr, i64 %overflow64 monotonic
  br label %done
done:
  ret void
}
)NVVM";
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(std::max(capture_slots, 2)));
  replace_all(result, "@MATCH_LIMIT@", std::to_string(match_limit));
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i32, i32, i8*", kernel_name);
}

}  // namespace cudf::experimental::detail::regex_jit
