/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cudf::benchmark::regex {

template <typename Case, std::size_t N>
std::vector<std::string> case_names(std::array<Case, N> const& cases)
{
  std::vector<std::string> result;
  result.reserve(cases.size());
  std::transform(cases.begin(), cases.end(), std::back_inserter(result), [](auto& item) {
    return std::string{item.name};
  });
  return result;
}

template <typename Case, std::size_t N>
Case const* find_case(std::array<Case, N> const& cases, std::string_view name)
{
  auto found =
    std::find_if(cases.begin(), cases.end(), [name](auto& item) { return item.name == name; });
  return found == cases.end() ? nullptr : &*found;
}

inline std::uint64_t random_value(std::uint64_t& state)
{
  state ^= state >> 12U;
  state ^= state << 25U;
  state ^= state >> 27U;
  return state * 0x2545f4914f6cdd1dULL;
}

inline void inject(std::string& row, std::string_view value, std::size_t position)
{
  if (value.size() <= row.size()) {
    row.replace(std::min(position, row.size() - value.size()), value.size(), value);
  }
}

template <typename MakeSample>
std::unique_ptr<table> make_sampled_input(size_type num_rows, MakeSample make_sample)
{
  constexpr std::size_t sample_count = 64;
  std::vector<std::string> samples;
  samples.reserve(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    samples.push_back(make_sample(sample));
  }

  test::strings_column_wrapper samples_column(samples.begin(), samples.end());
  auto profile = data_profile_builder().no_validity().distribution(
    type_to_id<size_type>(), distribution_id::UNIFORM, 0ul, sample_count - 1);
  auto map = create_random_column(type_to_id<size_type>(), row_count{num_rows}, profile);
  return gather(table_view{{samples_column}}, map->view(), out_of_bounds_policy::DONT_CHECK);
}

}  // namespace cudf::benchmark::regex
