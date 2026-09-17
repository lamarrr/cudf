/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/dictionary/encode.hpp>
#include <cudf/hashing.hpp>

class XXHash_32_Test : public cudf::test::BaseFixture {};

TEST_F(XXHash_32_Test, TestInteger)
{
  auto col1             = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 42, 825}};
  auto constexpr seed   = 0u;
  auto const output     = cudf::hashing::xxhash_32(cudf::table_view({col1}), seed);
  auto const jit_output = cudf::hashing::xxhash_32_jit(cudf::table_view({col1}), seed);

  // Expected results were generated with the reference implementation:
  // https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h
  auto expected =
    cudf::test::fixed_width_column_wrapper<uint32_t>({148298089u, 1161967057u, 1066694813u});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(jit_output->view(), expected);
}

TEST_F(XXHash_32_Test, TestDouble)
{
  auto col1           = cudf::test::fixed_width_column_wrapper<double>{{-8., 25., 90.}};
  auto constexpr seed = 42u;

  auto const output     = cudf::hashing::xxhash_32(cudf::table_view({col1}), seed);
  auto const jit_output = cudf::hashing::xxhash_32_jit(cudf::table_view({col1}), seed);

  // Expected results were generated with the reference implementation:
  // https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h
  auto expected =
    cudf::test::fixed_width_column_wrapper<uint32_t>({2276435783u, 3120212431u, 3454197470u});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(jit_output->view(), expected);
}

TEST_F(XXHash_32_Test, StringType)
{
  auto col1           = cudf::test::strings_column_wrapper({"I", "am", "AI"});
  auto constexpr seed = 825u;

  auto output     = cudf::hashing::xxhash_32(cudf::table_view({col1}), seed);
  auto jit_output = cudf::hashing::xxhash_32_jit(cudf::table_view({col1}), seed);

  // Expected results were generated with the reference implementation:
  // https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h
  auto expected =
    cudf::test::fixed_width_column_wrapper<uint32_t>({320624298u, 1612654309u, 1409499009u});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(jit_output->view(), expected);
}

TEST_F(XXHash_32_Test, ZeroColumns)
{
  auto const input      = cudf::table_view{std::vector<cudf::column_view>{}, 5};
  auto const output     = cudf::hashing::xxhash_32(input, 42);
  auto const jit_output = cudf::hashing::xxhash_32_jit(input, 42);
  cudf::test::fixed_width_column_wrapper<uint32_t> const expected({42u, 42u, 42u, 42u, 42u});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(jit_output->view(), expected);
}

TEST_F(XXHash_32_Test, MixedNullableParity)
{
  auto const ints = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, -1, 42, 7}, cudf::test::iterators::null_at(1));
  auto const strings = cudf::test::strings_column_wrapper({"", "one", "two", "three"},
                                                          cudf::test::iterators::null_at(2));
  auto const input   = cudf::table_view({ints, strings});

  for (auto const seed : {0u, 42u, 825u}) {
    auto const expected = cudf::hashing::xxhash_32(input, seed);
    auto const actual   = cudf::hashing::xxhash_32_jit(input, seed);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), actual->view());
  }
}

TEST_F(XXHash_32_Test, NestedParity)
{
  using LCW                = cudf::test::lists_column_wrapper<int32_t>;
  auto const list_validity = std::vector<bool>{true, true, false, true};
  auto const lists         = LCW{{{}, {1}, {2, 3}, {}}, list_validity.begin()};
  auto ints                = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
  auto strings             = cudf::test::strings_column_wrapper{"a", "b", "c", "d"};
  auto const structs =
    cudf::test::structs_column_wrapper{{ints, strings}, cudf::test::iterators::null_at(1)};
  auto const input = cudf::table_view({lists, structs});

  auto const expected = cudf::hashing::xxhash_32(input, 42);
  auto const actual   = cudf::hashing::xxhash_32_jit(input, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), actual->view());
}

TEST_F(XXHash_32_Test, DictionaryDecimalAndChronoParity)
{
  auto strings    = cudf::test::strings_column_wrapper{"alpha", "beta", "alpha", "gamma"};
  auto dictionary = cudf::dictionary::encode(strings);
  auto const decimals =
    cudf::test::fixed_point_column_wrapper<int32_t>({0, 100, -100, 42}, numeric::scale_type{-3});
  using timestamp = cudf::timestamp_s;
  auto const timestamps =
    cudf::test::fixed_width_column_wrapper<timestamp, timestamp::duration>{timestamp::duration{0},
                                                                           timestamp::duration{1},
                                                                           timestamp::duration{-1},
                                                                           timestamp::duration{42}};
  auto const input = cudf::table_view({dictionary->view(), decimals, timestamps});

  auto const expected = cudf::hashing::xxhash_32(input, 7);
  auto const actual   = cudf::hashing::xxhash_32_jit(input, 7);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), actual->view());
}
