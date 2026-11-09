/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/scalar/scalar_column_device_view.cuh>
#include <cudf/scalar/scalar_column_factories.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream.hpp>

#include <type_traits>

using ScalarColumnTest = cudf::test::BaseFixture;
namespace factories    = cudf::scalar_column_factories;
static_assert(std::is_trivially_copyable_v<cudf::scalar_column_device_view>);
static_assert(std::is_trivially_copyable_v<cudf::mutable_scalar_column_device_view>);

TEST_F(ScalarColumnTest, OwnershipAndOffsets)
{
  cudf::test::fixed_width_column_wrapper<int> input{1, 2, 3};
  auto sliced = cudf::slice(input, {1, 2})[0];
  auto view   = cudf::scalar_column_view{sliced};
  EXPECT_EQ(view.offset(), 1);
  EXPECT_EQ(view.head<int>(), sliced.head<int>());
  cudf::scalar_column copy{view};
  EXPECT_EQ(copy.view().offset(), 0);
  EXPECT_NE(copy.view().head<int>(), view.head<int>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(copy.view().as_column_view(), sliced);
  auto ptr = copy.view().head<int>();
  cudf::scalar_column moved{copy.release()};
  EXPECT_EQ(ptr, moved.view().head<int>());
  EXPECT_THROW(cudf::scalar_column_view{input}, std::logic_error);
  EXPECT_THROW(cudf::scalar_column{std::unique_ptr<cudf::column>{}}, std::logic_error);
}

TEST_F(ScalarColumnTest, Factories)
{
  auto number = factories::make_fixed_width_scalar<int>(42);
  cudf::test::fixed_width_column_wrapper<int> expected{42};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(number->view().as_column_view(), expected);
  auto invalid = factories::make_numeric_scalar(cudf::data_type{cudf::type_id::INT32});
  EXPECT_EQ(invalid->null_count(), 1);
  EXPECT_THROW(factories::make_numeric_scalar(cudf::data_type{cudf::type_id::STRING}),
               cudf::logic_error);
  auto string = factories::make_string_scalar("hello");
  cudf::test::strings_column_wrapper expected_string{"hello"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(string->view().as_column_view(), expected_string);
  auto empty = factories::make_string_scalar("");
  EXPECT_FALSE(empty->has_nulls());
  auto null_string =
    factories::make_default_constructed_scalar(cudf::data_type{cudf::type_id::STRING});
  EXPECT_TRUE(null_string->has_nulls());
  auto list = factories::make_list_scalar(expected);
  EXPECT_EQ(list->view().num_children(), 2);
  auto null_list = factories::make_empty_scalar_like(list->view().as_column_view());
  EXPECT_TRUE(null_list->has_nulls());
  auto structure = factories::make_struct_scalar(cudf::table_view{{expected}});
  EXPECT_EQ(structure->view().num_children(), 1);
  auto null_struct = factories::make_empty_scalar_like(structure->view().as_column_view());
  EXPECT_TRUE(null_struct->has_nulls());
}

__global__ void write_scalar(cudf::mutable_scalar_column_device_view view)
{
  view.element<int>() = 17;
  view.set_valid();
}
__global__ void check_scalar(cudf::scalar_column_device_view view, bool* result)
{
  *result = view.is_valid() && !view.is_null() && view.element<int>() == 17;
}
TEST_F(ScalarColumnTest, DeviceMutation)
{
  auto value        = factories::make_numeric_scalar(cudf::data_type{cudf::type_id::INT32});
  auto mutable_view = cudf::mutable_scalar_column_device_view::create(value->mutable_view());
  write_scalar<<<1, 1, 0, cudf::get_default_stream().get()>>>(*mutable_view);
  value->set_null_count(0);
  auto view = cudf::scalar_column_device_view::create(value->view());
  cudf::detail::device_scalar<bool> result{cudf::get_default_stream()};
  check_scalar<<<1, 1, 0, cudf::get_default_stream().get()>>>(*view, result.data());
  EXPECT_TRUE(result.value(cudf::get_default_stream()));
}

TEST_F(ScalarColumnTest, DecimalChronoAndExplicitStream)
{
  rmm::cuda_stream stream;
  auto mr      = cudf::get_current_device_resource_ref();
  auto decimal = factories::make_fixed_point_scalar<numeric::decimal64>(
    123, numeric::scale_type{-2}, stream.view(), mr);
  EXPECT_EQ(decimal->type(), (cudf::data_type{cudf::type_id::DECIMAL64, -2}));
  auto timestamp = factories::make_timestamp_scalar(
    cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, stream.view(), mr);
  auto duration = factories::make_duration_scalar(
    cudf::data_type{cudf::type_id::DURATION_SECONDS}, stream.view(), mr);
  EXPECT_TRUE(timestamp->has_nulls());
  EXPECT_TRUE(duration->has_nulls());
  cudf::scalar_column copy{decimal->view(), stream.view(), mr};
  auto view = cudf::scalar_column_device_view::create(copy.view(), stream.view(), mr);
  stream.synchronize();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(copy.view().as_column_view(), decimal->view().as_column_view());
}

TEST_F(ScalarColumnTest, SlicedDeviceMutation)
{
  cudf::test::fixed_width_column_wrapper<int> input({1, 2, 3}, {true, false, true});
  auto owner = input.release();
  auto base  = owner->mutable_view();
  auto sliced =
    cudf::mutable_column_view{base.type(), 1, base.head<void>(), base.null_mask(), 1, 1};
  auto mutable_view =
    cudf::mutable_scalar_column_device_view::create(cudf::mutable_scalar_column_view{sliced});
  write_scalar<<<1, 1, 0, cudf::get_default_stream().get()>>>(*mutable_view);
  owner->set_null_count(0);
  cudf::test::fixed_width_column_wrapper<int> expected{1, 17, 3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(owner->view(), expected);
}

__global__ void check_string(cudf::scalar_column_device_view view, bool* result)
{
  *result = !view.is_null() && view.element<cudf::string_view>().size_bytes() == 5;
}
TEST_F(ScalarColumnTest, NestedDeviceMetadata)
{
  auto value = factories::make_string_scalar("hello");
  auto view  = cudf::scalar_column_device_view::create(value->view());
  cudf::detail::device_scalar<bool> result{cudf::get_default_stream()};
  check_string<<<1, 1, 0, cudf::get_default_stream().get()>>>(*view, result.data());
  EXPECT_TRUE(result.value(cudf::get_default_stream()));
}
