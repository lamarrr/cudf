/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

struct ColumnDeviceViewTest : public cudf::test::BaseFixture {};

TEST_F(ColumnDeviceViewTest, Sample)
{
  using T = int32_t;
  cuda::stream_ref stream{cudf::get_default_stream()};
  cudf::test::fixed_width_column_wrapper<T> input({1, 2, 3, 4, 5, 6});
  auto output            = cudf::allocate_like(input);
  auto input_device_view = cudf::column_device_view::create(input, stream);
  auto output_device_view =
    cudf::mutable_column_device_view::create(output->mutable_view(), stream);

  EXPECT_NO_THROW(thrust::copy(rmm::exec_policy_nosync(stream),
                               input_device_view->begin<T>(),
                               input_device_view->end<T>(),
                               output_device_view->begin<T>()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, output->view());
}

TEST_F(ColumnDeviceViewTest, MismatchingType)
{
  using T = int32_t;
  cuda::stream_ref stream{cudf::get_default_stream()};
  cudf::test::fixed_width_column_wrapper<T> input({1, 2, 3, 4, 5, 6});
  auto output            = cudf::allocate_like(input);
  auto input_device_view = cudf::column_device_view::create(input, stream);
  auto output_device_view =
    cudf::mutable_column_device_view::create(output->mutable_view(), stream);

  EXPECT_THROW(thrust::copy(rmm::exec_policy_nosync(stream),
                            input_device_view->begin<T>(),
                            input_device_view->end<T>(),
                            output_device_view->begin<int64_t>()),
               cudf::logic_error);
}

TEST_F(ColumnDeviceViewTest, ExplicitMemoryResourceControl)
{
  auto harness = cudf::test::memory_resource_test_harness{this->mr()};
  auto stream  = cudf::get_default_stream();
  auto input   = cudf::test::strings_column_wrapper({"one", "two"}).release();

  auto immutable_view = [&] {
    auto current_scope = harness.fail_on_current_device_resource_use();
    auto result = cudf::column_device_view::create(input->view(), stream, harness.output_mr());
    harness.synchronize(stream);
    return result;
  }();
  auto mutable_view = [&] {
    auto current_scope = harness.fail_on_current_device_resource_use();
    auto result =
      cudf::mutable_column_device_view::create(input->mutable_view(), stream, harness.output_mr());
    harness.synchronize(stream);
    return result;
  }();

  harness.expect_output_allocations_live(stream);
  immutable_view.reset();
  mutable_view.reset();
  harness.expect_no_live_allocations(stream);
}

namespace {

static_assert(cudf::is_list_element<cudf::list_element>);
static_assert(!cudf::is_list_element<int32_t>);
static_assert(cudf::is_mutable_list_element<cudf::mutable_list_element>);
static_assert(!cudf::is_list_element<cudf::mutable_list_element>);
static_assert(!cudf::is_mutable_list_element<cudf::list_element>);
static_assert(!cudf::column_device_view::has_element_accessor<cudf::mutable_list_element>());
static_assert(cudf::mutable_column_device_view::has_element_accessor<cudf::mutable_list_element>());
static_assert(cudf::mutable_column_device_view::has_element_accessor<cudf::list_element>());
static_assert(!cuda::std::is_convertible_v<cudf::list_element, cudf::mutable_list_element>);

template <typename T>
struct nullable_list_metadata {
  cudf::column_device_view input;
  cudf::mutable_column_device_view output;

  __device__ bool operator()(cudf::size_type i) const
  {
    auto in  = input.nullable_element<cudf::list_element>(i);
    auto out = output.nullable_element<cudf::mutable_list_element>(i);
    if (i == 1) { return !in && !out; }
    if (!in || !out) { return false; }
    auto input_row  = input.element<cudf::list_element>(i);
    auto output_row = output.element<cudf::mutable_list_element>(i);
    auto value      = in->template nullable_element<T>(0);
    if (!value || in->template nullable_element<T>(1) || out->template nullable_element<T>(1)) {
      return false;
    }
    if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
      if (*value != cudf::string_view{"hello", 5}) { return false; }
      auto writable = out->template element<cudf::mutable_string_view>(0);
      out->assign(0, writable);
      if (writable.data() != value->data() || writable.size_bytes() != 5 ||
          !out->template nullable_element<cudf::mutable_string_view>(0)) {
        return false;
      }
    } else if constexpr (cudf::is_fixed_point<T>()) {
      if (value->value() != 3 || value->scale() != numeric::scale_type{-2}) { return false; }
      out->assign(0, T{numeric::scaled_integer<typename T::rep>{7, numeric::scale_type{-1}}});
      auto assigned = out->template element<T>(0);
      if (assigned.value() != 7 || assigned.scale() != numeric::scale_type{-2}) { return false; }
      out->assign(0, *value);
    } else {
      if (*value != 3) { return false; }
      out->assign(0, T{7});
      if (out->template element<T>(0) != 7) { return false; }
      out->assign(0, *value);
    }
    out->set_null(0);
    if (in->template nullable_element<T>(0)) { return false; }
    out->set_valid(0);
    out->set_valid(1);
    if (!in->template nullable_element<T>(0) || !in->template nullable_element<T>(1)) {
      return false;
    }
    out->set_null(1);
    return in->size() == 2 && out->size() == 2 && !in->empty() && !out->empty() && in->nullable() &&
           out->nullable() && in->is_valid(0) && out->is_valid(0) && in->is_null(1) &&
           out->is_null(1) && input_row.size() == in->size() && output_row.size() == out->size();
  }
};

struct nonnullable_list_metadata {
  cudf::column_device_view input;
  cudf::mutable_column_device_view output;

  __device__ bool operator()(cudf::size_type i) const
  {
    auto in  = input.nullable_element<cudf::list_element>(i);
    auto out = output.nullable_element<cudf::mutable_list_element>(i);
    if (!in || !out) { return false; }
    return in->size() == i && out->size() == i && in->empty() == (i == 0) &&
           out->empty() == (i == 0) && !in->nullable() && !out->nullable() &&
           (i == 0 || (in->is_valid(0) && out->is_valid(0)));
  }
};

template <typename T>
struct ListElementMetadataTest : cudf::test::BaseFixture {};
using ListChildTypes = ::testing::
  Types<int32_t, numeric::decimal32, numeric::decimal64, numeric::decimal128, cudf::string_view>;
TYPED_TEST_SUITE(ListElementMetadataTest, ListChildTypes);

TYPED_TEST(ListElementMetadataTest, NullableRowsAndChildrenInSlicedViews)
{
  using T    = TypeParam;
  auto child = [] {
    if constexpr (cudf::is_fixed_point<T>()) {
      return cudf::test::fixed_point_column_wrapper<typename T::rep>{
        {99, 3, 0}, {true, true, false}, numeric::scale_type{-2}}
        .release();
    } else if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::test::strings_column_wrapper{{"skip", "hello", "unused"}, {true, true, false}}
        .release();
    } else {
      return cudf::test::fixed_width_column_wrapper<T>{{99, 3, 0}, {true, true, false}}.release();
    }
  }();
  auto parent_mask =
    cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 0}, {true, true, false}}.release();
  auto lists = cudf::make_lists_column(
    3,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 3, 3}.release(),
    std::move(child),
    1,
    std::move(*parent_mask->release().null_mask));
  auto stream = cudf::get_default_stream();
  auto input = cudf::column_device_view::create(cudf::slice(lists->view(), {1, 3}).front(), stream);
  auto view  = lists->mutable_view();
  auto mutable_slice = cudf::mutable_column_view{
    view.type(), 2, view.head(), view.null_mask(), 1, 1, {view.child(0), view.child(1)}};
  auto output = cudf::mutable_column_device_view::create(mutable_slice, stream);
  cudf::test::fixed_width_column_wrapper<bool> expected{true, true};
  auto result = cudf::allocate_like(expected);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<cudf::size_type>{0},
                    thrust::counting_iterator<cudf::size_type>{2},
                    result->mutable_view().data<bool>(),
                    nullable_list_metadata<T>{*input, *output});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(ColumnDeviceViewTest, EmptyAndNonnullableListRows)
{
  cudf::test::lists_column_wrapper<int32_t> input{{}, {7}};
  auto lists  = input.release();
  auto stream = cudf::get_default_stream();
  auto in     = cudf::column_device_view::create(lists->view(), stream);
  auto out    = cudf::mutable_column_device_view::create(lists->mutable_view(), stream);
  cudf::test::fixed_width_column_wrapper<bool> expected{true, true};
  auto result = cudf::allocate_like(expected);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<cudf::size_type>{0},
                    thrust::counting_iterator<cudf::size_type>{2},
                    result->mutable_view().data<bool>(),
                    nonnullable_list_metadata{*in, *out});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

}  // namespace
