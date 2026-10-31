/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Compare TPC-DS Q13's LTO aggregation with existing cuDF APIs on resident synthetic data.
 *
 * Dimension joins are represented by resolved demographic/address branch codes before timing.
 * Predicates are evaluated on every invocation: the JIT path fuses them into its filter, and
 * AOT uses compute_column followed by apply_retention_mask and reduce.
 * AOT results remain scalars.
 * Compilation, AST construction, and data generation are excluded from timing.
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <cudf_tpc_reduction_fragments.hpp>
#include <nvbench/nvbench.cuh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using cudf::data_type;
using cudf::size_type;
using cudf::type_id;
using columns = std::vector<std::unique_ptr<cudf::column>>;
using scalars = std::vector<std::unique_ptr<cudf::scalar>>;

cudf::lto_fragments precompiled_fragments()
{
  auto const range =
    cudf_tpc_reduction_fragments::file_ranges[cudf_tpc_reduction_fragments::tpcds_q13];
  static cudf::lto_fragment const result[]{
    {cudf::lto_binary_type::FATBIN,
     cudf_tpc_reduction_fragments::files.subspan(range[0], range[1])}};
  return result;
}

cudf::reduce_udf make_udf()
{
  return {precompiled_fragments(),
          "tpcds_q13_init",
          "tpcds_q13_row_update",
          "tpcds_q13_merge",
          "tpcds_q13_finalize",
          "tpcds_q13_filter"};
}

columns make_inputs(size_type rows, int32_t null_percent)
{
  columns input;

  auto add = [&](type_id type, double lower, double upper, bool nullable) {
    auto profile = data_profile_builder();
    profile.distribution(type, distribution_id::UNIFORM, lower, upper);
    if (nullable && null_percent != 0) {
      profile.null_probability(null_percent / 100.0);
    } else {
      profile.no_validity();
    }
    auto const seed = static_cast<unsigned>(input.size() + 1);
    input.push_back(create_random_column(type, row_count{rows}, profile, seed));
  };

  add(type_id::INT32, 1, 50, true);        // ss_quantity
  add(type_id::FLOAT64, 0, 1000, true);    // ss_ext_sales_price
  add(type_id::FLOAT64, 0, 800, true);     // ss_ext_wholesale_cost
  add(type_id::FLOAT64, 0, 200, false);    // ss_sales_price
  add(type_id::FLOAT64, 0, 300, false);    // ss_net_profit
  add(type_id::INT32, 0, 3, false);        // resolved demographic branch
  add(type_id::INT32, 0, 3, false);        // resolved address branch
  add(type_id::INT32, 1998, 2002, false);  // d_year

  return input;
}

cudf::ast::tree make_predicate(scalars& literals, cuda::stream_ref stream)
{
  namespace ast = cudf::ast;
  using op      = ast::ast_operator;
  ast::tree tree;

  auto literal = [&]<typename T>(T value) -> ast::literal const& {
    auto scalar        = std::make_unique<cudf::numeric_scalar<T>>(value, true, stream);
    auto const& result = tree.emplace<ast::literal>(*scalar);
    literals.push_back(std::move(scalar));
    return result;
  };

  auto branch = [&](ast::expression const& code,
                    int32_t expected,
                    ast::expression const& value,
                    double lower,
                    double upper) -> ast::operation const& {
    auto const& matches = tree.emplace<ast::operation>(op::EQUAL, code, literal(expected));
    auto const& minimum = tree.emplace<ast::operation>(op::GREATER_EQUAL, value, literal(lower));
    auto const& maximum = tree.emplace<ast::operation>(op::LESS_EQUAL, value, literal(upper));
    auto const& range   = tree.emplace<ast::operation>(op::LOGICAL_AND, minimum, maximum);
    return tree.emplace<ast::operation>(op::LOGICAL_AND, matches, range);
  };

  auto const& sales_price  = tree.emplace<ast::column_reference>(3);
  auto const& net_profit   = tree.emplace<ast::column_reference>(4);
  auto const& demographic  = tree.emplace<ast::column_reference>(5);
  auto const& address      = tree.emplace<ast::column_reference>(6);
  auto const& demographics = tree.emplace<ast::operation>(
    op::LOGICAL_OR,
    tree.emplace<ast::operation>(op::LOGICAL_OR,
                                 branch(demographic, 1, sales_price, 100, 150),
                                 branch(demographic, 2, sales_price, 50, 100)),
    branch(demographic, 3, sales_price, 150, 200));

  auto const& addresses = tree.emplace<ast::operation>(
    op::LOGICAL_OR,
    tree.emplace<ast::operation>(op::LOGICAL_OR,
                                 branch(address, 1, net_profit, 100, 200),
                                 branch(address, 2, net_profit, 150, 300)),
    branch(address, 3, net_profit, 50, 250));

  auto const& year = tree.emplace<ast::operation>(
    op::EQUAL, tree.emplace<ast::column_reference>(7), literal(int32_t{2001}));

  auto const& branches = tree.emplace<ast::operation>(op::LOGICAL_AND, demographics, addresses);
  tree.emplace<ast::operation>(op::LOGICAL_AND, branches, year);

  return tree;
}

scalars reduce_aot(cudf::table_view values,
                   cudf::table_view all_inputs,
                   cudf::ast::expression const& predicate,
                   cudf::reduce_aggregation const& sum,
                   cudf::reduce_aggregation const& mean,
                   cuda::stream_ref stream)
{
  auto selection             = cudf::compute_column(all_inputs, predicate, stream);
  auto selected              = cudf::apply_retention_mask(values, selection->view(), stream);
  auto const selected_values = selected->view();
  data_type const result_type{type_id::FLOAT64};
  scalars result;

  for (size_type column = 0; column < selected_values.num_columns(); ++column) {
    result.push_back(cudf::reduce(selected_values.column(column), mean, result_type, stream));
  }
  result.push_back(cudf::reduce(selected_values.column(2), sum, result_type, stream));

  return result;
}

void tpcds_q13(nvbench::state& state)
{
  constexpr size_t value_input_count = 3;

  data_type const state_types[]{data_type{type_id::FLOAT64},
                                data_type{type_id::FLOAT64},
                                data_type{type_id::FLOAT64},
                                data_type{type_id::INT64},
                                data_type{type_id::INT64},
                                data_type{type_id::INT64}};

  data_type const output_types[]{data_type{type_id::FLOAT64},
                                 data_type{type_id::FLOAT64},
                                 data_type{type_id::FLOAT64},
                                 data_type{type_id::FLOAT64}};

  auto const rows   = static_cast<size_type>(state.get_int64("rows"));
  auto const nulls  = static_cast<int32_t>(state.get_int64("null_percent"));
  auto const stream = cudf::get_default_stream();
  auto input        = make_inputs(rows, nulls);

  std::vector<cudf::reduce_input> arguments;
  std::vector<cudf::column_view> all_views;

  for (auto const& column : input) {
    arguments.emplace_back(column->view());
    all_views.push_back(column->view());
    state.add_global_memory_reads(column->alloc_size());
  }

  cudf::table_view const all_inputs{all_views};

  std::vector<cudf::column_view> value_views(all_views.begin(),
                                             all_views.begin() + value_input_count);

  cudf::table_view const values{value_views};

  // AST literals borrow their scalars; keep the storage alive for every benchmark invocation.
  scalars literals;

  auto predicate = make_predicate(literals, stream);

  auto sum  = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto mean = cudf::make_mean_aggregation<cudf::reduce_aggregation>();

  auto const udf = make_udf();

  auto jit = [&] {
    return cudf::reduce_lto(
      udf, cudf::null_aware::YES, arguments, state_types, output_types, stream);
  };

  auto aot = [&] { return reduce_aot(values, all_inputs, predicate.back(), *sum, *mean, stream); };

  bool const use_jit = state.get_string("engine") == "jit";
  auto execute       = [&] {
    if (use_jit) {
      jit();
    } else {
      aot();
    }
  };

  // Warm the selected implementation so compilation is excluded from timing.
  execute();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_element_count(rows);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { execute(); });
}

}  // namespace

NVBENCH_BENCH(tpcds_q13)
  .set_name("tpcds_q13_aggregates")
  .add_string_axis("engine", {"jit", "aot"})
  .add_int64_axis("rows", {0, 1'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("null_percent", {0, 10, 100});
