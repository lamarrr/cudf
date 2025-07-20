
/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>

#include <numeric>

namespace CUDF_EXPORT cudf {
namespace ast::detail {
/**
 * @brief Base "visitor" pattern class with the `expression` class for expression transformer.
 *
 * This class can be used to implement recursive traversal of AST tree, and used to validate or
 * translate an AST expression.
 */
class expression_transformer {
 public:
  /**
   * @brief Visit a literal expression.
   *
   * @param expr Literal expression
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> visit(literal const& expr) = 0;

  /**
   * @brief Visit a column reference expression.
   *
   * @param expr Column reference expression
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> visit(column_reference const& expr) = 0;

  /**
   * @brief Visit an expression expression
   *
   * @param expr Expression expression
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> visit(operation const& expr) = 0;

  /**
   * @brief Visit a column name reference expression.
   *
   * @param expr Column name reference expression
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> visit(column_name_reference const& expr) = 0;

  virtual ~expression_transformer() {}
};

struct udf_transformer {
  struct resolve_info {
    std::span<cudf::data_type const> left_table_types;
    std::map<std::string, cudf::data_type> left_table_named_types;
    std::span<cudf::data_type const> right_table_types;
    std::map<std::string, cudf::data_type> right_table_named_types;
  };

  struct arguments {
   
    
  };

  struct cuda_info {
    std::string identifier;
    std::string expression;
    std::optional<cudf::data_type> type;
  };

  // [ ] how to get column
  // [ ] needs to resolve input columns and literals

  struct node {
    std::string identifier_;
    cudf::data_type type_ = cudf::data_type{cudf::type_id::EMPTY};

    virtual void resolve_type(resolve_info const& info) = 0;

    virtual cudf::data_type get_type() const { return type_; }

    virtual cuda_info process_cuda(arguments&) = 0;
  };

  struct column_ref : node {
    std::uint32_t input_column_index_;
    std::uint32_t input_table_index_;

    void resolve_type(resolve_info const& info) override
    {
      if (input_table_index_ == 0) {
        type_ = info.left_table_types[input_column_index_];
      } else if (input_table_index_ == 1) {
        type_ = info.right_table_types[input_column_index_];
      } else {
        CUDF_FAIL("Invalid input table index for column reference.");
      }
    }

   cuda_info process_cuda(arguments & scope)   override {  

    scope.add_column_ref();

    }
  };

  struct column_name_ref : node {
    std::string input_column_name_;
    std::uint32_t input_table_index_;

    void resolve_type(resolve_info const& info) override
    {
      if (input_table_index_ == 0) {
        type_ = info.left_table_named_types.at(input_column_name_);
      } else if (input_table_index_ == 1) {
        type_ = info.right_table_named_types.at(input_column_name_);
      } else {
        CUDF_FAIL("Invalid input table index for column name reference.");
      }
    }

    std::optional<std::string> get_cuda_expression() const override { return identifier_; }
  };

  struct literal : node {};

  struct operation : node {
    cudf::ast::ast_operator op_;
    std::vector<node*> operands_;

    void resolve_type(resolve_info const& info) override
    {
      std::vector<cudf::data_type> operand_types;
      for (auto& operand : operands_) {
        operand->resolve_type(info);
        operand_types.emplace_back(operand->get_type());
      }
      type_ = ast_operator_return_type(op_, operand_types);
    }

    std::optional<std::string> get_cuda_expression() const override
    {
      auto arguments = std::accumulate(
        operands_.begin(), operands_.end(), std::string{}, [](auto const& a, auto const& b) {
          return std::format("{}, {}", a, b->get_identifier());
        });

      return std::format("{}({})", ast_operator_to_string(op_), arguments);
    }
  };

  node* output_ = nullptr;
  std::vector<std::unique_ptr<node>> nodes_;

  void resolve(resolve_info const& info) {
    // [ ] resolve column references
    // [ ] resolve column names
    // [ ] resolve literals
    // [ ] resolve operations
    // [ ] resolve output
  };

  struct intermediate {
    std::string identifier;
    std::string expression;
    std::optional<cudf::data_type> type;
  };

  struct input_column {
    std::string identifier;
    std::variant<column_reference, column_name_reference> source;
  };

  struct input_literal {
    std::string identifier;
    literal value;
  };

  struct output {
    std::string identifier;
    std::string expression;
    std::optional<cudf::data_type> type;
  };

  std::vector<input_column> columns_;
  std::vector<input_literal> literals_;
  std::vector<intermediate> intermediates_;
  std::optional<output> output_;

  std::size_t num_columns() const { return columns_.size(); }
  std::size_t num_literals() const { return literals_.size(); }
  std::size_t num_intermediates() const { return intermediates_.size(); }
  std::size_t num_outputs() const { return output_.has_value() ? 1 : 0; }

  // [ ] how to handle literals

  std::tuple<std::string, data_type> add_reference(literal const& lit)
  {
    auto identifier = literal_prefix_ + std::to_string(literals_.size());
    literals_.emplace_back(identifier, lit);
    return {identifier, lit.get_data_type()};
  }

  std::tuple<std::string, data_type> add_reference(column_reference const& column)
  {
    auto identifier = column_prefix_ + std::to_string(columns_.size());
    columns_.emplace_back(identifier, column);
    return {identifier, column.get_data_type()};
  }

  std::tuple<std::string, data_type> add_reference(column_name_reference const& column)
  {
    auto identifier = column_prefix_ + std::to_string(columns_.size());
    columns_.emplace_back(identifier, column);
    return {identifier, column.get_data_type()};
  }

  std::tuple<std::string, data_type> add_intermediate(operation const& op)
  {
    auto identifier = intermediate_prefix_ + std::to_string(intermediates_.size());
    std::vector<std::string> operands_cuda;
    std::vector<cudf::data_type> operand_types;
    for (auto& operand : op.get_operands()) {
      auto [operand_cuda, operand_type] = operand.get().accept(*this);
      operands_cuda.emplace_back(operand_cuda);
      operand_types.emplace_back(operand_type);
    }
    auto [expression, type] = get_cuda_expression(op.get_operator(), operands_cuda, operand_types);
    intermediates_.emplace_back(identifier, expression, type);
    return {identifier, type};
  }

  std::tuple<std::string, data_type> add_cuda_intermediate(std::string_view expression,
                                                           cudf::data_type type)
  {
    auto identifier = intermediate_prefix_ + std::to_string(intermediates_.size());
    intermediates_.emplace_back(identifier, std::string(expression), type);
    return {identifier, type};
  }

  void set_output(cudf::data_type type, std::string_view intermediate)
  {
    output_ =
      output{output_prefix_ + std::to_string(num_outputs()), std::string(intermediate), type};
  }

  std::string generate_udf();
  // std::span<cudf::column_view const> get_inputs(cudf::table_view const& left,
  // cudf::table_view const& right) const

  std::tuple<std::string, data_type> visit(literal const& expr) { return add_reference(expr); }

  std::tuple<std::string, data_type> visit(column_reference const& expr)
  {
    return add_reference(expr);
  }

  std::tuple<std::string, data_type> visit(operation const& expr) { return add_intermediate(expr); }

  std::tuple<std::string, data_type> visit(column_name_reference const& expr)
  {
    return add_reference(expr);
  }
};

}  // namespace ast::detail

}  // namespace CUDF_EXPORT cudf
