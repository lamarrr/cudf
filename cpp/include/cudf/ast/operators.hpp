/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cmath>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <simt/type_traits>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {

namespace ast {

enum class ast_operator {
  // Binary operators
  ADD,            ///< operator +
  SUB,            ///< operator -
  MUL,            ///< operator *
  DIV,            ///< operator / using common type of lhs and rhs
  TRUE_DIV,       ///< operator / after promoting type to floating point
  FLOOR_DIV,      ///< operator / after promoting to 64 bit floating point and then
                  ///< flooring the result
  MOD,            ///< operator %
  PYMOD,          ///< operator % but following python's sign rules for negatives
  POW,            ///< lhs ^ rhs
  EQUAL,          ///< operator ==
  NOT_EQUAL,      ///< operator !=
  LESS,           ///< operator <
  GREATER,        ///< operator >
  LESS_EQUAL,     ///< operator <=
  GREATER_EQUAL,  ///< operator >=
  BITWISE_AND,    ///< operator &
  BITWISE_OR,     ///< operator |
  BITWISE_XOR,    ///< operator ^
  LOGICAL_AND,    ///< operator &&
  LOGICAL_OR,     ///< operator ||
  /*
  COALESCE,       ///< operator x,y  x is null ? y : x
  // GENERIC_BINARY,        ///< generic binary operator to be generated with input
  //                       ///< ptx code
  SHIFT_LEFT,            ///< operator <<
  SHIFT_RIGHT,           ///< operator >>
  SHIFT_RIGHT_UNSIGNED,  ///< operator >>> (from Java)
                         ///< Logical right shift. Casts to an unsigned value before shifting.
  LOG_BASE,              ///< logarithm to the base
  ATAN2,                 ///< 2-argument arctangent
  PMOD,                  ///< positive modulo operator
                         ///< If remainder is negative, this returns (remainder + divisor) % divisor
                         ///< else, it returns (dividend % divisor)
  NULL_EQUALS,           ///< Returns true when both operands are null; false when one is null; the
                         ///< result of equality when both are non-null
  NULL_MAX,              ///< Returns max of operands when both are non-null; returns the non-null
                         ///< operand when one is null; or invalid when both are null
  NULL_MIN,              ///< Returns min of operands when both are non-null; returns the non-null
                         ///< operand when one is null; or invalid when both are null
  */
  // Unary operators
  SIN,         ///< Trigonometric sine
  COS,         ///< Trigonometric cosine
  TAN,         ///< Trigonometric tangent
  ARCSIN,      ///< Trigonometric sine inverse
  ARCCOS,      ///< Trigonometric cosine inverse
  ARCTAN,      ///< Trigonometric tangent inverse
  SINH,        ///< Hyperbolic sine
  COSH,        ///< Hyperbolic cosine
  TANH,        ///< Hyperbolic tangent
  ARCSINH,     ///< Hyperbolic sine inverse
  ARCCOSH,     ///< Hyperbolic cosine inverse
  ARCTANH,     ///< Hyperbolic tangent inverse
  EXP,         ///< Exponential (base e, Euler number)
  LOG,         ///< Natural Logarithm (base e)
  SQRT,        ///< Square-root (x^0.5)
  CBRT,        ///< Cube-root (x^(1.0/3))
  CEIL,        ///< Smallest integer value not less than arg
  FLOOR,       ///< largest integer value not greater than arg
  ABS,         ///< Absolute value
  RINT,        ///< Rounds the floating-point argument arg to an integer value
  BIT_INVERT,  ///< Bitwise Not (~)
  NOT,         ///< Logical Not (!)
  // Other operators included in BlazingSQL (TODO: may or may not implement)
  /*
  IS_NULL,            ///< Unary comparator returning whether the value is null
  COMPONENT_YEAR,     ///< Get year from a timestamp
  COMPONENT_MONTH,    ///< Get month from a timestamp
  COMPONENT_DAY,      ///< Get day from a timestamp
  COMPONENT_WEEKDAY,  ///< Get weekday from a timestamp
  COMPONENT_HOUR,     ///< Get hour from a timestamp
  COMPONENT_MINUTE,   ///< Get minute from a timestamp
  COMPONENT_SECOND,   ///< Get second from a timestamp
  ROUND,              ///< Round a value to a desired precision
  IS_NOT_NULL,        ///< Unary comparator returning whether the value is not null
  COTAN,              ///< Trigonometric cotangent
  CAST,               ///< Type cast operator (TODO: special case)
  CHAR_LENGTH,        ///< String length
  RAND,               ///< Random number (nullary operator)
  NOW,                ///< Current timestamp
  ROW,                ///< Current row of the table
  THREAD_ID,          ///< Could be useful for debugging
  BLOCK_ID            ///< Could be useful for debugging
  */
};

// Traits for valid operator / type combinations
template <typename Op, typename LHS, typename RHS>
constexpr bool is_valid_binary_op = simt::std::is_invocable<Op, LHS, RHS>::value;

template <typename Op, typename T>
constexpr bool is_valid_unary_op = simt::std::is_invocable<Op, T>::value;

// Operator dispatcher
template <typename F, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr void ast_operator_dispatcher(ast_operator op,
                                                                 F&& f,
                                                                 Ts&&... args)
{
  switch (op) {
    case ast_operator::ADD:
      f.template operator()<ast_operator::ADD>(std::forward<Ts>(args)...);
      break;
    case ast_operator::SUB:
      f.template operator()<ast_operator::SUB>(std::forward<Ts>(args)...);
      break;
    case ast_operator::MUL:
      f.template operator()<ast_operator::MUL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::DIV:
      f.template operator()<ast_operator::DIV>(std::forward<Ts>(args)...);
      break;
    case ast_operator::TRUE_DIV:
      f.template operator()<ast_operator::TRUE_DIV>(std::forward<Ts>(args)...);
      break;
    case ast_operator::FLOOR_DIV:
      f.template operator()<ast_operator::FLOOR_DIV>(std::forward<Ts>(args)...);
      break;
    case ast_operator::MOD:
      f.template operator()<ast_operator::MOD>(std::forward<Ts>(args)...);
      break;
    case ast_operator::PYMOD:
      f.template operator()<ast_operator::PYMOD>(std::forward<Ts>(args)...);
      break;
    case ast_operator::POW:
      f.template operator()<ast_operator::POW>(std::forward<Ts>(args)...);
      break;
    case ast_operator::EQUAL:
      f.template operator()<ast_operator::EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::NOT_EQUAL:
      f.template operator()<ast_operator::NOT_EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LESS:
      f.template operator()<ast_operator::LESS>(std::forward<Ts>(args)...);
      break;
    case ast_operator::GREATER:
      f.template operator()<ast_operator::GREATER>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LESS_EQUAL:
      f.template operator()<ast_operator::LESS_EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::GREATER_EQUAL:
      f.template operator()<ast_operator::GREATER_EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::BITWISE_AND:
      f.template operator()<ast_operator::BITWISE_AND>(std::forward<Ts>(args)...);
      break;
    case ast_operator::BITWISE_OR:
      f.template operator()<ast_operator::BITWISE_OR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::BITWISE_XOR:
      f.template operator()<ast_operator::BITWISE_XOR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LOGICAL_AND:
      f.template operator()<ast_operator::LOGICAL_AND>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LOGICAL_OR:
      f.template operator()<ast_operator::LOGICAL_OR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::SIN:
      f.template operator()<ast_operator::SIN>(std::forward<Ts>(args)...);
      break;
    case ast_operator::COS:
      f.template operator()<ast_operator::COS>(std::forward<Ts>(args)...);
      break;
    case ast_operator::TAN:
      f.template operator()<ast_operator::TAN>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCSIN:
      f.template operator()<ast_operator::ARCSIN>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCCOS:
      f.template operator()<ast_operator::ARCCOS>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCTAN:
      f.template operator()<ast_operator::ARCTAN>(std::forward<Ts>(args)...);
      break;
    case ast_operator::SINH:
      f.template operator()<ast_operator::SINH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::COSH:
      f.template operator()<ast_operator::COSH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::TANH:
      f.template operator()<ast_operator::TANH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCSINH:
      f.template operator()<ast_operator::ARCSINH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCCOSH:
      f.template operator()<ast_operator::ARCCOSH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCTANH:
      f.template operator()<ast_operator::ARCTANH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::EXP:
      f.template operator()<ast_operator::EXP>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LOG:
      f.template operator()<ast_operator::LOG>(std::forward<Ts>(args)...);
      break;
    case ast_operator::SQRT:
      f.template operator()<ast_operator::SQRT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::CBRT:
      f.template operator()<ast_operator::CBRT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::CEIL:
      f.template operator()<ast_operator::CEIL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::FLOOR:
      f.template operator()<ast_operator::FLOOR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ABS:
      f.template operator()<ast_operator::ABS>(std::forward<Ts>(args)...);
      break;
    case ast_operator::RINT:
      f.template operator()<ast_operator::RINT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::BIT_INVERT:
      f.template operator()<ast_operator::BIT_INVERT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::NOT:
      f.template operator()<ast_operator::NOT>(std::forward<Ts>(args)...);
      break;
    default:
      // TODO: Error handling?
      break;
  }
}

template <ast_operator op>
struct operator_functor {
};

template <>
struct operator_functor<ast_operator::ADD> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs + rhs)
  {
    return lhs + rhs;
  }
};

template <>
struct operator_functor<ast_operator::SUB> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs - rhs)
  {
    return lhs - rhs;
  }
};

template <>
struct operator_functor<ast_operator::MUL> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs * rhs)
  {
    return lhs * rhs;
  }
};

template <>
struct operator_functor<ast_operator::DIV> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs / rhs)
  {
    return lhs / rhs;
  }
};

template <>
struct operator_functor<ast_operator::TRUE_DIV> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs)
    -> decltype(static_cast<double>(lhs) / static_cast<double>(rhs))
  {
    return static_cast<double>(lhs) / static_cast<double>(rhs);
  }
};

template <>
struct operator_functor<ast_operator::FLOOR_DIV> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs)
    -> decltype(floor(static_cast<double>(lhs) / static_cast<double>(rhs)))
  {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

template <>
struct operator_functor<ast_operator::MOD> {
  static constexpr auto arity{2};

  template <typename LHS,
            typename RHS,
            typename CommonType                                    = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_integral<CommonType>::value>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs)
    -> decltype(static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs))
  {
    return static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs);
  }

  template <typename LHS,
            typename RHS,
            typename CommonType = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_same<CommonType, float>::value>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)))
  {
    return fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs));
  }

  template <typename LHS,
            typename RHS,
            typename CommonType = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_same<CommonType, double>::value>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)))
  {
    return fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs));
  }
};

template <>
struct operator_functor<ast_operator::PYMOD> {
  static constexpr auto arity{2};

  template <typename LHS,
            typename RHS,
            typename CommonType                                    = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_integral<CommonType>::value>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs)
    -> decltype(((static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs)) +
                 static_cast<CommonType>(rhs)) %
                static_cast<CommonType>(rhs))
  {
    return ((static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs)) +
            static_cast<CommonType>(rhs)) %
           static_cast<CommonType>(rhs);
  }

  template <typename LHS,
            typename RHS,
            typename CommonType = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_same<CommonType, float>::value>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmodf(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                        static_cast<CommonType>(rhs),
                      static_cast<CommonType>(rhs)))
  {
    return fmodf(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                   static_cast<CommonType>(rhs),
                 static_cast<CommonType>(rhs));
  }

  template <typename LHS,
            typename RHS,
            typename CommonType = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_same<CommonType, double>::value>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmod(fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                       static_cast<CommonType>(rhs),
                     static_cast<CommonType>(rhs)))
  {
    return fmod(fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                  static_cast<CommonType>(rhs),
                static_cast<CommonType>(rhs));
  }
};

template <>
struct operator_functor<ast_operator::POW> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(std::pow(lhs, rhs))
  {
    return std::pow(lhs, rhs);
  }
};

template <>
struct operator_functor<ast_operator::EQUAL> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs == rhs)
  {
    return lhs == rhs;
  }
};

template <>
struct operator_functor<ast_operator::NOT_EQUAL> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs != rhs)
  {
    return lhs != rhs;
  }
};

template <>
struct operator_functor<ast_operator::LESS> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs < rhs)
  {
    return lhs < rhs;
  }
};

template <>
struct operator_functor<ast_operator::GREATER> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs > rhs)
  {
    return lhs > rhs;
  }
};

template <>
struct operator_functor<ast_operator::LESS_EQUAL> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs <= rhs)
  {
    return lhs <= rhs;
  }
};

template <>
struct operator_functor<ast_operator::GREATER_EQUAL> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs >= rhs)
  {
    return lhs >= rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_AND> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs & rhs)
  {
    return lhs & rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_OR> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs | rhs)
  {
    return lhs | rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_XOR> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs ^ rhs)
  {
    return lhs ^ rhs;
  }
};

template <>
struct operator_functor<ast_operator::LOGICAL_AND> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs && rhs)
  {
    return lhs && rhs;
  }
};

template <>
struct operator_functor<ast_operator::LOGICAL_OR> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs || rhs)
  {
    return lhs || rhs;
  }
};

template <>
struct operator_functor<ast_operator::SIN> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::sin(input))
  {
    return std::sin(input);
  }
};

template <>
struct operator_functor<ast_operator::COS> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::cos(input))
  {
    return std::cos(input);
  }
};

template <>
struct operator_functor<ast_operator::TAN> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::tan(input))
  {
    return std::tan(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCSIN> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::asin(input))
  {
    return std::asin(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCCOS> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::acos(input))
  {
    return std::acos(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCTAN> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::atan(input))
  {
    return std::atan(input);
  }
};

template <>
struct operator_functor<ast_operator::SINH> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::sinh(input))
  {
    return std::sinh(input);
  }
};

template <>
struct operator_functor<ast_operator::COSH> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::cosh(input))
  {
    return std::cosh(input);
  }
};

template <>
struct operator_functor<ast_operator::TANH> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::tanh(input))
  {
    return std::tanh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCSINH> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::asinh(input))
  {
    return std::asinh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCCOSH> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::acosh(input))
  {
    return std::acosh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCTANH> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::atanh(input))
  {
    return std::atanh(input);
  }
};

template <>
struct operator_functor<ast_operator::EXP> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::exp(input))
  {
    return std::exp(input);
  }
};

template <>
struct operator_functor<ast_operator::LOG> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::log(input))
  {
    return std::log(input);
  }
};

template <>
struct operator_functor<ast_operator::SQRT> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::sqrt(input))
  {
    return std::sqrt(input);
  }
};

template <>
struct operator_functor<ast_operator::CBRT> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::cbrt(input))
  {
    return std::cbrt(input);
  }
};

template <>
struct operator_functor<ast_operator::CEIL> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::ceil(input))
  {
    return std::ceil(input);
  }
};

template <>
struct operator_functor<ast_operator::FLOOR> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::floor(input))
  {
    return std::floor(input);
  }
};

template <>
struct operator_functor<ast_operator::ABS> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::abs(input))
  {
    return std::abs(input);
  }
};

template <>
struct operator_functor<ast_operator::RINT> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(std::rint(input))
  {
    return std::rint(input);
  }
};

template <>
struct operator_functor<ast_operator::BIT_INVERT> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(~input)
  {
    return ~input;
  }
};

template <>
struct operator_functor<ast_operator::NOT> {
  static constexpr auto arity{1};

  template <typename InputT>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(InputT input) -> decltype(!input)
  {
    return !input;
  }
};

namespace detail {

template <typename OperatorFunctor>
struct dispatch_unary_operator_types {
  template <typename InputT,
            typename F,
            typename... Ts,
            std::enable_if_t<is_valid_unary_op<OperatorFunctor, InputT>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, InputT>(std::forward<Ts>(args)...);
  }

  template <typename InputT,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_unary_op<OperatorFunctor, InputT>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid unary operation.");
#else
    release_assert(false && "Invalid unary operation.");
#endif
  }
};

template <typename OperatorFunctor>
struct double_dispatch_binary_operator_types {
  template <typename LHS,
            typename RHS,
            typename F,
            typename... Ts,
            std::enable_if_t<is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, LHS, RHS>(std::forward<Ts>(args)...);
  }

  template <typename LHS,
            typename RHS,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation.");
#else
    release_assert(false && "Invalid binary operation.");
#endif
  }
};

template <typename OperatorFunctor>
struct single_dispatch_binary_operator_types {
  template <typename LHS,
            typename F,
            typename... Ts,
            std::enable_if_t<is_valid_binary_op<OperatorFunctor, LHS, LHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, LHS, LHS>(std::forward<Ts>(args)...);
  }

  template <typename LHS,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, LHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation.");
#else
    release_assert(false && "Invalid binary operation.");
#endif
  }
};

struct type_dispatch_unary_op {
  template <ast_operator op, typename F, typename... Ts>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type input_t, F&& f, Ts&&... args)
  {
    type_dispatcher(input_t,
                    detail::dispatch_unary_operator_types<operator_functor<op>>{},
                    std::forward<F>(f),
                    std::forward<Ts>(args)...);
  }
};

struct type_dispatch_binary_op {
  template <ast_operator op, typename F, typename... Ts>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type lhs_t,
                                            cudf::data_type rhs_t,
                                            F&& f,
                                            Ts&&... args)
  {
    // Double dispatch
    /*
    double_type_dispatcher(lhs_t,
                           rhs_t,
                           detail::double_dispatch_binary_operator_types<operator_functor<op>>{},
                           std::forward<F>(f),
                           std::forward<Ts>(args)...);
    */
    // Single dispatch (assume lhs_t == rhs_t)
    type_dispatcher(lhs_t,
                    detail::single_dispatch_binary_operator_types<operator_functor<op>>{},
                    std::forward<F>(f),
                    std::forward<Ts>(args)...);
  }
};

struct binary_return_type_functor {
  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out = simt::std::invoke_result_t<OperatorFunctor, LHS, RHS>,
            std::enable_if_t<cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type* result)
  {
    *result = cudf::data_type(cudf::type_to_id<Out>());
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out                                                                 = void,
            std::enable_if_t<!cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type* result)
  {
    *result = cudf::data_type(cudf::type_id::EMPTY);
  }
};

struct unary_return_type_functor {
  template <typename OperatorFunctor,
            typename T,
            typename Out = simt::std::invoke_result_t<OperatorFunctor, T>,
            std::enable_if_t<cudf::ast::is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type* result)
  {
    *result = cudf::data_type(cudf::type_to_id<Out>());
  }

  template <typename OperatorFunctor,
            typename T,
            typename Out                                                         = void,
            std::enable_if_t<!cudf::ast::is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type* result)
  {
    *result = cudf::data_type(cudf::type_id::EMPTY);
  }
};

struct arity_functor {
  template <typename OperatorFunctor>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::size_type* result)
  {
    *result = OperatorFunctor::arity;
  }
};

}  // namespace detail

/**
 * @brief Dispatches a single type template parameters to a unary operator.
 *
 */
template <typename F, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr void unary_operator_dispatcher(ast_operator op,
                                                                   cudf::data_type input_t,
                                                                   F&& f,
                                                                   Ts&&... args)
{
  ast_operator_dispatcher(
    op, detail::type_dispatch_unary_op{}, input_t, std::forward<F>(f), std::forward<Ts>(args)...);
}

/**
 * @brief Dispatches two type template parameters to a binary operator.
 *
 */
template <typename F, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr void binary_operator_dispatcher(
  ast_operator op, cudf::data_type lhs_t, cudf::data_type rhs_t, F&& f, Ts&&... args)
{
  ast_operator_dispatcher(op,
                          detail::type_dispatch_binary_op{},
                          lhs_t,
                          rhs_t,
                          std::forward<F>(f),
                          std::forward<Ts>(args)...);
}

/**
 * @brief Gets the return type of an AST operator.
 *
 */
inline cudf::data_type ast_operator_return_type(ast_operator op,
                                                std::vector<cudf::data_type> const& operand_types)
{
  auto result = cudf::data_type(cudf::type_id::EMPTY);
  switch (operand_types.size()) {
    case 0:
      // TODO: Nullary return type functor
      break;
    case 1:
      unary_operator_dispatcher(
        op, operand_types.at(0), detail::unary_return_type_functor{}, &result);
      break;
    case 2:
      binary_operator_dispatcher(op,
                                 operand_types.at(0),
                                 operand_types.at(1),
                                 detail::binary_return_type_functor{},
                                 &result);
      break;
    case 3:
      // TODO: Ternary return type functor
      break;
    default: break;
  }
  return result;
}

/**
 * @brief Gets the arity of an AST operator.
 *
 */
CUDA_HOST_DEVICE_CALLABLE cudf::size_type ast_operator_arity(ast_operator op)
{
  switch (op) {
    case ast_operator::ADD: return operator_functor<ast_operator::ADD>::arity;
    case ast_operator::SUB: return operator_functor<ast_operator::SUB>::arity;
    case ast_operator::MUL: return operator_functor<ast_operator::MUL>::arity;
    case ast_operator::DIV: return operator_functor<ast_operator::DIV>::arity;
    case ast_operator::TRUE_DIV: return operator_functor<ast_operator::TRUE_DIV>::arity;
    case ast_operator::FLOOR_DIV: return operator_functor<ast_operator::FLOOR_DIV>::arity;
    case ast_operator::MOD: return operator_functor<ast_operator::MOD>::arity;
    case ast_operator::PYMOD: return operator_functor<ast_operator::PYMOD>::arity;
    case ast_operator::POW: return operator_functor<ast_operator::POW>::arity;
    case ast_operator::EQUAL: return operator_functor<ast_operator::EQUAL>::arity;
    case ast_operator::NOT_EQUAL: return operator_functor<ast_operator::NOT_EQUAL>::arity;
    case ast_operator::LESS: return operator_functor<ast_operator::LESS>::arity;
    case ast_operator::GREATER: return operator_functor<ast_operator::GREATER>::arity;
    case ast_operator::LESS_EQUAL: return operator_functor<ast_operator::LESS_EQUAL>::arity;
    case ast_operator::GREATER_EQUAL: return operator_functor<ast_operator::GREATER_EQUAL>::arity;
    case ast_operator::BITWISE_AND: return operator_functor<ast_operator::BITWISE_AND>::arity;
    case ast_operator::BITWISE_OR: return operator_functor<ast_operator::BITWISE_OR>::arity;
    case ast_operator::BITWISE_XOR: return operator_functor<ast_operator::BITWISE_XOR>::arity;
    case ast_operator::LOGICAL_AND: return operator_functor<ast_operator::LOGICAL_AND>::arity;
    case ast_operator::LOGICAL_OR: return operator_functor<ast_operator::LOGICAL_OR>::arity;
    case ast_operator::SIN: return operator_functor<ast_operator::SIN>::arity;
    case ast_operator::COS: return operator_functor<ast_operator::COS>::arity;
    case ast_operator::TAN: return operator_functor<ast_operator::TAN>::arity;
    case ast_operator::ARCSIN: return operator_functor<ast_operator::ARCSIN>::arity;
    case ast_operator::ARCCOS: return operator_functor<ast_operator::ARCCOS>::arity;
    case ast_operator::ARCTAN: return operator_functor<ast_operator::ARCTAN>::arity;
    case ast_operator::SINH: return operator_functor<ast_operator::SINH>::arity;
    case ast_operator::COSH: return operator_functor<ast_operator::COSH>::arity;
    case ast_operator::TANH: return operator_functor<ast_operator::TANH>::arity;
    case ast_operator::ARCSINH: return operator_functor<ast_operator::ARCSINH>::arity;
    case ast_operator::ARCCOSH: return operator_functor<ast_operator::ARCCOSH>::arity;
    case ast_operator::ARCTANH: return operator_functor<ast_operator::ARCTANH>::arity;
    case ast_operator::EXP: return operator_functor<ast_operator::EXP>::arity;
    case ast_operator::LOG: return operator_functor<ast_operator::LOG>::arity;
    case ast_operator::SQRT: return operator_functor<ast_operator::SQRT>::arity;
    case ast_operator::CBRT: return operator_functor<ast_operator::CBRT>::arity;
    case ast_operator::CEIL: return operator_functor<ast_operator::CEIL>::arity;
    case ast_operator::FLOOR: return operator_functor<ast_operator::FLOOR>::arity;
    case ast_operator::ABS: return operator_functor<ast_operator::ABS>::arity;
    case ast_operator::RINT: return operator_functor<ast_operator::RINT>::arity;
    case ast_operator::BIT_INVERT: return operator_functor<ast_operator::BIT_INVERT>::arity;
    case ast_operator::NOT: return operator_functor<ast_operator::NOT>::arity;
    default: return 0;
  }
}

}  // namespace ast

}  // namespace cudf
