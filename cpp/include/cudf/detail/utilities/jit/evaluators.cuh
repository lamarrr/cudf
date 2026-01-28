/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/utilities/jit/jit_bit_utilities.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/tuple>

namespace cudf::detail::jit::evaluators {

template <typename T>
using optional = cuda::std::optional<T>;

using Scope    = void* __restrict__* const __restrict__;
using UserData = void* __restrict__;

template <int Index, typename T, bool Nullable>
struct column_element_getter {
  using Arg = column_device_view_core const* __restrict__;

  __device__ static T get(Scope scope, size_type i)
    requires(!Nullable)
  {
    auto p = static_cast<Arg>(scope[Index]);
    return p->element<T>(i);
  }

  __device__ static optional<T> get(Scope scope, size_type i)
    requires(Nullable)
  {
    auto p = static_cast<Arg>(scope[Index]);
    return p->nullable_element<T>(i);
  }
};

template <int Index, typename T, bool Nullable>
struct span_element_getter {
  using Arg = jit::device_optional_span<T> const* __restrict__;

  __device__ static T get(Scope scope, size_type i)
    requires(!Nullable)
  {
    auto p = static_cast<Arg>(scope[Index]);
    return p->element(i);
  }

  __device__ static optional<T> get(Scope scope, size_type i)
    requires(Nullable)
  {
    auto p = static_cast<Arg>(scope[Index]);
    if (p->is_valid(i)) {
      return p->element(i);
    } else {
      return cuda::std::nullopt;
    }
  }
};

template <int Index,
          typename T,
          bool Nullable,
          bool IsFixedPoint,
          bool OutputBoolMask,
          int BoolMaskIndex>
struct column_element_setter {
  using Arg = mutable_column_device_view_core const* __restrict__;

  __device__ static T output_arg(Scope scope)
    requires(!IsFixedPoint && !Nullable)
  {
    return T{};
  }

  __device__ static optional<T> output_arg(Scope scope)
    requires(!IsFixedPoint && Nullable)
  {
    return cuda::std::nullopt;
  }

  __device__ static T output_arg(Scope scope)
    requires(IsFixedPoint && !Nullable)
  {
    auto p     = static_cast<Arg>(scope[Index]);
    auto scale = static_cast<numeric::scale_type>(p->type().scale());
    T out{numeric::scaled_integer<typename T::rep>{0, scale}};
    return out;
  }

  __device__ static optional<T> output_arg(Scope scope)
    requires(IsFixedPoint && Nullable)
  {
    auto p     = static_cast<Arg>(scope[Index]);
    auto scale = static_cast<numeric::scale_type>(p->type().scale());
    T out{numeric::scaled_integer<typename T::rep>{0, scale}};
    return out;
  }

  __device__ static void set(Scope scope, size_type i, T const& value)
    requires(!Nullable)
  {
    auto p = static_cast<Arg>(scope[Index]);
    p->assign(i, value);
  }

  __device__ static void set(Scope scope, size_type i, optional<T> const& value)
    requires(Nullable)
  {
    auto p = static_cast<Arg>(scope[Index]);
    p->assign(i, *value);

    if constexpr (OutputBoolMask) {
      auto null_mask = static_cast<bool*>(scope[BoolMaskIndex]);
      null_mask[i]   = value.has_value();
    }
  }
};

template <int Index, typename T, bool Nullable, bool OutputBoolMask, int BoolMaskIndex>
struct span_element_setter {
  using Arg = jit::device_optional_span<T> const* __restrict__;

  __device__ static T output_arg(Scope scope)
    requires(!Nullable)
  {
    return T{};
  }

  __device__ static optional<T> output_arg(Scope scope)
    requires(Nullable)
  {
    return cuda::std::nullopt;
  }

  __device__ static void set(Scope scope, size_type i, T const& value)
    requires(!Nullable)
  {
    auto p = static_cast<Arg>(scope[Index]);
    p->assign(i, value);
  }

  __device__ static void set(Scope scope, size_type i, optional<T> const& value)
    requires(Nullable)
  {
    auto p = static_cast<Arg>(scope[Index]);
    p->assign(i, *value);

    if constexpr (OutputBoolMask) {
      auto null_mask = static_cast<bool*>(scope[BoolMaskIndex]);
      null_mask[i]   = value.has_value();
    }
  }
};

// TODO: how will expression evaluation work to handle multiple outputs? we might need to use manual string concatenation of c++ source code for this
/// sub-expression handling
template <bool IsNullAware, int UserDataIndex, typename InputGetters, typename OutputSetters>
struct element_transform_operation {
  template <typename Operator>
  __device__ static void evaluate(Scope scope, size_type i)
  {
    auto output_args = OutputSetters::output_args(scope);
    auto input_args  = InputGetters::get(scope, i);

    if constexpr (IsNullAware) {
      auto user_data      = static_cast<UserData>(scope[UserDataIndex]);
      auto user_data_args = cuda::std::tuple{user_data, i};
      auto args           = cuda::std::tuple_cat(user_data_args, output_args, input_args);
      cuda::std::apply(GENERIC_TRANSFORM_OP, args);
    } else {
      auto args = cuda::std::tuple_cat(output_args, input_args);
      cuda::std::apply(GENERIC_TRANSFORM_OP, args);
    }

    OutputSetters::set(scope, i, output_args);
  }
};

// TODO: make work with multiple inputs and outputs
template <int OutputIndex, int OutputValidCountIndex, int... InputColumnIndices>
struct and_null_mask_evaluator {
  __device__ static void evaluate(Scope scope, size_type size)
  {
    auto get_column_bitmask = __device__[&]<int Column>()
    {
      auto p = static_cast<column_device_view_core const* __restrict__>(scope[Column]);
      return p->null_mask();
    };

    auto get_column_offset = __device__[&]<int Column>()
    {
      auto p = static_cast<column_device_view_core const* __restrict__>(scope[Column]);
      return p->offset();
    };

    bitmask_type const* __restrict__ srcs[sizeof...(InputColumnIndices)] = {
      get_column_bitmask<InputColumnIndices>()...};
    size_type offsets[sizeof...(InputColumnIndices)] = {get_column_offset<InputColumnIndices>()...};

    nullmask_and_subkernel(srcs,
                           offsets,
                           size,
                           static_cast<bitmask_type* __restrict__>(scope[OutputIndex]),
                           static_cast<size_type* __restrict__>(scope[OutputValidCountIndex]));
  }
};

template <int OutputIndex, bool AllValid>
struct fill_null_mask_evaluator {
  __device__ static void evaluate(Scope scope, size_type size)
  {
    auto dst = static_cast<bitmask_type* __restrict__>(scope[OutputIndex]);
    detail::jit::bit_utilities::fill_subkernel(size, dst, AllValid);
  }
};

template <int SrcIndex, int DstIndex>
struct copy_null_mask_evaluator {
  __device__ static void evaluate(Scope scope, size_type size)
  {
    auto src = static_cast<bitmask_type const* __restrict__>(scope[SrcIndex]);
    auto dst = static_cast<bitmask_type* __restrict__>(scope[DstIndex]);
    detail::jit::bit_utilities::copy_subkernel(src, size, dst);
  }
};

template <int OutputIndex, int BoolsNullMaskIndex>
struct bools_to_null_mask_evaluator {
  __device__ static void evaluate(Scope scope, size_type size)
  {
    auto src = static_cast<bool const* __restrict__>(scope[BoolsNullMaskIndex]);
    auto dst = static_cast<bitmask_type* __restrict__>(scope[OutputIndex]);
    detail::jit::bit_utilities::boolean_mask_to_nullmask_subkernel(src, size, dst, nullptr);
  }
};

}  // namespace cudf::detail::jit::evaluators
