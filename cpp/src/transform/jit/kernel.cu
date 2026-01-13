/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/detail/operator_functor.cuh>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/cstddef>

#include <jit/accessors.cuh>
#include <jit/span.cuh>

// clang-format off
// This header is an inlined header that defines the GENERIC_FILTER_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/operation-udf.hpp>
// clang-format on

namespace cudf {
namespace transformation {
namespace jit {

using Scope = void* __restrict__* const __restrict__;

template <int Index, typename T>
struct column_element_getter {
  using Arg = column_device_view_core const* __restrict__;

  __device__ static T get(Scope scope, size_type i)
  {
    auto p = static_cast<Arg>(scope[Index]);
    return p->element<T>(i);
  }
};

template <int Index, typename T>
struct optional_column_element_getter {
  using Arg = column_device_view_core const* __restrict__;

  __device__ static cuda::std::optional<T> get(Scope scope, size_type i)
  {
    auto p = static_cast<Arg>(scope[Index]);
    return p->nullable_element<T>(i);
  }
};

template <int Index, typename T>
struct span_element_getter {
  using Arg = jit::device_optional_span<T> const* __restrict__;

  __device__ static T get(Scope scope, size_type i)
  {
    auto p = static_cast<Arg>(scope[Index]);
    return p->element(i);
  }
};

template <int Index, typename T>
struct span_optional_element_getter {
  using Arg = jit::device_optional_span<T> const* __restrict__;

  __device__ static cuda::std::optional<T> get(Scope scope, size_type i)
  {
    auto p = static_cast<Arg>(scope[Index]);
    if (p->is_valid(i)) {
      return p->element(i);
    } else {
      return cuda::std::nullopt;
    }
  }
};

template <int Index, typename T>
struct column_element_assigner {
  using Arg = mutable_column_device_view_core const* __restrict__;

  __device__ static T output_arg(Scope scope) { return T{}; }

  __device__ static void assign(Scope scope, size_type i, T const& value)
  {
    auto p = static_cast<Arg>(scope[Index]);
    p->assign(i, value);
  }
};

template <int Index, typename T>
struct fixed_point_column_element_assigner {
  using Arg = mutable_column_device_view_core const* __restrict__;

  __device__ static T output_arg(Scope scope)
  {
    auto p     = static_cast<Arg>(scope[Index]);
    auto scale = static_cast<numeric::scale_type>(p->type().scale());
    T out{numeric::scaled_integer<typename T::rep>{0, scale}};
    return out;
  }

  __device__ static void assign(Scope scope, size_type i, T const& value)
  {
    auto p = static_cast<Arg>(scope[Index]);
    p->assign(i, value.rep());
  }
};

template <int Index, typename T>
struct span_element_assigner {
  using Arg = jit::device_optional_span<T> const* __restrict__;

  __device__ static T output_arg(Scope scope)
  {
    auto p = static_cast<Arg>(scope[Index]);
    return p->element<T>(0);
  }

  __device__ static void assign(Scope scope, size_type i, T const& value)
  {
    auto p = static_cast<Arg>(scope[Index]);
    p->assign(i, value);
  }
};

// [ ] scope variables should be aligned to avoid uncoalesced reads/writes
template <int Index, typename T>
struct span_optional_element_assigner {
  using Arg = jit::device_optional_span<T> const* __restrict__;

  __device__ static T output_arg(Scope scope) { return {}; }

  __device__ static void assign(Scope scope, size_type i, T const& value)
  {
    auto p = static_cast<Arg>(scope[Index]);
    p->assign(i, value);
  }
};

// TODO: how will expression evaluation work to handle multiple outputs?
///
template <int NumInputs,
          int NumOutputs,
          int UserDataIndex,
          typename InputGetters,
          typename OutputSetters>
struct element_operation {
  template <typename Operator>
  __device__ static void evaluate(Scope scope, cudf::size_type i, Operator&& op)
  {
    if constexpr (UserDataIndex >= 0) {
      auto output_args;
      GENERIC_TRANSFORM_OP(user_data, i, &res, In::element(inputs, i)...);
    } else {
      GENERIC_TRANSFORM_OP(&res, In::element(inputs, i)...);
    }
  }
};

// TODO: make work with multiple inputs and outputs
template <int NumInputs, int NumOutputs, typename InputGetter, typename OutputSetter>
struct null_mask_evaluator {};

template <null_aware is_null_aware,
          bool may_evaluate_null,
          bool has_user_data,
          typename Out,
          typename... In>
CUDF_KERNEL void kernel(void const* __restrict__ outputs,
                        bool** __restrict__ intermediate_null_masks,
                        cudf::size_type* __restrict__ output_valid_counts,
                        void const* __restrict__ inputs,
                        void* __restrict__ user_data)
{
  // inputs to JITIFY kernels have to be either sized-integral types or pointers. Structs or
  // references can't be passed directly/correctly as they will be crossing an ABI boundary

  auto const start                   = cudf::detail::grid_1d::global_thread_id();
  auto const stride                  = cudf::detail::grid_1d::grid_stride();
  auto const size                    = outputs[0].size();
  cudf::size_type thread_valid_count = 0;

  for (auto i = start; i < size; i += stride) {
    if constexpr (is_null_aware == null_aware::NO) {
      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &Out::element(outputs, i), In::element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&Out::element(outputs, i), In::element(inputs, i)...);
      }

    } else {  // is_null_aware == null_aware::YES
      cuda::std::optional<typename Out::type> result;

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &result, In::nullable_element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&result, In::nullable_element(inputs, i)...);
      }

      Out::assign(outputs, i, *result);

      if constexpr (may_evaluate_null) { intermediate_null_masks[0][i] = result.has_value(); }
    }
  }

  if constexpr (may_evaluate_null) {
    // __syncthreads();
    // transform_bitmask_subkernel();
    if constexpr (is_null_aware == null_aware::NO) {
    } else {
    }
  }
}

template <null_aware is_null_aware,
          bool may_evaluate_null,
          bool has_user_data,
          typename Out,
          typename... In>
CUDF_KERNEL void fixed_point_kernel(cudf::mutable_column_device_view_core const* outputs,
                                    cudf::column_device_view_core const* inputs,
                                    bool* intermediate_null_mask,
                                    void* user_data)
{
  auto const start        = cudf::detail::grid_1d::global_thread_id();
  auto const stride       = cudf::detail::grid_1d::grid_stride();
  auto const size         = outputs[0].size();
  auto const output_scale = static_cast<numeric::scale_type>(outputs[0].type().scale());

  for (auto i = start; i < size; i += stride) {
    if constexpr (is_null_aware == null_aware::NO) {
      typename Out::type result{numeric::scaled_integer<typename Out::type::rep>{0, output_scale}};

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &result, In::element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&result, In::element(inputs, i)...);
      }

      Out::assign(outputs, i, result);

    } else {  // is_null_aware == null_aware::YES
      cuda::std::optional<typename Out::type> result{
        typename Out::type{numeric::scaled_integer<typename Out::type::rep>{0, output_scale}}};

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &result, In::nullable_element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&result, In::nullable_element(inputs, i)...);
      }

      Out::assign(outputs, i, *result);

      if constexpr (may_evaluate_null) { intermediate_null_mask[i] = result.has_value(); }
    }
  }
}

// TODO: combine all kernels into one, use a sub-kernel approach if necessary,
// add additional span parameters or make the inputs and outputs untyped.
// or just have a single expansion tparam that dispatches the correct operations
template <null_aware is_null_aware,
          bool may_evaluate_null,
          bool has_user_data,
          typename Out,
          typename... In>
CUDF_KERNEL void span_kernel(cudf::jit::device_optional_span<typename Out::type> const* outputs,
                             cudf::column_device_view_core const* inputs,
                             bool* intermediate_null_mask,
                             void* user_data)
{
  auto const start  = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto const size   = outputs[0].size();

  for (auto i = start; i < size; i += stride) {
    if constexpr (is_null_aware == null_aware::NO) {
      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &Out::element(outputs, i), In::element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&Out::element(outputs, i), In::element(inputs, i)...);
      }
    } else {  // is_null_aware == null_aware::YES
      cuda::std::optional<typename Out::type> result;

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &result, In::nullable_element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&result, In::nullable_element(inputs, i)...);
      }

      Out::assign(outputs, i, *result);

      if constexpr (may_evaluate_null) { intermediate_null_mask[i] = result.has_value(); }
    }
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
