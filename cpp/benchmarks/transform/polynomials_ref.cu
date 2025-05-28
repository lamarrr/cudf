

/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <cuda/ptx>
#include <cuda/std/atomic>
#include <cuda/std/cstdint>
#include <thrust/iterator/counting_iterator.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <random>

struct NormalLoad {
  __device__ double operator()(double const* addr) const { return *addr; }
};

struct NormalStore {
  __device__ void operator()(double* addr, double v) const { *addr = v; }
};

struct StreamLoad {
  __device__ double operator()(double const* addr) const
  {
    double v;
    asm volatile("ld.global.nc.f64 %0, [%1];\n\t" : "=d"(v) : "l"(addr));

    return v;
  }
};

struct AsyncStore {
  __device__ void operator()(double* addr, double v) const
  {
    asm("st.async.global.f64 [%0], %1;\n\t" : : "l"(addr), "d"(v));
  }
};

template <typename... P>
__device__ void transform(double* out, double x, double p0, P... p)
{
  ((p0 = (p0 * x + p)), ...);
  *out = p0;
}

template <typename Load, typename Store, typename... P>
__device__ void kernel_impl(int n, double* out, P*... p)
{
  auto i            = (int64_t)threadIdx.x + (int64_t)blockDim.x * (int64_t)blockIdx.x;
  auto const stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

  constexpr Load load;
  constexpr Store store;

  for (; i < n; i += stride) {
    double tmp;
    transform(&tmp, load(p + i)...);
    store(out + i, tmp);
  }
}

template <typename Load, typename Store, typename... P>
__global__ void kernel(int n, double* out, double const* x, double const* p0, P const*... p)
{
  kernel_impl<Load, Store>(n, out, x, p0, p...);
}

template <typename LoadType, typename StoreType>
static void BM_transform_polynomials_ref(nvbench::state& state)
{
  typedef double key_type;
  auto const num_rows{static_cast<cudf::size_type>(state.get_int64("num_rows"))};
  auto const order            = 8;
  auto const null_probability = 0;

  CUDF_EXPECTS(order > 0, "Polynomial order must be greater than 0");

  data_profile profile;
  profile.set_null_probability(null_probability);
  profile.set_distribution_params(cudf::type_to_id<key_type>(),
                                  distribution_id::NORMAL,
                                  static_cast<key_type>(0),
                                  static_cast<key_type>(1));
  auto column = create_random_column(cudf::type_to_id<key_type>(), row_count{num_rows}, profile);

  std::vector<std::unique_ptr<cudf::column>> constants;

  std::transform(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(order + 1),
    std::back_inserter(constants),
    [&](int) { return create_random_column(cudf::type_to_id<key_type>(), row_count{1}, profile); });

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(num_rows);
  state.add_global_memory_writes<key_type>(num_rows);

  std::vector<cudf::column_view> inputs{*column};
  std::transform(constants.begin(),
                 constants.end(),
                 std::back_inserter(inputs),
                 [](auto& col) -> cudf::column_view { return *col; });

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // computes polynomials: (((ax + b)x + c)x + d)x + e... = ax**4 + bx**3 + cx**2 + dx + e....
    auto output = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64},
                                                num_rows,
                                                cudf::mask_state::ALL_VALID,
                                                launch.get_stream().get_stream(),
                                                cudf::get_current_device_resource_ref());

    int grid, block;
    CUDF_EXPECTS(CUresult::CUDA_SUCCESS == cuOccupancyMaxPotentialBlockSizeWithFlags(
                                             &grid, &block, nullptr, nullptr, 0, 0, 0),
                 "");

    kernel<LoadType, StoreType>
      <<<grid, block, 0, launch.get_stream().get_stream()>>>(num_rows,
                                                             output->mutable_view().data<double>(),
                                                             inputs[0].data<double>(),
                                                             inputs[1].data<double>(),
                                                             inputs[2].data<double>(),
                                                             inputs[3].data<double>(),
                                                             inputs[4].data<double>(),
                                                             inputs[5].data<double>(),
                                                             inputs[6].data<double>(),
                                                             inputs[7].data<double>(),
                                                             inputs[8].data<double>());
  });
}

#define TRANSFORM_POLYNOMIALS_REF_BENCHMARK_DEFINE(name, LoadType, StoreType) \
                                                                              \
  static void name(::nvbench::state& st)                                      \
  {                                                                           \
    ::BM_transform_polynomials_ref<LoadType, StoreType>(st);                  \
  }                                                                           \
  NVBENCH_BENCH(name).set_name(#name).add_int64_axis(                         \
    "num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})

TRANSFORM_POLYNOMIALS_REF_BENCHMARK_DEFINE(transform_polynomials_float64_ref,
                                           NormalLoad,
                                           NormalStore);
TRANSFORM_POLYNOMIALS_REF_BENCHMARK_DEFINE(transform_polynomials_float64_ref_async_store,
                                           NormalLoad,
                                           NormalStore);
TRANSFORM_POLYNOMIALS_REF_BENCHMARK_DEFINE(transform_polynomials_float64_ref_load_nc_async_store,
                                           StreamLoad,
                                           AsyncStore);