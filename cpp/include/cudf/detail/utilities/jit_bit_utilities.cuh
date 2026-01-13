/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <cuda/std/algorithm>

namespace cudf::detail::jit::bit_utilities {

/// @brief Load a bitmask word chunk from a source bitmask that may not be word-aligned.
/// @param chunk The output chunk index to load/compute
/// @param src_null_mask A functor that computes the bitmask word at a given word index
/// @param src_size The size in bits of the source bitmask
/// @param dst_word Pointer to output the loaded bitmask word
/// @return The number of valid (set) bits in the output word
template <typename BitMaskWordFunc>
__device__ cudf::size_type get_bitmask_word(cudf::size_type chunk,
                                            BitMaskWordFunc&& src_null_mask,
                                            cudf::size_type src_size,
                                            cudf::bitmask_type* dst_word)
{
  // important checks for __funnelshift_r and __popc
  static_assert(sizeof(cudf::bitmask_type) == 4, "bitmask_type must be 32 bits");

  constexpr auto num_chunk_bits = static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);

  auto chunk_bit_begin     = chunk * static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);
  auto src_bit_begin       = chunk_bit_begin;
  auto num_src_words       = (src_size + (num_chunk_bits - 1)) / num_chunk_bits;
  auto leading_word_index  = src_bit_begin / num_chunk_bits;
  auto trailing_word_index = (src_bit_begin + (num_chunk_bits - 1)) / num_chunk_bits;

  if (trailing_word_index < num_src_words) [[likely]] {
    auto leading_bits  = src_null_mask(leading_word_index);
    auto trailing_bits = src_null_mask(trailing_word_index);
    auto bit_shift     = src_bit_begin % num_chunk_bits;
    auto merged      = (cudf::bitmask_type)__funnelshift_r(leading_bits, trailing_bits, bit_shift);
    auto valid_count = (cudf::size_type)__popc(merged);
    *dst_word        = merged;
    return valid_count;
  } else {
    auto leading_bits     = src_null_mask(leading_word_index);
    auto bit_shift        = src_bit_begin % num_chunk_bits;
    auto num_discard_bits = (src_bit_begin + num_chunk_bits) - src_size;
    auto mask             = (~cudf::bitmask_type{0}) >> num_discard_bits;
    auto output           = (leading_bits >> bit_shift) & mask;
    auto valid_count      = (cudf::size_type)__popc(output);
    *dst_word             = output;
    return valid_count;
  }
}

/// @param total Pointer to global memory to accumulate the total. Must be initialized to zero.
/// @param thread_total The per-thread total to add to the global total.
__device__ void sum_counts_subkernel(cudf::size_type* total, cudf::size_type thread_total)
{
  __syncwarp();

  cudf::size_type warp_total = thread_total;

  for (int num_warp_sums = 16; num_warp_sums > 0; num_warp_sums /= 2) {
    warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, num_warp_sums);
  }

  if (threadIdx.x == 0) { atomicAdd(total, warp_total); }
}

/// @brief Compute the null bitmask of chunks (word-sized) from a boolean source array.
/// @param src The source boolean array
/// @param word_chunk_start The starting word chunk index to process
/// @param num_word_chunks The number of word chunks to process
/// @param dst_word The output bitmask word
/// @return The number of valid (set) bits written to the null mask
__device__ cudf::size_type bools_to_bits_chunk(bool const* __restrict__ src,
                                               cudf::size_type src_size,
                                               cudf::size_type word_chunk,
                                               cudf::bitmask_type* __restrict__ dst_word)
{
  static constexpr auto num_word_bits =
    static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);

  auto bit_start = word_chunk * num_word_bits;
  auto bit_end   = cuda::std::min(bit_start + num_word_bits, src_size);

  cudf::bitmask_type out_word = 0;
  for (auto b = bit_start; b < bit_end; b++) {
    auto bit_pos = (b % num_word_bits);
    auto bits    = (src[b] ? cudf::bitmask_type{1} : cudf::bitmask_type{0}) << bit_pos;
    out_word |= bits;
  }

  *dst_word = out_word;

  return __popc(out_word);
}

// TODO: needs to take care of scalars, all valid, all null, etc.
// maybe have a separate evaluator for the null masks
template <int NumSrcs>
__device__ void nullmask_and_subkernel(cudf::bitmask_type* __restrict__ (&srcs)[NumSrcs],
                                       cudf::size_type const (&bitmask_offsets)[NumSrcs],
                                       cudf::size_type src_size,
                                       cudf::bitmask_type* __restrict__ dst,
                                       cudf::size_type* __restrict__ valid_count)
{
  constexpr auto num_word_bits       = static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);
  auto num_chunks                    = (src_size + (num_word_bits - 1)) / num_word_bits;
  cudf::size_type thread_valid_count = 0;

  auto iterator = [srcs, bitmask_offsets] __device__(cudf::size_type word) {
    cudf::bitmask_type out = ~cudf::bitmask_type{0};

#pragma unroll
    for (cudf::size_type src = 0; src < NumSrcs; src++) {
      out &= srcs[src][word + bitmask_offsets[src] / num_word_bits];
    }

    return out;
  };

  auto i = static_cast<int64_t>(threadIdx.x) +
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  auto stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

  for (; i < num_chunks; i += stride) {
    thread_valid_count += get_bitmask_word(i, iterator, src_size, &dst[i]);
  }

  sum_counts_subkernel(valid_count, thread_valid_count);
}

__device__ void boolean_mask_to_nullmask_subkernel(bool const* __restrict__ src,
                                                   cudf::size_type src_size,
                                                   cudf::bitmask_type* __restrict__ dst,
                                                   cudf::size_type* __restrict__ valid_count)
{
  constexpr auto num_word_bits       = static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);
  auto num_chunks                    = (src_size + (num_word_bits - 1)) / num_word_bits;
  cudf::size_type thread_valid_count = 0;

  auto i = static_cast<int64_t>(threadIdx.x) +
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  auto stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

  for (; i < num_chunks; i += stride) {
    thread_valid_count += bools_to_bits_chunk(src, src_size, i, &dst[i]);
  }

  sum_counts_subkernel(valid_count, thread_valid_count);
}

__device__ void set_all(cudf::size_type src_size,
                        cudf::bitmask_type* __restrict__ dst,
                        bool all_valid)
{
  constexpr auto num_word_bits = static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);
  auto num_chunks              = (src_size + (num_word_bits - 1)) / num_word_bits;
  auto fill_value              = all_valid ? ~cudf::bitmask_type{0} : cudf::bitmask_type{0};

  auto i = static_cast<int64_t>(threadIdx.x) +
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  auto stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

  for (; i < num_chunks; i += stride) {
    dst[i] = fill_value;
  }
}

__device__ void copy(cudf::bitmask_type const* __restrict__ src,
                     cudf::size_type src_size,
                     cudf::bitmask_type* __restrict__ dst)
{
  constexpr auto num_word_bits = static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);
  auto num_chunks              = (src_size + (num_word_bits - 1)) / num_word_bits;

  auto i = static_cast<int64_t>(threadIdx.x) +
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  auto stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

  for (; i < num_chunks; i += stride) {
    dst[i] = src[i];
  }
}

}  // namespace cudf::detail::jit::bit_utilities
