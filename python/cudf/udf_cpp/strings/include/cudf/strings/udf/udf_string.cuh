/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include "udf_string.hpp"

#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>

#include <algorithm>
#include <limits>
#include <string>

namespace cudf {
namespace strings {
namespace udf {
namespace detail {

/**
 * @brief Count the bytes in a null-terminated character array
 *
 * @param str Null-terminated string
 * @return Number of bytes in `str` up to but not including the null-terminator
 */
__device__ inline static cudf::size_type bytes_in_null_terminated_string(char const* str)
{
  if (!str) return 0;
  cudf::size_type bytes = 0;
  while (*str++)
    ++bytes;
  return bytes;
}

}  // namespace detail

template <typename Buffer>
__device__ string<Buffer>::string(allocation_scope scope) : Buffer{scope}, m_bytes{0}
{
}

template <typename Buffer>
__device__ string<Buffer>::string(char const* data, cudf::size_type bytes, allocation_scope scope)
  : Buffer{make_storage_copy<Buffer>(data, bytes, scope)}, m_bytes{bytes}
{
}

template <typename Buffer>
__device__ string<Buffer>::string(cudf::size_type count,
                                  cudf::char_utf8 chr,
                                  allocation_scope scope)
  : Buffer{scope}
{
  if (count <= 0) { return; }
  auto size = cudf::strings::detail::bytes_in_char_utf8(chr) * count;
  Buffer::reserve(size);
  m_bytes = size;

  auto out = data();
  for (cudf::size_type idx = 0; idx < count; ++idx) {
    out += cudf::strings::detail::from_char_utf8(chr, out);
  }
}

template <typename Buffer>
__device__ string<Buffer>::string(char const* data, allocation_scope scope)
  : string{data, detail::bytes_in_null_terminated_string(data), scope}
{
}

template <typename Buffer>
template <typename SrcBuffer>
__device__ inline string<Buffer>::string(string<SrcBuffer> const& src)
  : string{src.data(), src.m_bytes, src.m_scope}
{
}

template <typename Buffer>
template <typename SrcBuffer>
__device__ inline string<Buffer>::string(string<SrcBuffer>&& src)
  : Buffer{std::move(static_cast<SrcBuffer&>(src))}, m_bytes{src.m_bytes}
{
  src.m_bytes = 0;
}

template <typename Buffer>
__device__ inline string<Buffer>::string(cudf::string_view str, allocation_scope scope)
  : string(str.data(), str.size_bytes(), scope)
{
}

template <typename Buffer>
template <typename SrcBuffer>
__device__ inline string<Buffer>& string<Buffer>::operator=(string<SrcBuffer> const& str)
{
  return assign(str);
}

template <typename Buffer>
template <typename SrcBuffer>
__device__ inline string<Buffer>& string<Buffer>::operator=(string<SrcBuffer>&& str)
{
  return assign(std::move(str));
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::operator=(cudf::string_view str)
{
  return assign(str);
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::operator=(char const* str)
{
  return assign(str);
}

template <typename Buffer>
template <typename SrcBuffer>
__device__ string<Buffer>& string<Buffer>::assign(string<SrcBuffer>&& str)
{
  if (this == &str) { return *this; }

  Buffer::adopt(std::move(static_cast<SrcBuffer&&>(str)), str.m_bytes);
  str.m_bytes = 0;

  return *this;
}

template <typename Buffer>
__device__ string<Buffer>& string<Buffer>::assign(cudf::string_view str)
{
  return assign(str.data(), str.size_bytes());
}

template <typename Buffer>
__device__ string<Buffer>& string<Buffer>::assign(char const* str)
{
  return assign(str, detail::bytes_in_null_terminated_string(str));
}

template <typename Buffer>
__device__ string<Buffer>& string<Buffer>::assign(char const* str, cudf::size_type bytes)
{
  Buffer::reserve(m_bytes);
  memcpy(data(), str, bytes);
  return *this;
}

template <typename Buffer>
__device__ inline cudf::size_type string<Buffer>::size_bytes() const
{
  return m_bytes;
}

template <typename Buffer>
__device__ inline cudf::size_type string<Buffer>::length() const
{
  return cudf::strings::detail::characters_in_string(data(), m_bytes);
}

template <typename Buffer>
__device__ constexpr cudf::size_type string<Buffer>::max_size() const
{
  return std::numeric_limits<cudf::size_type>::max() - 1;
}

template <typename Buffer>
__device__ inline char* string<Buffer>::data()
{
  return static_cast<char*>(Buffer::memory());
}

template <typename Buffer>
__device__ inline char const* string<Buffer>::data() const
{
  return static_cast<char const*>(Buffer::memory());
}

template <typename Buffer>
__device__ inline bool string<Buffer>::is_empty() const
{
  return m_bytes == 0;
}

template <typename Buffer>
__device__ inline cudf::string_view::const_iterator string<Buffer>::begin() const
{
  return cudf::string_view::const_iterator(cudf::string_view(data(), m_bytes), 0);
}

template <typename Buffer>
__device__ inline cudf::string_view::const_iterator string<Buffer>::end() const
{
  return cudf::string_view::const_iterator(cudf::string_view(data(), m_bytes), length());
}

template <typename Buffer>
__device__ inline cudf::char_utf8 string<Buffer>::at(cudf::size_type pos) const
{
  auto const offset = byte_offset(pos);
  auto chr          = cudf::char_utf8{0};
  if (offset < m_bytes) { cudf::strings::detail::to_char_utf8(data() + offset, chr); }
  return chr;
}

template <typename Buffer>
__device__ inline cudf::char_utf8 string<Buffer>::operator[](cudf::size_type pos) const
{
  return at(pos);
}

template <typename Buffer>
__device__ inline cudf::size_type string<Buffer>::byte_offset(cudf::size_type pos) const
{
  cudf::size_type offset = 0;

  auto start = data();
  auto end   = start + m_bytes;
  while ((pos > 0) && (start < end)) {
    auto const byte       = static_cast<uint8_t>(*start++);
    auto const char_bytes = cudf::strings::detail::bytes_in_utf8_byte(byte);
    if (char_bytes) { --pos; }
    offset += char_bytes;
  }
  return offset;
}

template <typename Buffer>
__device__ inline int string<Buffer>::compare(cudf::string_view in) const
{
  return compare(in.data(), in.size_bytes());
}

template <typename Buffer>
__device__ inline int string<Buffer>::compare(char const* data, cudf::size_type bytes) const
{
  auto const view = static_cast<cudf::string_view>(*this);
  return view.compare(data, bytes);
}

template <typename Buffer>
__device__ inline bool string<Buffer>::operator==(cudf::string_view rhs) const
{
  return m_bytes == rhs.size_bytes() && compare(rhs) == 0;
}

template <typename Buffer>
__device__ inline bool string<Buffer>::operator!=(cudf::string_view rhs) const
{
  return compare(rhs) != 0;
}

template <typename Buffer>
__device__ inline bool string<Buffer>::operator<(cudf::string_view rhs) const
{
  return compare(rhs) < 0;
}

template <typename Buffer>
__device__ inline bool string<Buffer>::operator>(cudf::string_view rhs) const
{
  return compare(rhs) > 0;
}

template <typename Buffer>
__device__ inline bool string<Buffer>::operator<=(cudf::string_view rhs) const
{
  return compare(rhs) <= 0;
}

template <typename Buffer>
__device__ inline bool string<Buffer>::operator>=(cudf::string_view rhs) const
{
  return compare(rhs) >= 0;
}

template <typename Buffer>
__device__ inline void string<Buffer>::clear()
{
  m_bytes = 0;
}

template <typename Buffer>
__device__ inline void string<Buffer>::reset()
{
  Buffer::reset();
  m_bytes = 0;
}

template <typename Buffer>
__device__ inline void string<Buffer>::resize(cudf::size_type count)
{
  if (count > max_size()) { return; }
  Buffer::reserve(count);

  if (count > m_bytes) { memset(data() + m_bytes, 0, count - m_bytes); }

  m_bytes = count;
}

template <typename Buffer>
__device__ void string<Buffer>::reserve(cudf::size_type count)
{
  if (count < max_size()) { Buffer::reserve(count); }
}

template <typename Buffer>
__device__ cudf::size_type string<Buffer>::capacity() const
{
  return Buffer::capacity();
}

template <typename Buffer>
__device__ void string<Buffer>::shrink_to_fit()
{
  if (m_bytes < Buffer::capacity()) { Buffer::reallocate(m_bytes); }
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::append(char const* str, cudf::size_type bytes)
{
  auto const new_size_bytes = m_bytes + bytes;
  Buffer::grow(new_size_bytes);
  memcpy(data() + m_bytes, str, bytes);
  m_bytes = new_size_bytes;
  return *this;
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::append(char const* str)
{
  return append(str, detail::bytes_in_null_terminated_string(str));
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::append(cudf::char_utf8 chr, cudf::size_type count)
{
  auto d_str = string<Buffer>(count, chr);
  return append(d_str);
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::append(cudf::string_view in)
{
  return append(in.data(), in.size_bytes());
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::operator+=(cudf::string_view in)
{
  return append(in);
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::operator+=(cudf::char_utf8 chr)
{
  return append(chr);
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::operator+=(char const* str)
{
  return append(str);
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::insert(cudf::size_type pos,
                                                         char const* str,
                                                         cudf::size_type in_bytes)
{
  return replace(pos, 0, str, in_bytes);
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::insert(cudf::size_type pos, char const* str)
{
  return insert(pos, str, detail::bytes_in_null_terminated_string(str));
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::insert(cudf::size_type pos, cudf::string_view in)
{
  return insert(pos, in.data(), in.size_bytes());
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::insert(cudf::size_type pos,
                                                         cudf::size_type count,
                                                         cudf::char_utf8 chr)
{
  return replace(pos, 0, count, chr);
}

template <typename Buffer>
__device__ inline string<Buffer> string<Buffer>::substr(cudf::size_type pos,
                                                        cudf::size_type count) const
{
  if (pos < 0) { return string<Buffer>{"", 0}; }
  auto const start_pos = byte_offset(pos);
  if (start_pos >= m_bytes) { return string<Buffer>{"", 0}; }
  auto const end_pos = count < 0 ? m_bytes : std::min(byte_offset(pos + count), m_bytes);
  return string<Buffer>{data() + start_pos, end_pos - start_pos};
}

// utility for replace()
template <typename Buffer>
__device__ void string<Buffer>::shift_bytes(cudf::size_type start_pos,
                                            cudf::size_type end_pos,
                                            cudf::size_type nbytes)
{
  if (nbytes < m_bytes) {
    // shift bytes to the left [...wxyz] -> [wxyzxyz]
    auto src = end_pos;
    auto tgt = start_pos;
    while (tgt < nbytes) {
      data()[tgt++] = data()[src++];
    }
  } else if (nbytes > m_bytes) {
    // shift bytes to the right [abcd...] -> [abcabcd]
    auto src = m_bytes;
    auto tgt = nbytes;
    while (src > end_pos) {
      data()[--tgt] = data()[--src];
    }
  }
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::replace(cudf::size_type pos,
                                                          cudf::size_type count,
                                                          char const* str,
                                                          cudf::size_type in_bytes)
{
  if (pos < 0 || in_bytes < 0) { return *this; }
  auto const start_pos = byte_offset(pos);
  if (start_pos > m_bytes) { return *this; }
  auto const end_pos = count < 0 ? m_bytes : std::min(byte_offset(pos + count), m_bytes);

  // compute new size
  auto const nbytes = m_bytes + in_bytes - (end_pos - start_pos);
  Buffer::grow(nbytes);

  // move bytes -- make room for replacement
  shift_bytes(start_pos + in_bytes, end_pos, nbytes);

  // insert the replacement
  memcpy(data() + start_pos, str, in_bytes);

  m_bytes = nbytes;
  return *this;
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::replace(cudf::size_type pos,
                                                          cudf::size_type count,
                                                          char const* str)
{
  return replace(pos, count, str, detail::bytes_in_null_terminated_string(str));
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::replace(cudf::size_type pos,
                                                          cudf::size_type count,
                                                          cudf::string_view in)
{
  return replace(pos, count, in.data(), in.size_bytes());
}

template <typename Buffer>
__device__ inline string<Buffer>& string<Buffer>::replace(cudf::size_type pos,
                                                          cudf::size_type count,
                                                          cudf::size_type chr_count,
                                                          cudf::char_utf8 chr)
{
  auto d_str = string<Buffer>(chr_count, chr);
  return replace(pos, count, d_str);
}

template <typename Buffer>
__device__ string<Buffer>& string<Buffer>::erase(cudf::size_type pos, cudf::size_type count)
{
  return replace(pos, count, nullptr, 0);
}

template <typename Buffer>
__device__ inline cudf::size_type string<Buffer>::char_offset(cudf::size_type byte_pos) const
{
  return cudf::strings::detail::characters_in_string(data(), byte_pos);
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
