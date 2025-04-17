/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/udf/udf_string.hpp>

#include <algorithm>
#include <limits>

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

CUDF_HOST_DEVICE udf_string::udf_string(allocator_type allocator)
  : m_data(nullptr),
    m_bytes(0),
    m_capacity(0),
    m_source(memory_source::NONE),
    m_allocator(allocator)
{
}

__device__ inline udf_string::udf_string(char const* data,
                                         cudf::size_type bytes,
                                         allocator_type allocator)
  : udf_string(allocator)
{
  reserve(bytes);
  memcpy(m_data, data, bytes);
  m_bytes         = bytes;
  m_data[m_bytes] = '\0';
}

__device__ udf_string::udf_string(cudf::size_type count,
                                  cudf::char_utf8 chr,
                                  allocator_type allocator)
  : udf_string(allocator)
{
  if (count <= 0) { return; }
  auto target_size = cudf::strings::detail::bytes_in_char_utf8(chr) * count;
  reserve(target_size);
  m_bytes = target_size;

  auto out_ptr = m_data;
  for (cudf::size_type idx = 0; idx < count; ++idx) {
    out_ptr += cudf::strings::detail::from_char_utf8(chr, out_ptr);
  }
  *out_ptr = '\0';
}

__device__ inline udf_string::udf_string(char const* data, allocator_type allocator)
  : udf_string(data, detail::bytes_in_null_terminated_string(data), allocator)
{
}

__device__ inline udf_string::udf_string(udf_string const& src)
  : udf_string(src.m_data, src.m_bytes, src.m_allocator)
{
}

__device__ inline udf_string::udf_string(udf_string const& src, allocator_type allocator)
  : udf_string(src.m_data, src.m_bytes, allocator)
{
}

__device__ inline udf_string::udf_string(udf_string&& src)
  : m_data(src.m_data),
    m_bytes(src.m_bytes),
    m_capacity(src.m_capacity),
    m_source(src.m_source),
    m_allocator(src.m_allocator)
{
  src.m_data      = nullptr;
  src.m_bytes     = 0;
  src.m_capacity  = 0;
  src.m_source    = memory_source::NONE;
  src.m_allocator = allocator_type{};
}

__device__ inline udf_string::udf_string(udf_string&& src, fallback_allocator allocator)
  : udf_string{allocator}
{
  if (src.m_source == memory_source::NONE || src.m_source == memory_source::HEAP ||
      src.m_allocator == allocator) {
    assign(std::move(src));
    m_allocator = allocator;
  } else {
    assign(src);
  }
}

__device__ inline udf_string::udf_string(cudf::string_view str, allocator_type allocator)
  : udf_string(str.data(), str.size_bytes(), allocator)
{
}

__device__ inline udf_string::~udf_string() { reset(); }

__device__ inline udf_string& udf_string::operator=(udf_string const& str) { return assign(str); }

__device__ inline udf_string& udf_string::operator=(udf_string&& str)
{
  return assign(std::move(str));
}

__device__ inline udf_string& udf_string::operator=(cudf::string_view str) { return assign(str); }

__device__ inline udf_string& udf_string::operator=(char const* str) { return assign(str); }

__device__ udf_string& udf_string::assign(udf_string&& str)
{
  if (this == &str) { return *this; }

  reset();

  m_data          = str.m_data;
  m_bytes         = str.m_bytes;
  m_capacity      = str.m_capacity;
  m_source        = str.m_source;
  m_allocator     = str.m_allocator;
  str.m_data      = nullptr;
  str.m_bytes     = 0;
  str.m_capacity  = 0;
  str.m_source    = memory_source::NONE;
  str.m_allocator = allocator_type{};

  return *this;
}

__device__ udf_string& udf_string::assign(cudf::string_view str)
{
  return assign(str.data(), str.size_bytes());
}

__device__ udf_string& udf_string::assign(char const* str)
{
  return assign(str, detail::bytes_in_null_terminated_string(str));
}

__device__ udf_string& udf_string::assign(char const* str, cudf::size_type bytes)
{
  reserve(bytes);
  m_bytes = bytes;
  memcpy(m_data, str, bytes);
  m_data[m_bytes] = '\0';
  return *this;
}

__device__ inline cudf::size_type udf_string::size_bytes() const { return m_bytes; }

__device__ inline cudf::size_type udf_string::length() const
{
  return cudf::strings::detail::characters_in_string(m_data, m_bytes);
}

__device__ constexpr cudf::size_type udf_string::max_size() const
{
  return std::numeric_limits<cudf::size_type>::max() - 1;
}

__device__ inline char* udf_string::data() { return m_data; }

__device__ inline char const* udf_string::data() const { return m_data; }

__device__ inline bool udf_string::is_empty() const { return m_bytes == 0; }

__device__ inline cudf::string_view::const_iterator udf_string::begin() const
{
  return cudf::string_view::const_iterator(cudf::string_view(m_data, m_bytes), 0);
}

__device__ inline cudf::string_view::const_iterator udf_string::end() const
{
  return cudf::string_view::const_iterator(cudf::string_view(m_data, m_bytes), length());
}

__device__ inline cudf::char_utf8 udf_string::at(cudf::size_type pos) const
{
  auto const offset = byte_offset(pos);
  auto chr          = cudf::char_utf8{0};
  if (offset < m_bytes) { cudf::strings::detail::to_char_utf8(data() + offset, chr); }
  return chr;
}

__device__ inline cudf::char_utf8 udf_string::operator[](cudf::size_type pos) const
{
  return at(pos);
}

__device__ inline cudf::size_type udf_string::byte_offset(cudf::size_type pos) const
{
  cudf::size_type offset = 0;

  auto start = m_data;
  auto end   = start + m_bytes;
  while ((pos > 0) && (start < end)) {
    auto const byte       = static_cast<uint8_t>(*start++);
    auto const char_bytes = cudf::strings::detail::bytes_in_utf8_byte(byte);
    if (char_bytes) { --pos; }
    offset += char_bytes;
  }
  return offset;
}

__device__ inline int udf_string::compare(cudf::string_view in) const
{
  return compare(in.data(), in.size_bytes());
}

__device__ inline int udf_string::compare(char const* data, cudf::size_type bytes) const
{
  auto const view = static_cast<cudf::string_view>(*this);
  return view.compare(data, bytes);
}

__device__ inline bool udf_string::operator==(cudf::string_view rhs) const
{
  return m_bytes == rhs.size_bytes() && compare(rhs) == 0;
}

__device__ inline bool udf_string::operator!=(cudf::string_view rhs) const
{
  return compare(rhs) != 0;
}

__device__ inline bool udf_string::operator<(cudf::string_view rhs) const
{
  return compare(rhs) < 0;
}

__device__ inline bool udf_string::operator>(cudf::string_view rhs) const
{
  return compare(rhs) > 0;
}

__device__ inline bool udf_string::operator<=(cudf::string_view rhs) const
{
  return compare(rhs) <= 0;
}

__device__ inline bool udf_string::operator>=(cudf::string_view rhs) const
{
  return compare(rhs) >= 0;
}

__device__ inline void udf_string::clear() { m_bytes = 0; }

__device__ inline void udf_string::reset()
{
  m_allocator.deallocate(m_source, m_data, m_capacity);
  m_data     = nullptr;
  m_bytes    = 0;
  m_capacity = 0;
  m_source   = memory_source::NONE;
}

__device__ udf_string::allocation udf_string::release()
{
  allocation a{
    m_data,
    static_cast<size_t>(m_capacity),
    m_source,
  };
  m_data     = nullptr;
  m_bytes    = 0;
  m_capacity = 0;
  m_source   = memory_source::NONE;
  return a;
}

__device__ udf_string::allocator_type const& udf_string::get_allocator() const
{
  return m_allocator;
}

__device__ inline void udf_string::resize(cudf::size_type count)
{
  if (count > max_size()) { return; }

  grow(count);

  // add padding if necessary (null chars)
  if (count > m_bytes) { memset(m_data + m_bytes, 0, count - m_bytes); }

  m_bytes         = count;
  m_data[m_bytes] = '\0';
}

__device__ void udf_string::reserve(cudf::size_type count) { util_reserve(count + 1); }

__device__ void udf_string::shrink_to_fit()
{
  if ((m_bytes + 1) < m_capacity) { util_reallocate(m_bytes + 1); }
}

__device__ inline udf_string& udf_string::append(char const* str, cudf::size_type bytes)
{
  if (bytes <= 0) { return *this; }
  grow(m_bytes + bytes);
  memcpy(m_data + m_bytes, str, bytes);
  m_bytes += bytes;
  m_data[m_bytes] = '\0';
  return *this;
}

__device__ inline udf_string& udf_string::append(char const* str)
{
  return append(str, detail::bytes_in_null_terminated_string(str));
}

__device__ inline udf_string& udf_string::append(cudf::char_utf8 chr, cudf::size_type count)
{
  auto d_str = udf_string(count, chr, m_allocator);
  return append(d_str);
}

__device__ inline udf_string& udf_string::append(cudf::string_view in)
{
  return append(in.data(), in.size_bytes());
}

__device__ inline udf_string& udf_string::operator+=(cudf::string_view in) { return append(in); }

__device__ inline udf_string& udf_string::operator+=(cudf::char_utf8 chr) { return append(chr); }

__device__ inline udf_string& udf_string::operator+=(char const* str) { return append(str); }

__device__ inline udf_string& udf_string::insert(cudf::size_type pos,
                                                 char const* str,
                                                 cudf::size_type in_bytes)
{
  return replace(pos, 0, str, in_bytes);
}

__device__ inline udf_string& udf_string::insert(cudf::size_type pos, char const* str)
{
  return insert(pos, str, detail::bytes_in_null_terminated_string(str));
}

__device__ inline udf_string& udf_string::insert(cudf::size_type pos, cudf::string_view in)
{
  return insert(pos, in.data(), in.size_bytes());
}

__device__ inline udf_string& udf_string::insert(cudf::size_type pos,
                                                 cudf::size_type count,
                                                 cudf::char_utf8 chr)
{
  return replace(pos, 0, count, chr);
}

__device__ inline udf_string udf_string::substr(cudf::size_type pos, cudf::size_type count) const
{
  if (pos < 0) { return udf_string{"", 0, m_allocator}; }
  auto const start_pos = byte_offset(pos);
  if (start_pos >= m_bytes) { return udf_string{"", 0, m_allocator}; }
  auto const end_pos = count < 0 ? m_bytes : std::min(byte_offset(pos + count), m_bytes);
  return udf_string{data() + start_pos, end_pos - start_pos, m_allocator};
}

// utility for replace()
__device__ void udf_string::shift_bytes(cudf::size_type start_pos,
                                        cudf::size_type end_pos,
                                        cudf::size_type nbytes)
{
  if (nbytes < m_bytes) {
    // shift bytes to the left [...wxyz] -> [wxyzxyz]
    auto src = end_pos;
    auto tgt = start_pos;
    while (tgt < nbytes) {
      m_data[tgt++] = m_data[src++];
    }
  } else if (nbytes > m_bytes) {
    // shift bytes to the right [abcd...] -> [abcabcd]
    auto src = m_bytes;
    auto tgt = nbytes;
    while (src > end_pos) {
      m_data[--tgt] = m_data[--src];
    }
  }
}

__device__ inline udf_string& udf_string::replace(cudf::size_type pos,
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
  grow(nbytes);

  // move bytes -- make room for replacement
  shift_bytes(start_pos + in_bytes, end_pos, nbytes);

  // insert the replacement
  memcpy(m_data + start_pos, str, in_bytes);

  m_bytes         = nbytes;
  m_data[m_bytes] = '\0';
  return *this;
}

__device__ inline udf_string& udf_string::replace(cudf::size_type pos,
                                                  cudf::size_type count,
                                                  char const* str)
{
  return replace(pos, count, str, detail::bytes_in_null_terminated_string(str));
}

__device__ inline udf_string& udf_string::replace(cudf::size_type pos,
                                                  cudf::size_type count,
                                                  cudf::string_view in)
{
  return replace(pos, count, in.data(), in.size_bytes());
}

__device__ inline udf_string& udf_string::replace(cudf::size_type pos,
                                                  cudf::size_type count,
                                                  cudf::size_type chr_count,
                                                  cudf::char_utf8 chr)
{
  auto d_str = udf_string(chr_count, chr, m_allocator);
  return replace(pos, count, d_str);
}

__device__ udf_string& udf_string::erase(cudf::size_type pos, cudf::size_type count)
{
  return replace(pos, count, nullptr, 0);
}

__device__ inline cudf::size_type udf_string::char_offset(cudf::size_type byte_pos) const
{
  return cudf::strings::detail::characters_in_string(data(), byte_pos);
}

__device__ void udf_string::util_reallocate(size_type capacity)
{
  assert(capacity > 0 && capacity < max_size() && "Invalid capacity");
  if (capacity > max_size()) { return; }

  // TODO: handle alloc errors
  bool success =
    m_allocator.reallocate(m_source, reinterpret_cast<void*&>(m_data), m_capacity, capacity);

  assert(success && "Memory allocation failed");
}

__device__ void udf_string::util_reserve(size_type capacity)
{
  if (m_capacity >= capacity) { return; }

  util_reallocate(capacity);
}

__device__ void udf_string::grow(size_type count) { util_reserve(max(m_capacity * 2, count + 1)); }

}  // namespace udf
}  // namespace strings
}  // namespace cudf
