/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf {
namespace lite {

using int8_t   = signed char;
using int16_t  = signed short;
using int32_t  = signed int;
using int64_t  = signed long long;
using int128_t = __int128_t;
using uint8_t  = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
using uint128_t = __uint128_t;

using size_t    = unsigned long long;
using intptr_t  = int64_t;
using uintptr_t = uint64_t;

using intmax_t  = int64_t;
using uintmax_t = uint64_t;

using float32_t = float;
using float64_t = double;

using size_type = int32_t;

using char_utf8 = uint32_t;

using bitmask_type = uint32_t;

enum class type_id : int32_t {
  EMPTY                  = 0,
  INT8                   = 1,
  INT16                  = 2,
  INT32                  = 3,
  INT64                  = 4,
  UINT8                  = 5,
  UINT16                 = 6,
  UINT32                 = 7,
  UINT64                 = 8,
  FLOAT32                = 9,
  FLOAT64                = 10,
  BOOL8                  = 11,
  TIMESTAMP_DAYS         = 12,
  TIMESTAMP_SECONDS      = 13,
  TIMESTAMP_MILLISECONDS = 14,
  TIMESTAMP_MICROSECONDS = 15,
  TIMESTAMP_NANOSECONDS  = 16,
  DURATION_DAYS          = 17,
  DURATION_SECONDS       = 18,
  DURATION_MILLISECONDS  = 19,
  DURATION_MICROSECONDS  = 20,
  DURATION_NANOSECONDS   = 21,
  DICTIONARY32           = 22,
  STRING                 = 23,
  LIST                   = 24,
  DECIMAL32              = 25,
  DECIMAL64              = 26,
  DECIMAL128             = 27,
  STRUCT                 = 28,
  NUM_TYPE_IDS           = 29
};

template <typename T0, typename T1>
struct pair {
  T0 v0{};
  T1 v1{};
};

struct data_type {
  type_id _id = {};

  int32_t _scale = 0;

  __device__ constexpr type_id id() const { return _id; }

  __device__ constexpr int32_t scale() const { return _scale; }
};

template <typename T>
__device__ constexpr T min(T a, T b)
{
  return a < b ? a : b;
}

template <typename T>
__device__ constexpr T max(T a, T b)
{
  return a > b ? a : b;
}

template <typename T>
__device__ constexpr T ipow10(T exponent)
{
  if (exponent == 0) { return 1; }

  T extra  = 1;
  T square = 10;
  T n      = exponent;

  while (n > 1) {
    if ((n & 1) == 1) { extra *= square; }
    n >>= 1;
    square *= square;
  }

  return square * extra;
}

template <typename T>
__device__ constexpr T decimal_lshift(T v, int32_t scale)
{
  return v * ipow10(-scale);
}

template <typename T>
__device__ constexpr T decimal_rshift(T v, int32_t scale)
{
  return v / ipow10(scale);
}

template <typename T>
__device__ constexpr T decimal_shift(T v, int32_t scale)
{
  if (scale == 0) {
    return v;
  } else if (scale < 0) {
    return decimal_lshift(v, scale);
  } else {
    return decimal_rshift(v, scale);
  }
}

template <typename T>
__device__ constexpr T decimal_rescale(T v, int32_t from_scale, int32_t to_scale)
{
  return decimal_shift(v, to_scale - from_scale);
}

struct scaled_t {};

inline constexpr scaled_t scaled;

template <typename R>
struct decimal {
  using Rep = R;

  R _value = 0;

  int32_t _scale = 0;

  __device__ constexpr decimal(scaled_t, R value, int32_t scale) : _value{value}, _scale{scale} {}

  constexpr decimal() = default;

  __device__ constexpr R value() const { return _value; }

  __device__ constexpr int32_t scale() const { return _scale; }
};

using decimal32  = decimal<int32_t>;
using decimal64  = decimal<int64_t>;
using decimal128 = decimal<int128_t>;

template <typename R>
__device__ constexpr auto rescale(decimal<R> a, int32_t scale)
{
  return decimal<R>{scaled, decimal_rescale(a._value, a._scale, scale), scale};
}

template <typename R>
__device__ constexpr auto operator+(decimal<R> a, decimal<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value + rescale(b, scale)._value;
  return decimal<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr auto operator-(decimal<R> a, decimal<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value - rescale(b, scale)._value;
  return decimal<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr auto operator*(decimal<R> a, decimal<R> b)
{
  return decimal<R>{scaled, a._value * b._value, a._scale + b._scale};
}

template <typename R>
__device__ constexpr auto operator/(decimal<R> a, decimal<R> b)
{
  return decimal<R>{scaled, a._value / b._value, a._scale - b._scale};
}

template <typename R>
__device__ constexpr auto operator%(decimal<R> a, decimal<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value % rescale(b, scale)._value;
  return decimal<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr int operator<=>(decimal<R> a, decimal<R> b)
{
  auto scale = min(a._scale, b._scale);
  return rescale(a, scale)._value - rescale(b, scale)._value;
}

enum class timestamp_unit : int32_t { D, h, m, s, ms, us, ns };

template <typename R, timestamp_unit Unit>
struct timestamp {
  using Rep = R;

  R _rep = 0;

  __device__ constexpr R count() const { return _rep; }
};

using timestamp_D  = timestamp<int32_t, timestamp_unit::D>;
using timestamp_h  = timestamp<int32_t, timestamp_unit::h>;
using timestamp_m  = timestamp<int32_t, timestamp_unit::m>;
using timestamp_s  = timestamp<int64_t, timestamp_unit::s>;
using timestamp_ms = timestamp<int64_t, timestamp_unit::ms>;
using timestamp_us = timestamp<int64_t, timestamp_unit::us>;
using timestamp_ns = timestamp<int64_t, timestamp_unit::ns>;

template <typename R, timestamp_unit Unit>
__device__ constexpr int operator<=>(timestamp<R, Unit> a, timestamp<R, Unit> b)
{
  return a._rep - b._rep;
}

template <typename R, timestamp_unit Unit>
struct duration {
  using Rep = R;

  R _rep = 0;

  __device__ constexpr R count() const { return _rep; }
};

using duration_D  = duration<int32_t, timestamp_unit::D>;
using duration_h  = duration<int32_t, timestamp_unit::h>;
using duration_m  = duration<int32_t, timestamp_unit::m>;
using duration_s  = duration<int64_t, timestamp_unit::s>;
using duration_ms = duration<int64_t, timestamp_unit::ms>;
using duration_us = duration<int64_t, timestamp_unit::us>;
using duration_ns = duration<int64_t, timestamp_unit::ns>;

template <typename R, timestamp_unit Unit>
__device__ constexpr duration<R, Unit> operator+(duration<R, Unit> a, duration<R, Unit> b)
{
  return duration<R, Unit>{a._rep + b._rep};
}

template <typename R, timestamp_unit Unit>
__device__ constexpr duration<R, Unit> operator-(duration<R, Unit> a, duration<R, Unit> b)
{
  return duration<R, Unit>{a._rep - b._rep};
}

template <typename R, timestamp_unit Unit>
__device__ constexpr int operator<=>(duration<R, Unit> a, duration<R, Unit> b)
{
  return a._rep - b._rep;
}

struct inplace_t {};

inline constexpr inplace_t inplace;

struct nullopt_t {};

inline constexpr nullopt_t nullopt;

template <typename T>
struct optional {
  T _val = {};

  bool _is_valid = false;

  constexpr optional() = default;

  __device__ constexpr optional(nullopt_t) {}

  template <typename... Args>
  __device__ constexpr optional(inplace_t, Args&&... args)
    : _val{static_cast<Args&&>(args)...}, _is_valid{true}
  {
  }

  __device__ constexpr optional(T val) : _val{val}, _is_valid{true} {}

  __device__ constexpr bool is_valid() const { return _is_valid; }

  __device__ constexpr bool is_null() const { return !_is_valid; }

  __device__ constexpr void reset() { _is_valid = false; }

  __device__ constexpr T const& get() const { return _val; }

  __device__ constexpr T& get() { return _val; }

  __device__ constexpr T const* operator->() const { return &_val; }

  __device__ constexpr T* operator->() { return &_val; }

  __device__ constexpr T const& operator*() const { return _val; }

  __device__ constexpr T& operator*() { return _val; }

  __device__ constexpr T const& value() const { return _val; }

  __device__ constexpr T& value() { return _val; }

  __device__ constexpr explicit operator bool() const { return _is_valid; }

  __device__ constexpr T value_or(T v) const { return _is_valid ? _val : v; }
};

template <typename T>
optional(T) -> optional<T>;

template <typename T>
struct span {
  T* _data = nullptr;

  size_t _size = 0;

  constexpr span() = default;

  __device__ constexpr span(T* data, size_t size) : _data{data}, _size{size} {}

  __device__ constexpr T* data() const { return _data; }

  __device__ constexpr size_t size() const { return _size; }

  __device__ constexpr bool empty() const { return _size == 0; }

  __device__ constexpr T& operator[](size_t pos) const { return _data[pos]; }

  __device__ constexpr T* begin() const { return _data; }

  __device__ constexpr T* end() const { return _data + _size; }

  __device__ constexpr T const* cbegin() const { return _data; }

  __device__ constexpr T const* cend() const { return _data + _size; }

  __device__ constexpr span<T const> as_const() const { return span<T const>{_data, _size}; }

  __device__ constexpr T& element(size_t i) const { return _data[i]; }
};

template <typename T>
span(T*, size_t) -> span<T>;

struct string_view {
  static constexpr size_type const UNKNOWN_STRING_LENGTH{-1};
  static constexpr size_type const npos{-1};

  char const* _data = "";

  size_type _bytes = 0;

  mutable size_type _length = UNKNOWN_STRING_LENGTH;

  constexpr string_view() = default;

  __device__ constexpr string_view(char const* data, size_type bytes) : _data{data}, _bytes{bytes}
  {
  }

  __device__ constexpr size_type size_bytes() const { return _bytes; }

  __device__ constexpr char const* data() const { return _data; }

  __device__ constexpr bool empty() const { return _bytes == 0; }

  __device__ constexpr size_type compare(string_view const& other) const
  {
    auto* s0 = _data;
    auto n0  = _bytes;
    auto* s1 = other._data;
    auto n1  = other._bytes;
    auto max = n0 < n1 ? n0 : n1;

    if (s0 == s1 && n0 == n1) return 0;

    size_type i = 0;

    while (i < max) {
      if (*s0 != *s1) return static_cast<int32_t>(*s0) - static_cast<int32_t>(*s1);
      s0++;
      s1++;
      i++;
    }

    if (i < n0) { return 1; }
    if (i < n1) { return -1; }

    return 0;
  }
};

__device__ constexpr int operator<=>(string_view const& a, string_view const& b)
{
  return a.compare(b);
}

using mutable_string_view = span<char>;

template <typename IndexType, typename KeyType>
struct dictionary_element {
  KeyType key{};
};

namespace traits {

template <typename T, typename U>
inline constexpr bool IsSame = false;

template <typename T>
inline constexpr bool IsSame<T, T> = true;

template <typename T, typename U>
concept Same = IsSame<T, U>;

template <typename T>
concept ReprCompatible =
  Same<T, bool> || Same<T, int8_t> || Same<T, int16_t> || Same<T, int32_t> || Same<T, int64_t> ||
  Same<T, uint8_t> || Same<T, uint16_t> || Same<T, uint32_t> || Same<T, uint64_t> ||
  Same<T, float32_t> || Same<T, float64_t> || Same<T, timestamp_D> || Same<T, timestamp_h> ||
  Same<T, timestamp_m> || Same<T, timestamp_s> || Same<T, timestamp_ms> || Same<T, timestamp_us> ||
  Same<T, timestamp_ns> || Same<T, duration_D> || Same<T, duration_h> || Same<T, duration_m> ||
  Same<T, duration_s> || Same<T, duration_ms> || Same<T, duration_us> || Same<T, duration_ns>;

template <typename T>
concept Decimal = Same<T, decimal32> || Same<T, decimal64> || Same<T, decimal128>;

template <typename T>
constexpr bool IsDictionaryElement = false;

template <typename IndexType, typename KeyType>
constexpr bool IsDictionaryElement<dictionary_element<IndexType, KeyType>> = true;

template <typename T>
concept DictionaryElement = IsDictionaryElement<T>;

}  // namespace traits

template <typename T>
__device__ constexpr bool bit_is_set(T* bitmask, size_t bit_index)
{
  constexpr auto bits_per_word = sizeof(T) * 8;
  return bitmask[bit_index / bits_per_word] & (T{1} << (bit_index % bits_per_word));
}

template <bool Mutable, bool AsScalar, bool MayBeNullable>
struct alignas(16) column_accessor {
  static constexpr int32_t STRING_OFFSETS_CHILD_INDEX = 0;

  static constexpr bool is_mutable      = Mutable;
  static constexpr bool as_scalar       = AsScalar;
  static constexpr bool may_be_nullable = MayBeNullable;

 private:
  data_type _type = {};

  size_type _size = 0;

  void* __restrict__ _data = nullptr;

  bitmask_type* __restrict__ _null_mask = nullptr;

  size_type _offset = 0;

  void* __restrict__ _children = nullptr;

  size_type _num_children = 0;

  __device__ pair<int64_t, int64_t> get_string_offsets(size_type i) const __restrict__
  {
    using offsets_accessor = column_accessor<false, AsScalar, true>;

    auto* __restrict__ offsets =
      static_cast<offsets_accessor*>(_children) + STRING_OFFSETS_CHILD_INDEX;
    auto* __restrict__ i32_run = static_cast<int32_t const*>(offsets->_data) + _offset + i;
    auto* __restrict__ i64_run = static_cast<int64_t const*>(offsets->_data) + _offset + i;

    int64_t run_begin = 0;
    int64_t run_end   = 0;

    switch (offsets->type().id()) {
      case type_id::INT32:
        run_begin = i32_run[0];
        run_end   = i32_run[1];
        break;
      case type_id::INT64:
        run_begin = i64_run[0];
        run_end   = i64_run[1];
        break;
      default: __builtin_unreachable();
    }

    return {run_begin, run_end - run_begin};
  }

  __device__ static constexpr size_type map_index(size_type i)
  {
    if constexpr (AsScalar) { return 0; }
    return i;
  }

 public:
  __device__ constexpr data_type type() const __restrict__ { return _type; }

  __device__ constexpr size_type size() const __restrict__ { return _size; }

  __device__ constexpr bool nullable() const __restrict__ { return _null_mask != nullptr; }

  __device__ constexpr bitmask_type* __restrict__ null_mask() const
    __restrict__ requires(Mutable) { return _null_mask; }

  __device__ constexpr bitmask_type const* __restrict__ null_mask() const
    __restrict__ requires(!Mutable) { return _null_mask; }

  __device__ constexpr size_type offset() const __restrict__
  {
    return _offset;
  }

  __device__ constexpr bool is_valid(size_type i) const __restrict__
  {
    if constexpr (MayBeNullable) { return true; }
    return !nullable() || bit_is_set(_null_mask, _offset + map_index(i));
  }

  __device__ constexpr bool is_null(size_type i) const __restrict__ { return !is_valid(i); }

  __device__ constexpr size_type num_child_columns() const __restrict__ { return _num_children; }

  template <traits::ReprCompatible T>
  __device__ auto& element(size_type i) const __restrict__
  {
    if constexpr (Mutable) {
      auto* __restrict__ p = static_cast<T*>(_data) + _offset + map_index(i);
      return *p;
    } else {
      auto* __restrict__ p = static_cast<T const*>(_data) + _offset + map_index(i);
      return *p;
    }
  }

  template <traits::Decimal T>
  __device__ auto element(size_type i) const __restrict__
  {
    auto* __restrict__ p = static_cast<typename T::Rep const*>(_data) + _offset + map_index(i);
    return T{scaled, *p, _type.scale()};
  }

  template <traits::Same<string_view> T>
  __device__ string_view element(size_type i) const __restrict__
  {
    auto* __restrict__ chars   = static_cast<char const*>(_data);
    auto [run_begin, run_size] = get_string_offsets(map_index(i));
    return string_view{chars + run_begin, static_cast<size_type>(run_size)};
  }

  template <traits::Same<span<char>> T>
  __device__ mutable_string_view element(size_type i) const __restrict__ requires(Mutable) {
    auto* __restrict__ chars   = static_cast<char*>(_data);
    auto [run_begin, run_size] = get_string_offsets(map_index(i));
    auto* __restrict__ str     = chars + run_begin;
    return mutable_string_view{str, static_cast<size_t>(run_size)};
  }

  template <typename T>
  __device__ optional<T> nullable_element(size_type i) const __restrict__
  {
    if (!is_valid(i)) return nullopt;
    return element<T>(i);
  }

  template <traits::ReprCompatible T>
  __device__ void assign(size_type i, T value) const __restrict__ requires(Mutable && !AsScalar) {
    auto* __restrict__ p = static_cast<T*>(_data) + _offset + i;
    *p                   = value;
  }

  template <traits::Decimal T>
  __device__ void assign(size_type i, T value) const __restrict__ requires(Mutable && !AsScalar) {
    auto* __restrict__ p = static_cast<typename T::Rep*>(_data) + _offset + i;
    *p                   = value.value();
  }

  template <traits::Same<mutable_string_view> T>
  __device__ void assign(size_type i, T value) const __restrict__ requires(Mutable && !AsScalar) {
    // no-op
    return;
  }

  __device__ void assign_null_word(size_type word_index, bitmask_type value) const
    __restrict__ requires(Mutable && !AsScalar && !MayBeNullable) {
      _null_mask[word_index] = value;
    }
};

// TODO: typed column_accessor

// TODO: better name, this should work for
// fixed-width columns, including decimals
//
// TODO: vector_inplace_column
// TODO: asumptions: offset is a multiple of vector size
// TODO: alignment assumptions for the data, null-masks, offsets; they need to be vectorized
// TODO: maybe add type-specific prefetch types, i.e. string_prefetch
// TODO: aligned load/stores: __builtin_assume_aligned(ptr, alignment)
// how to load?
// template<typename T, size_t alignment/multiple> : register memory columns
// void vector_load(column_accessor<...>);
//
// void vector_store(size_type offset, column_accessor);
//
//
// TODO: dictionary
// TODO: struct?
// will not support list, string, map, they will not benefit from vectorized reads/writes due to
// their variable-length and pointer-chasing nature
//

template <size_t alignment, size_t size>
struct buffer {
  alignas(alignment) char _data[size];
};

template <size_t alignment>
struct buffer<alignment, 0> {
  static inline constexpr char* _data = nullptr;
};

template <size_t data_alignment,
          size_t data_size,
          size_t null_mask_alignment,
          size_t null_mask_size,
          size_t offsets_alignment,
          size_t offsets_size>
struct column_buffer {
  buffer<data_alignment, data_size> _data;
  buffer<null_mask_alignment, null_mask_size> _null_mask;
  buffer<offsets_alignment, offsets_size> _offsets;

  //   __device__ column_accessor<> as_column();
  //   __device__ void read();  // TODO: element size, multiple, must be aligned, etc.
  //   __device__ void write(size_type offset, column_accessor<>);
};

#if !defined(CUDF_JIT_LITE_EXCLUDE_OPERATORS)

namespace operators {

template <typename T>
__device__ inline void abs(T* out, T const* a)
{
  *out = (*a < 0) ? -*a : *a;
}

template <typename T>
__device__ inline void abs(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    abs(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void add(T* out, T const* a, T const* b)
{
  *out = (*a + *b);
}

template <typename T>
__device__ inline void add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    add(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void arccos(float32_t* out, float32_t const* a) { *out = ::acosf(*a); }

__device__ inline void arccos(float64_t* out, float64_t const* a) { *out = ::acos(*a); }

template <typename T>
__device__ inline void arccos(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arccos(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void arccosh(float32_t* out, float32_t const* a) { *out = ::acoshf(*a); }

__device__ inline void arccosh(float64_t* out, float64_t const* a) { *out = ::acosh(*a); }

template <typename T>
__device__ inline void arccosh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arccosh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void arcsin(float32_t* out, float32_t const* a) { *out = ::asinf(*a); }

__device__ inline void arcsin(float64_t* out, float64_t const* a) { *out = ::asin(*a); }

template <typename T>
__device__ inline void arcsin(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arcsin(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void arcsinh(float32_t* out, float32_t const* a) { *out = ::asinhf(*a); }

__device__ inline void arcsinh(float64_t* out, float64_t const* a) { *out = ::asinh(*a); }

template <typename T>
__device__ inline void arcsinh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arcsinh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void arctan(float32_t* out, float32_t const* a) { *out = ::atanf(*a); }

__device__ inline void arctan(float64_t* out, float64_t const* a) { *out = ::atan(*a); }

template <typename T>
__device__ inline void arctan(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arctan(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void arctanh(float32_t* out, float32_t const* a) { *out = ::atanhf(*a); }

__device__ inline void arctanh(float64_t* out, float64_t const* a) { *out = ::atanh(*a); }

template <typename T>
__device__ inline void arctanh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arctanh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void bit_and(T* out, T const* a, T const* b)
{
  *out = (*a & *b);
}

template <typename T>
__device__ inline void bit_and(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    bit_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void bit_invert(T* out, T const* a)
{
  *out = ~(*a);
}

template <typename T>
__device__ inline void bit_invert(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    bit_invert(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void bit_or(T* out, T const* a, T const* b)
{
  *out = (*a | *b);
}

template <typename T>
__device__ inline void bit_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    bit_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void bit_xor(T* out, T const* a, T const* b)
{
  *out = (*a ^ *b);
}

template <typename T>
__device__ inline void bit_xor(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    bit_xor(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void cast_to_float64(float64_t* out, T const* a)
{
  *out = static_cast<float64_t>(*a);
}

template <typename T>
__device__ inline void cast_to_float64(optional<float64_t>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    float64_t r;
    cast_to_float64(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void cast_to_int64(int64_t* out, T const* a)
{
  *out = static_cast<int64_t>(*a);
}

template <typename T>
__device__ inline void cast_to_int64(optional<int64_t>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    int64_t r;
    cast_to_int64(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void cast_to_uint64(uint64_t* out, T const* a)
{
  *out = static_cast<uint64_t>(*a);
}

template <typename T>
__device__ inline void cast_to_uint64(optional<uint64_t>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    uint64_t r;
    cast_to_uint64(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void cbrt(float32_t* out, float32_t const* a) { *out = ::cbrtf(*a); }

__device__ inline void cbrt(float64_t* out, float64_t const* a) { *out = ::cbrt(*a); }

template <typename T>
__device__ inline void cbrt(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cbrt(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void ceil(float32_t* out, float32_t const* a) { *out = ::ceilf(*a); }

__device__ inline void ceil(float64_t* out, float64_t const* a) { *out = ::ceil(*a); }

template <typename T>
__device__ inline void ceil(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    ceil(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void cos(float32_t* out, float32_t const* a) { *out = ::cosf(*a); }

__device__ inline void cos(float64_t* out, float64_t const* a) { *out = ::cos(*a); }

template <typename T>
__device__ inline void cos(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cos(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void cosh(float32_t* out, float32_t const* a) { *out = ::coshf(*a); }

__device__ inline void cosh(float64_t* out, float64_t const* a) { *out = ::cosh(*a); }

template <typename T>
__device__ inline void cosh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cosh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void div(T* out, T const* a, T const* b)
{
  *out = (*a / *b);
}

template <typename T>
__device__ inline void div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    div(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
}

template <typename T>
__device__ inline void equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*a == *b);
  } else if (a->is_null() && b->is_null()) {
    *out = true;
  } else {
    *out = false;
  }
}

__device__ inline void exp(float32_t* out, float32_t const* a) { *out = ::expf(*a); }

__device__ inline void exp(float64_t* out, float64_t const* a) { *out = ::exp(*a); }

template <typename T>
__device__ inline void exp(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    exp(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void floor(float32_t* out, float32_t const* a) { *out = ::floorf(*a); }

__device__ inline void floor(float64_t* out, float64_t const* a) { *out = ::floor(*a); }

template <typename T>
__device__ inline void floor(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    floor(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void greater(bool* out, T const* a, T const* b)
{
  *out = (*a > *b);
}

template <typename T>
__device__ inline void greater(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*a > *b);
  } else {
    *out = false;
  }
}

template <typename T>
__device__ inline void greater_equal(bool* out, T const* a, T const* b)
{
  *out = (*a >= *b);
}

template <typename T>
__device__ inline void greater_equal(optional<bool>* out,
                                     optional<T> const* a,
                                     optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*a >= *b);
  } else {
    *out = false;
  }
}

template <typename T>
__device__ inline void identity(T* out, T const* a)
{
  *out = *a;
}

template <typename T>
__device__ inline void identity(optional<T>* out, optional<T> const* a)
{
  *out = *a;
}

template <typename T>
__device__ inline void is_null(bool* out, T const* a)
{
  *out = false;
}

template <typename T>
__device__ inline void is_null(optional<bool>* out, optional<T> const* a)
{
  *out = a->is_null();
}

template <typename T>
__device__ inline void less(bool* out, T const* a, T const* b)
{
  *out = (*a < *b);
}

template <typename T>
__device__ inline void less(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*a < *b);
  } else {
    *out = false;
  }
}

template <typename T>
__device__ inline void less_equal(bool* out, T const* a, T const* b)
{
  *out = (*a <= *b);
}

template <typename T>
__device__ inline void less_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*a <= *b);
  } else {
    *out = false;
  }
}

__device__ inline void log(float32_t* out, float32_t const* a) { *out = ::logf(*a); }

__device__ inline void log(float64_t* out, float64_t const* a) { *out = ::log(*a); }

template <typename T>
__device__ inline void log(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    log(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void logical_and(T* out, T const* a, T const* b)
{
  *out = (*a && *b);
}

template <typename T>
__device__ inline void logical_and(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void logical_or(T* out, T const* a, T const* b)
{
  *out = (*a || *b);
}

template <typename T>
__device__ inline void logical_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void mod(T* out, T const* a, T const* b)
{
  *out = (*a % *b);
}

__device__ inline void mod(float32_t* out, float32_t const* a, float32_t const* b)
{
  *out = ::fmodf(*a, *b);
}

__device__ inline void mod(float64_t* out, float64_t const* a, float64_t const* b)
{
  *out = ::fmod(*a, *b);
}

template <typename T>
__device__ inline void mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    mod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void mul(T* out, T const* a, T const* b)
{
  *out = (*a * *b);
}

template <typename T>
__device__ inline void mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    mul(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void null_equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
}

template <typename T>
__device__ inline void null_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*a == *b);
  } else if (a->is_null() && b->is_null()) {
    *out = true;
  } else {
    *out = false;
  }
}

__device__ inline void null_logical_and(bool* out, bool const* a, bool const* b)
{
  *out = (*a && *b);
}

__device__ inline void null_logical_and(optional<bool>* out,
                                        optional<bool> const* a,
                                        optional<bool> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*a && *b);
  } else if (a->is_null() && b->is_null()) {
    *out = nullopt;
  } else {
    if (a->is_valid() ? *a : *b) {
      *out = nullopt;
    } else {
      *out = false;
    }
  }
}

__device__ inline void null_logical_or(bool* out, bool const* a, bool const* b)
{
  *out = (*a || *b);
}

__device__ inline void null_logical_or(optional<bool>* out,
                                       optional<bool> const* a,
                                       optional<bool> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*a || *b);
  } else if (a->is_null() && b->is_null()) {
    *out = nullopt;
  } else {
    if (a->is_valid() ? *a : *b) {
      *out = true;
    } else {
      *out = nullopt;
    }
  }
}

__device__ inline void pow(float32_t* out, float32_t const* a, float32_t const* b)
{
  *out = ::powf(*a, *b);
}

__device__ inline void pow(float64_t* out, float64_t const* a, float64_t const* b)
{
  *out = ::pow(*a, *b);
}

template <typename T>
__device__ inline void pow(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    pow(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void pymod(T* out, T const* a, T const* b)
{
  *out = (*a % *b + *b) % *b;
}

__device__ inline void pymod(float32_t* out, float32_t const* a, float32_t const* b)
{
  *out = ::fmodf(::fmodf(*a, *b) + *b, *b);
}

__device__ inline void pymod(float64_t* out, float64_t const* a, float64_t const* b)
{
  *out = ::fmod(::fmod(*a, *b) + *b, *b);
}

template <typename T>
__device__ inline void pymod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    pymod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void rint(float32_t* out, float32_t const* a) { *out = ::rintf(*a); }

__device__ inline void rint(float64_t* out, float64_t const* a) { *out = ::rint(*a); }

template <typename T>
__device__ inline void rint(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    rint(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void sin(float32_t* out, float32_t const* a) { *out = ::sinf(*a); }

__device__ inline void sin(float64_t* out, float64_t const* a) { *out = ::sin(*a); }

template <typename T>
__device__ inline void sin(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    sin(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void sinh(float32_t* out, float32_t const* a) { *out = ::sinhf(*a); }

__device__ inline void sinh(float64_t* out, float64_t const* a) { *out = ::sinh(*a); }

template <typename T>
__device__ inline void sinh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    sinh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void sub(T* out, T const* a, T const* b)
{
  *out = *a - *b;
}

template <typename T>
__device__ inline void sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    sub(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

__device__ inline void tanh(float32_t* out, float32_t const* a) { *out = ::tanhf(*a); }

__device__ inline void tanh(float64_t* out, float64_t const* a) { *out = ::tanh(*a); }

template <typename T>
__device__ inline void tanh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    tanh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
}

template <typename T>
__device__ inline void if_else(T* out,
                               bool const* condition,
                               T const* true_value,
                               T const* false_value)
{
  *out = *condition ? *true_value : *false_value;
}

template <typename T>
__device__ inline void if_else(optional<T>* out,
                               optional<bool> const* condition,
                               optional<T> const* true_value,
                               optional<T> const* false_value)
{
  if (condition->is_valid() && true_value->is_valid() && false_value->is_valid()) {
    if_else<T>(&out->value(), &condition->value(), &true_value->value(), &false_value->value());
  } else {
    *out = nullopt;
  }
}

}  // namespace operators

#endif

}  // namespace lite
}  // namespace cudf
