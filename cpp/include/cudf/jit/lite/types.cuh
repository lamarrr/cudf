/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf {
namespace jit {
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

using size_t    = unsigned long long;
using intptr_t  = int64_t;
using uintptr_t = uint64_t;

using intmax_t  = int64_t;
using uintmax_t = uint64_t;

using float32_t = float;
using float64_t = double;

using size_type = int32_t;

using char_utf8 = uint32_t;

using bitmask_t = uint32_t;

template <typename T>
__device__ constexpr bool bit_is_set(T const* bitmask, size_t bit_index)
{
  constexpr auto bits_per_word = sizeof(T) * 8;
  return bitmask[bit_index / bits_per_word] & (T{1} << (bit_index % bits_per_word));
}

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
__device__ constexpr T dec_lshift(T v, int32_t scale)
{
  return v * ipow10(-scale);
}

template <typename T>
__device__ constexpr T dec_rshift(T v, int32_t scale)
{
  return v / ipow10(scale);
}

template <typename T>
__device__ constexpr T dec_shift(T v, int32_t scale)
{
  if (scale == 0) {
    return v;
  } else if (scale < 0) {
    return dec_lshift(v, scale);
  } else {
    return dec_rshift(v, scale);
  }
}

template <typename T>
__device__ constexpr T dec_rescale(T v, int32_t from_scale, int32_t to_scale)
{
  return dec_shift(v, to_scale - from_scale);
}

struct scaled_t {};

inline constexpr scaled_t scaled;

template <typename R>
struct dec {
  using Rep = R;

  R _value = 0;

  int32_t _scale = 0;

  __device__ constexpr dec(scaled_t, R value, int32_t scale) : _value{value}, _scale{scale} {}

  constexpr dec() = default;

  __device__ constexpr R value() const { return _value; }

  __device__ constexpr int32_t scale() const { return _scale; }
};

using dec32  = dec<int32_t>;
using dec64  = dec<int64_t>;
using dec128 = dec<int128_t>;

template <typename R>
__device__ constexpr auto rescale(dec<R> a, int32_t scale)
{
  return dec<R>{scaled, dec_rescale(a._value, a._scale, scale), scale};
}

template <typename R>
__device__ constexpr auto operator+(dec<R> a, dec<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value + rescale(b, scale)._value;
  return dec<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr auto operator-(dec<R> a, dec<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value - rescale(b, scale)._value;
  return dec<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr auto operator*(dec<R> a, dec<R> b)
{
  return dec<R>{scaled, a._value * b._value, a._scale + b._scale};
}

template <typename R>
__device__ constexpr auto operator/(dec<R> a, dec<R> b)
{
  return dec<R>{scaled, a._value / b._value, a._scale - b._scale};
}

template <typename R>
__device__ constexpr auto operator%(dec<R> a, dec<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value % rescale(b, scale)._value;
  return dec<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr int operator<=>(dec<R> a, dec<R> b)
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

struct string_view {
  static constexpr size_type const UNKNOWN_STRING_LENGTH{-1};
  static constexpr size_type const npos{-1};

  char const* _data = nullptr;

  size_type _bytes = 0;

  mutable size_type _length = UNKNOWN_STRING_LENGTH;

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

    size_type idx = 0;

    while (idx < max) {
      if (*s0 != *s1) return static_cast<int32_t>(*s0) - static_cast<int32_t>(*s1);
      s0++;
      s1++;
      idx++;
    }

    if (idx < n0) { return 1; }
    if (idx < n1) { return -1; }

    return 0;
  }
};

__device__ constexpr int operator<=>(string_view const& a, string_view const& b)
{
  return a.compare(b);
}

struct inplace_t {};

inline constexpr inplace_t inplace;

struct nullopt_t {};

inline constexpr nullopt_t nullopt;

template <typename T>
struct optional {
  T _val = {};

  bool _engaged = false;

  constexpr optional() = default;

  __device__ constexpr optional(nullopt_t) {}

  template <typename... Args>
  __device__ constexpr optional(inplace_t, Args&&... args)
    : _val{static_cast<Args&&>(args)...}, _engaged{true}
  {
  }

  __device__ constexpr optional(T val) : _val{val}, _engaged{true} {}

  __device__ constexpr bool has_value() const { return _engaged; }

  __device__ constexpr bool is_valid() const { return _engaged; }

  __device__ constexpr bool is_null() const { return !_engaged; }

  __device__ constexpr void reset() { _engaged = false; }

  __device__ constexpr T const& get() const { return _val; }

  __device__ constexpr T& get() { return _val; }

  __device__ constexpr T const* operator->() const { return &_val; }

  __device__ constexpr T* operator->() { return &_val; }

  __device__ constexpr T const& operator*() const { return _val; }

  __device__ constexpr T& operator*() { return _val; }

  __device__ constexpr T const& value() const { return _val; }

  __device__ constexpr T& value() { return _val; }

  __device__ constexpr explicit operator bool() const { return _engaged; }

  __device__ constexpr T value_or(T __v) const { return _engaged ? _val : __v; }
};

template <typename T>
optional(T) -> optional<T>;

template <typename T>
struct span {
  T* _data = nullptr;

  size_t _size = 0;

  __device__ constexpr T* data() const { return _data; }

  __device__ constexpr size_t size() const { return _size; }

  __device__ constexpr bool empty() const { return _size == 0; }

  __device__ constexpr T& operator[](size_t pos) const { return _data[pos]; }

  __device__ constexpr T* begin() const { return _data; }

  __device__ constexpr T* end() const { return _data + _size; }

  __device__ constexpr span<T const> as_const() const { return span<T const>{_data, _size}; }

  template <typename U = T>
  __device__ constexpr T& element(size_t idx) const
  {
    return _data[idx];
  }
};

template <typename T>
span(T*, size_t) -> span<T>;

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
concept Decimal = Same<T, dec32> || Same<T, dec64> || Same<T, dec128>;

template <typename T>
constexpr bool IsDictionaryElement = false;

template <typename IndexType, typename KeyType>
constexpr bool IsDictionaryElement<dictionary_element<IndexType, KeyType>> = true;

template <typename T>
concept DictionaryElement = IsDictionaryElement<T>;

}  // namespace traits

// TODO: scope variables should be aligned to avoid uncoalesced reads/writes
template <bool Mutable = false, bool AsScalar = false, bool AllValid = false>
struct alignas(16) column_accessor {
 private:
  data_type _type = {};

  size_type _size = 0;

  void* __restrict__ _data = nullptr;

  bitmask_t* __restrict__ _null_mask = nullptr;

  size_type _offset = 0;

  column_accessor* __restrict__ _d_children = nullptr;

  size_type _num_children = 0;

  // TODO: has_nulls

  __device__ pair<int64_t, int64_t> get_string_offsets(size_type idx) const __restrict__
  {
    static constexpr int32_t OFFSETS_CHILD = 0;
    auto i                                 = _offset + idx;
    auto* __restrict__ offsets             = _d_children + OFFSETS_CHILD;
    auto* __restrict__ i32_runs            = static_cast<int32_t const*>(offsets->_data);
    auto* __restrict__ i64_runs            = static_cast<int64_t const*>(offsets->_data);

    int64_t run_begin = 0;
    int64_t run_end   = 0;

    switch (offsets->type().id()) {
      case type_id::INT32:
        run_begin = i32_runs[i];
        run_end   = i32_runs[i + 1];
        break;
      case type_id::INT64:
        run_begin = i64_runs[i];
        run_end   = i64_runs[i + 1];
        break;
      default: __builtin_unreachable();
    }

    int64_t run_size = run_end - run_begin;

    return {run_begin, run_size};
  }

  __device__ static constexpr size_type map_index(size_type idx)
  {
    if constexpr (AsScalar) { return 0; }
    return idx;
  }

 public:
  __device__ constexpr data_type type() const __restrict__ { return _type; }

  __device__ constexpr size_type size() const __restrict__ { return _size; }

  __device__ constexpr bool nullable() const __restrict__ { return _null_mask != nullptr; }

  __device__ constexpr bitmask_t* __restrict__ null_mask() const
    __restrict__ requires(Mutable) { return _null_mask; }

  __device__ constexpr bitmask_t const* __restrict__ null_mask() const
    __restrict__ requires(!Mutable) { return _null_mask; }

  __device__ constexpr size_type offset() const __restrict__
  {
    return _offset;
  }

  __device__ constexpr bool is_valid(size_type idx) const __restrict__
  {
    if constexpr (AllValid) { return true; }
    return !nullable() || bit_is_set(_null_mask, _offset + map_index(idx));
  }

  __device__ constexpr bool is_null(size_type idx) const __restrict__ { return !is_valid(idx); }

  __device__ constexpr size_type num_child_columns() const __restrict__ { return _num_children; }

  template <traits::ReprCompatible B>
  __device__ auto& element(size_type idx) const __restrict__
  {
    if constexpr (Mutable) {
      auto* __restrict__ p = static_cast<B*>(_data) + _offset + map_index(idx);
      return *p;
    } else {
      auto* __restrict__ p = static_cast<B const*>(_data) + _offset + map_index(idx);
      return *p;
    }
  }

  template <traits::Decimal D>
  __device__ auto element(size_type idx) const __restrict__
  {
    auto* __restrict__ p = static_cast<typename D::Rep const*>(_data) + _offset + map_index(idx);
    return D{scaled, *p, _type.scale()};
  }

  template <traits::Same<string_view> T>
  __device__ string_view element(size_type idx) const __restrict__
  {
    auto* __restrict__ chars   = static_cast<char const*>(_data);
    auto [run_begin, run_size] = get_string_offsets(map_index(idx));
    return string_view{chars + run_begin, static_cast<size_type>(run_size)};
  }

  template <traits::Same<span<char>> T>
  __device__ span<char> element(size_type idx) const __restrict__ requires(Mutable) {
    auto* __restrict__ chars   = static_cast<char*>(_data);
    auto [run_begin, run_size] = get_string_offsets(map_index(idx));
    auto* __restrict__ str     = chars + run_begin;
    return span{str, static_cast<size_t>(run_size)};
  }

  template <typename T>
  __device__ optional<T> nullable_element(size_type idx) const __restrict__
  {
    if (!is_valid(idx)) return nullopt;
    return element<T>(idx);
  }

  /* __device__ void assign(args scope, size_type i, T value)
    {
      auto p     = static_cast<Arg>(scope[ScopeIndex]);
      auto index = IsScalar ? 0 : i;

      p->template assign<T>(index, value);
    }*/

  // TODO: assign_null_word()
};

#if !CUDF_JIT_LITE_EXCLUDE_OPERATORS

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

__device__ inline void arccos(float32_t* out, float32_t const* a) { *out = __builtin_acosf(*a); }

__device__ inline void arccos(float64_t* out, float64_t const* a) { *out = __builtin_acos(*a); }

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

__device__ inline void arccosh(float32_t* out, float32_t const* a) { *out = __builtin_acoshf(*a); }

__device__ inline void arccosh(float64_t* out, float64_t const* a) { *out = __builtin_acosh(*a); }

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

__device__ inline void arcsin(float32_t* out, float32_t const* a) { *out = __builtin_asinf(*a); }

__device__ inline void arcsin(float64_t* out, float64_t const* a) { *out = __builtin_asin(*a); }

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

__device__ inline void arcsinh(float32_t* out, float32_t const* a) { *out = __builtin_asinhf(*a); }

__device__ inline void arcsinh(float64_t* out, float64_t const* a) { *out = __builtin_asinh(*a); }

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

__device__ inline void arctan(float32_t* out, float32_t const* a) { *out = __builtin_atanf(*a); }

__device__ inline void arctan(float64_t* out, float64_t const* a) { *out = __builtin_atan(*a); }

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

__device__ inline void arctanh(float32_t* out, float32_t const* a) { *out = __builtin_atanhf(*a); }

__device__ inline void arctanh(float64_t* out, float64_t const* a) { *out = __builtin_atanh(*a); }

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

__device__ inline void cbrt(float32_t* out, float32_t const* a) { *out = __builtin_cbrtf(*a); }

__device__ inline void cbrt(float64_t* out, float64_t const* a) { *out = __builtin_cbrt(*a); }

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

__device__ inline void ceil(float32_t* out, float32_t const* a) { *out = __builtin_ceilf(*a); }

__device__ inline void ceil(float64_t* out, float64_t const* a) { *out = __builtin_ceil(*a); }

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

__device__ inline void cos(float32_t* out, float32_t const* a) { *out = __builtin_cosf(*a); }

__device__ inline void cos(float64_t* out, float64_t const* a) { *out = __builtin_cos(*a); }

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

__device__ inline void cosh(float32_t* out, float32_t const* a) { *out = __builtin_coshf(*a); }

__device__ inline void cosh(float64_t* out, float64_t const* a) { *out = __builtin_cosh(*a); }

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

__device__ inline void exp(float32_t* out, float32_t const* a) { *out = __builtin_expf(*a); }

__device__ inline void exp(float64_t* out, float64_t const* a) { *out = __builtin_exp(*a); }

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

__device__ inline void floor(float32_t* out, float32_t const* a) { *out = __builtin_floorf(*a); }

__device__ inline void floor(float64_t* out, float64_t const* a) { *out = __builtin_floor(*a); }

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

__device__ inline void log(float32_t* out, float32_t const* a) { *out = __builtin_logf(*a); }

__device__ inline void log(float64_t* out, float64_t const* a) { *out = __builtin_log(*a); }

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
  *out = __builtin_fmodf(*a, *b);
}

__device__ inline void mod(float64_t* out, float64_t const* a, float64_t const* b)
{
  *out = __builtin_fmod(*a, *b);
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
  *out = __builtin_powf(*a, *b);
}

__device__ inline void pow(float64_t* out, float64_t const* a, float64_t const* b)
{
  *out = __builtin_pow(*a, *b);
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
  *out = __builtin_fmodf(__builtin_fmodf(*a, *b) + *b, *b);
}

__device__ inline void pymod(float64_t* out, float64_t const* a, float64_t const* b)
{
  *out = __builtin_fmod(__builtin_fmod(*a, *b) + *b, *b);
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

__device__ inline void rint(float32_t* out, float32_t const* a) { *out = __builtin_rintf(*a); }

__device__ inline void rint(float64_t* out, float64_t const* a) { *out = __builtin_rint(*a); }

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

__device__ inline void sin(float32_t* out, float32_t const* a) { *out = __builtin_sinf(*a); }

__device__ inline void sin(float64_t* out, float64_t const* a) { *out = __builtin_sin(*a); }

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

__device__ inline void sinh(float32_t* out, float32_t const* a) { *out = __builtin_sinhf(*a); }

__device__ inline void sinh(float64_t* out, float64_t const* a) { *out = __builtin_sinh(*a); }

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

__device__ inline void tanh(float32_t* out, float32_t const* a) { *out = __builtin_tanhf(*a); }

__device__ inline void tanh(float64_t* out, float64_t const* a) { *out = __builtin_tanh(*a); }

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
}  // namespace jit
}  // namespace cudf
