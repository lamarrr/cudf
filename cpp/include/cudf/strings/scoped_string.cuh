
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

#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>

#include <cuda/std/atomic>

namespace cudf {

/**
 * @brief A scope-local arena. Typically a thread-scope. At the end of the scope, the allocation
 * becomes invalid. This is ideal for temporary strings.
 */
class alignas(32) arena {
  /// @brief Current allocation offset
  uint8_t* m_next;

  /// @brief End of the memory block
  uint8_t* m_end;

  /// @brief The beginning of the memory block
  uint8_t* m_begin;

  /// @brief The total size (in bytes) of all allocations made
  size_t m_allocated;

 public:
  CUDF_HOST_DEVICE arena(uint8_t* begin, size_t size)
    : m_next(begin), m_end(begin + size), m_begin(begin), m_allocated(0)
  {
  }

  CUDF_HOST_DEVICE arena() : arena{static_cast<uint8_t*>(nullptr), 0} {}

  CUDF_HOST_DEVICE arena(void* begin, size_t size) : arena{static_cast<uint8_t*>(begin), size} {}

  CUDF_HOST_DEVICE arena(arena const&)            = delete;
  CUDF_HOST_DEVICE arena(arena&&)                 = delete;
  CUDF_HOST_DEVICE arena& operator=(arena const&) = delete;
  CUDF_HOST_DEVICE arena& operator=(arena&&)      = delete;
  CUDF_HOST_DEVICE ~arena()                       = default;

  /// @brief Allocate `size` bytes of memory
  /// @param size Allocation size in bytes
  /// @returns `nullptr` if allocation fails
  CUDF_HOST_DEVICE void* allocate(size_t size)
  {
    if (size == 0) { return nullptr; }

    if ((m_next + size) > m_end) { return nullptr; }

    auto allocation = m_next;
    m_next += size;
    m_allocated += size;
    return allocation;
  }

  /// @brief Deallocate the allocation `memory` from the arena
  /// @param memory The allocated memory
  /// @param size The size of the memory. must be 0 if nullptr
  CUDF_HOST_DEVICE void deallocate(void* memory, size_t size)
  {
    if (memory == nullptr || size == 0) { return; }

    uint8_t* mem = static_cast<uint8_t*>(memory);

    if ((mem + size) == m_next) {
      m_next -= size;
      return;
    }

    m_allocated -= size;
  }

  /// @brief Reallocate the allocation `old_memory` from size `old_size` to size `new_size`
  /// @param old_memory The allocated memory
  /// @param old_size The previous size of the memory
  /// @param new_size The new size of the memory
  /// @returns Returns the allocated memory if successful, otherwise false
  __device__ void* reallocate(void* old_memory, size_t old_size, size_t new_size)
  {
    if (old_memory == nullptr || old_size == 0) { return allocate(new_size); }

    if (new_size == 0) {
      deallocate(old_memory, old_size);
      return nullptr;
    }

    uint8_t* old = static_cast<uint8_t*>(old_memory);

    // if latest allocation, try to extend
    if ((old + old_size) == m_next && (old + new_size) <= m_end) {
      m_next      = old + new_size;
      m_allocated = m_next - m_begin;
      return old;
    }

    auto allocation = allocate(new_size);

    if (allocation == nullptr) { return nullptr; }

    memcpy(allocation, old_memory, std::min(old_size, new_size));

    deallocate(old_memory, old_size);

    return allocation;
  }

  /// @brief Release all the memory owned by this arena
  /// @warning This will invalidate all memory allocated from this arena
  CUDF_HOST_DEVICE void release_all()
  {
    m_next      = m_begin;
    m_allocated = 0;
  }

  /// @brief Check if the arena contains the specified memory region
  /// @returns `true` if the memory is owned by this arena, otherwise `false`
  CUDF_HOST_DEVICE bool contains(void* memory, size_t size) const
  {
    uint8_t* mem = static_cast<uint8_t*>(memory);
    return mem >= m_begin && (mem + size) <= m_end;
  }
};

/**
 * @brief A multi-threaded arena, This is ideal for allocating string outputs.
 */
class alignas(32) output_arena {
  /// @brief Current allocation offset
  uint8_t* m_next;

  /// @brief End of the memory block
  uint8_t* m_end;

  /// @brief The beginning of the memory block
  uint8_t* m_begin;

 public:
  CUDF_HOST_DEVICE output_arena(uint8_t* begin, size_t size)
    : m_next(begin), m_end(begin + size), m_begin(begin)
  {
  }

  CUDF_HOST_DEVICE output_arena() : output_arena{static_cast<uint8_t*>(nullptr), 0} {}

  CUDF_HOST_DEVICE output_arena(void* begin, size_t size)
    : output_arena{static_cast<uint8_t*>(begin), size}
  {
  }

  CUDF_HOST_DEVICE output_arena(output_arena const&)            = delete;
  CUDF_HOST_DEVICE output_arena(output_arena&&)                 = delete;
  CUDF_HOST_DEVICE output_arena& operator=(output_arena const&) = delete;
  CUDF_HOST_DEVICE output_arena& operator=(output_arena&&)      = delete;
  CUDF_HOST_DEVICE ~output_arena()                              = default;

  __device__ void* allocate(size_t size)
  {
    if (size == 0) { return nullptr; }

    cuda::std::atomic_ref next{m_next};

    uint8_t* allocation = nullptr;
    auto expected       = m_end + 1;
    auto target         = expected;

    while (!next.compare_exchange_weak(expected, target, cuda::std::memory_order_relaxed)) {
      // we've ran out of memory
      if ((expected + size) > m_end) { return nullptr; }

      allocation = expected;
      target     = expected + size;
    }

    return allocation;
  }

  __device__ void reset()
  {
    cuda::std::atomic_ref next{m_next};
    next.store(m_begin, cuda::std::memory_order_relaxed);
  }

  CUDF_HOST_DEVICE bool contains(void* memory, size_t size) const
  {
    uint8_t* mem = static_cast<uint8_t*>(memory);
    return mem >= m_begin && (mem + size) <= m_end;
  }
};

struct heap_allocator {
  static __device__ void* allocate(size_t size)
  {
    if (size == 0) { return nullptr; }
    return malloc(size);
  }

  static __device__ void deallocate(void* memory, [[maybe_unused]] size_t size) { free(memory); }

  static __device__ void* reallocate(void* old_memory, size_t old_size, size_t new_size)
  {
    if (old_memory == nullptr || old_size == 0) { return allocate(new_size); }

    if (new_size == 0) {
      deallocate(old_memory, old_size);
      return nullptr;
    }

    auto* new_mem = allocate(new_size);
    memcpy(new_mem, old_memory, std::min(old_size, new_size));
    deallocate(old_memory, old_size);

    return new_mem;
  }
};

enum class memory_source : int32_t { NONE = 0, HEAP = 1, ARENA = 2 };

struct scoped_string {
 private:
  char* m_data;
  size_t m_size;
  size_t m_capacity;
  arena* m_arena;
  memory_source m_source;

 public:
  __device__ scoped_string(
    char* data, size_t size, size_t capacity, arena* arena, memory_source source)
    : m_data{data}, m_size{size}, m_capacity{capacity}, m_arena{arena}, m_source{source}
  {
  }

  __device__ scoped_string(arena* arena) : scoped_string{nullptr, 0, 0, arena, memory_source::NONE}
  {
  }

  explicit __device__ scoped_string() : scoped_string{nullptr, 0, 0, nullptr, memory_source::NONE}
  {
  }

  __device__ scoped_string(scoped_string&& other)
    : m_data{other.m_data},
      m_size{other.m_size},
      m_capacity{other.m_capacity},
      m_arena{other.m_arena},
      m_source(other.m_source)
  {
    other.m_data     = nullptr;
    other.m_size     = 0;
    other.m_capacity = 0;
    other.m_arena    = nullptr;
    other.m_source   = memory_source::NONE;
  }

  __device__ scoped_string& operator=(scoped_string&& other)
  {
    if (this == &other) { return *this; }

    reset();

    m_data     = other.m_data;
    m_size     = other.m_size;
    m_capacity = other.m_capacity;
    m_arena    = other.m_arena;
    m_source   = other.m_source;

    other.m_data     = nullptr;
    other.m_size     = 0;
    other.m_capacity = 0;
    other.m_arena    = nullptr;
    other.m_source   = memory_source::NONE;

    return *this;
  }

  __device__ scoped_string(scoped_string const& other)            = delete;
  __device__ scoped_string& operator=(scoped_string const& other) = delete;

  __device__ ~scoped_string() { reset(); }

  __device__ void reset()
  {
    switch (m_source) {
      case memory_source::NONE: break;
      case memory_source::ARENA: m_arena->deallocate(m_data, m_capacity); break;
      case memory_source::HEAP: heap_allocator::deallocate(m_data, m_capacity); break;
    }

    m_data     = nullptr;
    m_size     = 0;
    m_capacity = 0;
    m_arena    = nullptr;
    m_source   = memory_source::NONE;
  }

  __device__ operator cudf::string_view() const
  {
    return cudf::string_view{m_data, static_cast<cudf::size_type>(m_size)};
  }

  __device__ cudf::string_view view() const { return *this; }

  CUDF_HOST_DEVICE size_t capacity() const { return m_capacity; }

  CUDF_HOST_DEVICE size_t size() const { return m_size; }

  __device__ size_t length() const
  {
    return cudf::strings::detail::characters_in_string(m_data, m_size);
  }

  CUDF_HOST_DEVICE bool is_empty() const { return size() > 0; }

  CUDF_HOST_DEVICE char const* data() const { return m_data; }

  CUDF_HOST_DEVICE char* data() { return m_data; }

  __device__ scoped_string& assign(char const* data, size_t size_bytes)
  {
    reserve(size_bytes);
    memcpy(m_data, data, size_bytes);
    m_size = size_bytes;
    return *this;
  }

  __device__ scoped_string& assign(cudf::string_view view)
  {
    return assign(view.data(), view.size_bytes());
  }

  __device__ scoped_string& append(std::initializer_list<cudf::string_view> strings)
  {
    size_t total = 0;
    for (auto s : strings) {
      total += s.size_bytes();
    }

    grow(m_size + total);

    auto it = m_data + m_size;

    for (auto s : strings) {
      memcpy(it, s.data(), s.size_bytes());
      it += s.size_bytes();
    }

    m_size = total;

    return *this;
  }

  __device__ scoped_string& append(cudf::string_view str) { return append({str}); }

  __device__ scoped_string& insert(size_t pos, cudf::string_view str);

  __device__ scoped_string& replace(size_t pos, cudf::string_view str);

  static __device__ scoped_string copy(cudf::string_view str, arena* arena)
  {
    scoped_string out{arena};
    out.assign(str.data(), str.size_bytes());
    return out;
  }

  static __device__ scoped_string repeat_char(size_t count, cudf::char_utf8 chr, arena* arena)
  {
    scoped_string out{arena};

    auto size = cudf::strings::detail::bytes_in_char_utf8(chr) * count;

    out.reserve(size);

    out.m_size = size;

    auto it = out.data();

    for (size_t i = 0; i < count; i++) {
      it += cudf::strings::detail::from_char_utf8(chr, it);
    }

    return out;
  }

  static __device__ scoped_string join(std::initializer_list<cudf::string_view> strings,
                                       arena* arena)
  {
    scoped_string out{arena};
    out.append(strings);
    return out;
  }

 private:
  __device__ bool try_reallocate(size_t target_capacity)
  {
    switch (m_source) {
      case memory_source::NONE: {
        assert(m_data == nullptr && m_capacity == 0);

        if (m_arena != nullptr) {
          if (auto p = m_arena->allocate(target_capacity)) {
            m_data     = static_cast<char*>(p);
            m_capacity = target_capacity;
            return true;
          }
        }

        if (auto p = heap_allocator::allocate(target_capacity)) {
          m_data     = static_cast<char*>(p);
          m_capacity = target_capacity;
          return true;
        }

      } break;

      case memory_source::ARENA: {
        // try re-allocating on the current source
        if (auto p = m_arena->reallocate(m_data, m_capacity, target_capacity); p != nullptr) {
          m_data     = static_cast<char*>(p);
          m_capacity = target_capacity;
          return true;
        }

        // arena has ran out of memory, transfer memory to heap
        if (auto p = heap_allocator::allocate(target_capacity)) {
          memcpy(p, m_data, std::min(m_capacity, target_capacity));
          m_arena->deallocate(m_data, m_capacity);
          m_data     = static_cast<char*>(p);
          m_capacity = target_capacity;
          return true;
        }

      } break;

      case memory_source::HEAP: {
        // try re-allocating on the current source
        if (auto p = heap_allocator::reallocate(m_data, m_capacity, target_capacity);
            p != nullptr) {
          m_data     = static_cast<char*>(p);
          m_capacity = target_capacity;
          return true;
        }
      } break;

      default: __builtin_unreachable(); break;
    }

    return false;
  }

  __device__ void reallocate(size_t target_capacity)
  {
    // TODO: handle alloc errors
    try_reallocate(target_capacity);
  }

  __device__ bool try_reserve(size_t target_capacity)
  {
    if (m_capacity >= target_capacity) { return true; }
    return try_reallocate(target_capacity);
  }

  __device__ void reserve(size_t target_capacity)
  {
    // TODO: handle alloc errors
    try_reserve(target_capacity);
  }

  __device__ bool try_grow(size_t target_size)
  {
    if (m_capacity >= target_size) { return true; }

    return try_reallocate(std::max(m_size << 1, target_size));
  }

  __device__ void grow(size_t target_size)
  {
    // TODO: handle alloc errors
    try_grow(target_size);
  }
};

struct string_output {
  char* data           = nullptr;
  size_t size          = 0;
  memory_source source = memory_source::NONE;

  __device__ void release()
  {
    switch (source) {
      case memory_source::NONE: break;
      case memory_source::HEAP: {
        heap_allocator::deallocate(data, size);
      }
      case memory_source::ARENA: break;
      default: __builtin_unreachable(); break;
    }

    data   = nullptr;
    size   = 0;
    source = memory_source::NONE;
  }
};

struct string_accumulator {
  output_arena* arena   = nullptr;
  string_output* column = nullptr;

  __device__ cuda::std::tuple<void*, memory_source> allocate(size_t size)
  {
    if (arena != nullptr) {
      if (auto* mem = arena->allocate(size); mem != nullptr) { return {mem, memory_source::ARENA}; }
    }

    if (auto* mem = heap_allocator::allocate(size); mem != nullptr) {
      return {mem, memory_source::HEAP};
    }

    // TODO: handle heap alloc failure
    return {};
  }

  __device__ void assign(size_t row, cudf::string_view str)
  {
    auto [mem, source] = allocate(str.size_bytes());

    memcpy(mem, str.data(), str.size_bytes());

    column[row] =
      string_output{static_cast<char*>(mem), static_cast<size_t>(str.size_bytes()), source};
  }
};

}  // namespace cudf
