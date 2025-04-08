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

#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/variant>
#include <cuda_runtime.h>

// This header contains all class and function declarations so that it
// can be included in a .cpp file which only has declaration requirements
// (i.e. sizeof, conditionally-comparable, explicit conversions, etc).
// The definitions are coded in string.cuh which is to be included
// in .cu files that use this class in kernel calls.

namespace cudf {
namespace strings {
namespace udf {

// needs to support element-per-thread and strides-per-thread

/**
 * @brief A scope-local arena, the arena is cleared at the end of a scope. The scope can be a thread
 * or loop iteration. At the end of the scope, the allocation becomes invalidated. This is ideal for
 * temporary strings.
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
  __device__ void* allocate(size_t size)
  {
    if (size == 0) { return nullptr; }

    if (m_next + size > m_end) { return nullptr; }

    auto allocation = m_next;
    m_next += size;
    m_allocated += size;
    return allocation;
  }

  /// @brief Deallocate the allocation `memory` from the arena
  /// @param memory The allocated memory
  /// @param size The size of the memory. must be 0 if nullptr
  __device__ void deallocate(void* memory, size_t size)
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
    if (old_memory == nullptr || old_size == 0) {
      return allocate(new_size);
    } else if (new_size == 0) {
      deallocate(old_memory, old_size);
      return nullptr;
    }

    uint8_t* old = static_cast<uint8_t*>(old_memory);

    // if latest allocation, try to extend
    if ((old + old_size) == m_next) {
      if ((old + new_size) > m_end) { return nullptr; }

      m_next      = old + new_size;
      m_allocated = m_next - m_begin;
      return old;
    }

    // memory is already sized enough, re-use.
    // although this can lead to fragmentation as the actual size is now different.
    if (old_size > new_size) { return old; }

    auto allocation = allocate(new_size);

    if (allocation == nullptr) { return nullptr; }

    memcpy(allocation, old_memory, std::min(old_size, new_size));

    deallocate(old_memory, old_size);

    return allocation;
  }

  /// @brief Release all the memory owned by this arena
  /// @warning This will invalidate all memory allocated from this arena
  __device__ void release_all()
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
template <cuda::thread_scope Scope>
class alignas(32) mt_arena {
  /// @brief Current allocation offset
  uint8_t* m_next;

  /// @brief End of the memory block
  uint8_t* m_end;

  /// @brief The beginning of the memory block
  uint8_t* m_begin;

 public:
  CUDF_HOST_DEVICE mt_arena(uint8_t* begin, size_t size)
    : m_next(begin), m_end(begin + size), m_begin(begin)
  {
  }

  CUDF_HOST_DEVICE mt_arena() : mt_arena{static_cast<uint8_t*>(nullptr), 0} {}

  CUDF_HOST_DEVICE mt_arena(void* begin, size_t size) : mt_arena{static_cast<uint8_t*>(begin), size}
  {
  }

  CUDF_HOST_DEVICE mt_arena(mt_arena const&)            = delete;
  CUDF_HOST_DEVICE mt_arena(mt_arena&&)                 = delete;
  CUDF_HOST_DEVICE mt_arena& operator=(mt_arena const&) = delete;
  CUDF_HOST_DEVICE mt_arena& operator=(mt_arena&&)      = delete;
  CUDF_HOST_DEVICE ~mt_arena()                          = default;

  __device__ void* allocate(size_t size)
  {
    if (size == 0) { return nullptr; }

    cuda::atomic_ref<uint8_t*, Scope> next{m_next};

    uint8_t* allocation = nullptr;
    auto expected       = m_end + 1;
    auto target         = expected;

    while (!next.compare_exchange_weak(expected, target, cuda::memory_order_relaxed)) {
      if ((expected + size) > m_end) { return nullptr; }

      allocation = expected;
      target     = expected + size;
    }

    return allocation;
  }

  __device__ void* reallocate(void* old_memory, size_t old_size, size_t new_size)
  {
    if (old_memory == nullptr || old_size == 0) {
      return allocate(new_size);
    } else if (new_size == 0) {
      deallocate(old_memory, old_size);
      return nullptr;
    }

    cuda::atomic_ref<uint8_t*, Scope> next{m_next};

    uint8_t* old = static_cast<uint8_t*>(old_memory);

    {
      // try to extend allocation if is the latest
      auto expected = old + old_size;
      if (auto target = old + new_size; target <= m_end) {
        if (next.compare_exchange_strong(expected, target, cuda::memory_order_relaxed)) {
          return old;
        }
      }
    }

    // memory is already sized enough, re-use.
    // although this can lead to fragmentation as the actual size is now different.
    if (old_size > new_size) { return old; }

    // allocate a new memory
    auto allocation = allocate(new_size);

    if (allocation == nullptr) { return nullptr; }

    memcpy(allocation, old_memory, std::min(old_size, new_size));

    deallocate(old_memory, old_size);

    return allocation;
  }

  void deallocate(void* memory, size_t size)
  {
    if (memory == nullptr || size == 0) { return; }

    cuda::atomic_ref<uint8_t*, Scope> next{m_next};

    uint8_t* allocation = static_cast<uint8_t*>(memory);

    // if latest allocation, attempt to adjust back
    {
      auto expected = allocation + size;
      auto target   = allocation;

      if (next.compare_exchange_strong(expected, target, cuda::memory_order_relaxed)) { return; }
    }
  }

  __device__ void reset()
  {
    cuda::atomic_ref<uint8_t*, Scope> next{m_next};
    next.store(m_begin, cuda::memory_order_relaxed);
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
    if (old_memory == nullptr || old_size == 0) {
      return allocate(new_size);
    } else if (new_size == 0) {
      free(old_memory);
      return nullptr;
    }

    return realloc(old_memory, new_size);
  }
};

enum class memory_source : int32_t {
  /// @brief there is no source (i.e. no allocation was made)
  NONE = 0,

  /// @brief the memory was sourced from the device heap
  /// Synchronized?: Yes, via call to malloc
  HEAP = 1,

  /// @brief the memory was sourced from the provide output buffer
  /// Synchronized?: Yes, via atomic operations
  OUTPUT = 2,

  /// @brief the memory was sourced from device's block shared memory (i.e. `__shared__ char[ ]`)
  /// Synchronized?: No
  SHARED = 3,

  /// @brief the memory was sourced from the thread's scope (i.e. per-thread buffer or
  /// per-thread-scope buffer).
  /// Synchronized?: No
  SCOPE = 4
};

struct allocation {
  /// @brief the allocated memory. `nullptr` if allocation fails
  void* memory;

  /// @brief source of the memory
  memory_source source;

  __device__ constexpr allocation() : memory{nullptr}, source{memory_source::NONE} {}
  __device__ constexpr allocation(void* memory, memory_source source)
    : memory{memory}, source{source}
  {
  }
  __device__ constexpr allocation(allocation const&)            = default;
  __device__ constexpr allocation(allocation&&)                 = default;
  __device__ constexpr allocation& operator=(allocation const&) = default;
  __device__ constexpr allocation& operator=(allocation&&)      = default;
  __device__ ~allocation()                                      = default;
};

enum class storage_type : int32_t {
  /// @brief a temporary string. Its lifetime doesn't extend beyond its thread-local scope
  TEMPORARY = 0,

  /// @brief an output string. Its lifetime outlives the kernel.
  OUTPUT = 1
};

class allocation_scope {
  /// @brief the device heap
  static constexpr heap_allocator m_heap;

  /// @brief the pre-allocated output buffer
  mt_arena<cuda::thread_scope_system>* m_output_arena;

  /// @brief device's shared memory arena for the current thread
  arena* m_shared_arena;

  /// @brief the scope's arena
  arena* m_scope_arena;

 public:
  __device__ allocation_scope()
    : m_output_arena(nullptr), m_shared_arena(nullptr), m_scope_arena(nullptr)
  {
  }

  __device__ allocation_scope(mt_arena<cuda::thread_scope_system>* output_arena,
                              arena* shared_arena,
                              arena* scope_arena)
    : m_output_arena(output_arena), m_shared_arena(shared_arena), m_scope_arena(scope_arena)
  {
  }

  __device__ allocation_scope(allocation_scope const&)            = default;
  __device__ allocation_scope(allocation_scope&&)                 = default;
  __device__ allocation_scope& operator=(allocation_scope const&) = default;
  __device__ allocation_scope& operator=(allocation_scope&&)      = default;
  __device__ ~allocation_scope()                                  = default;

  template <storage_type Type>
  __device__ allocation allocate(size_t size)
  {
    if constexpr (Type == storage_type::OUTPUT) {
      if (m_output_arena != nullptr) {
        if (auto mem = m_output_arena->allocate(size); mem != nullptr) {
          return allocation{mem, memory_source::OUTPUT};
        }
      }

      if (auto mem = m_heap.allocate(size); mem != nullptr) {
        return allocation{mem, memory_source::HEAP};
      }

      return allocation{nullptr, memory_source::NONE};

    } else if constexpr (Type == storage_type::TEMPORARY) {
      if (m_shared_arena != nullptr) {
        if (auto mem = m_shared_arena->allocate(size); mem != nullptr) {
          return allocation{mem, memory_source::SHARED};
        }
      }

      if (m_scope_arena != nullptr) {
        if (auto mem = m_scope_arena->allocate(size); mem != nullptr) {
          return allocation{mem, memory_source::SCOPE};
        }
      }

      if (auto mem = m_heap.allocate(size); mem != nullptr) {
        return allocation{mem, memory_source::HEAP};
      }

      return allocation{nullptr, memory_source::NONE};
    } else {
      __builtin_unreachable();
    }
  }

  template <storage_type Type>
  __device__ void deallocate(allocation allocation, size_t size)
  {
    switch (allocation.source) {
      case memory_source::NONE: {
        assert(allocation.memory == nullptr && size == 0);
      } break;
      case memory_source::HEAP: {
        m_heap.deallocate(allocation.memory, size);
      } break;
      case memory_source::OUTPUT: {
        m_output_arena->deallocate(allocation.memory, size);
      } break;
      case memory_source::SHARED: {
        m_shared_arena->deallocate(allocation.memory, size);
      } break;
      case memory_source::SCOPE: {
        m_scope_arena->deallocate(allocation.memory, size);
      } break;
      default: __builtin_unreachable();
    }
  }

  template <storage_type Type>
  __device__ allocation reallocate(allocation alloc, size_t old_size, size_t new_size)
  {
    // first try to re-allocate on its current source, it'd be cheaper and have less fragmentation
    switch (alloc.source) {
      case memory_source::NONE: {
        assert(alloc.memory == nullptr && old_size == 0 && new_size == 0);
      } break;
      case memory_source::HEAP: {
        if (auto* mem = m_heap.reallocate(alloc.memory, old_size, new_size); mem != nullptr) {
          return allocation{mem, memory_source::HEAP};
        }
      } break;
      case memory_source::OUTPUT: {
        if (auto* mem = m_output_arena->reallocate(alloc.memory, old_size, new_size);
            mem != nullptr) {
          return allocation{mem, memory_source::OUTPUT};
        }
      } break;
      case memory_source::SHARED: {
        if (auto* mem = m_shared_arena->reallocate(alloc.memory, old_size, new_size);
            mem != nullptr) {
          return allocation{mem, memory_source::SHARED};
        }
      } break;
      case memory_source::SCOPE: {
        if (auto* mem = m_scope_arena->reallocate(alloc.memory, old_size, new_size);
            mem != nullptr) {
          return allocation{mem, memory_source::SCOPE};
        }
      } break;

      default: __builtin_unreachable();
    }

    auto new_alloc = allocate<Type>(new_size);

    if (new_alloc.memory == nullptr) { return allocation{}; }

    memcpy(new_alloc.memory, alloc.memory, std::min(old_size, new_size));

    deallocate<Type>(alloc, old_size);

    return new_alloc;
  }
};

template <typename Storage>
static __device__ Storage make_storage(size_t capacity, allocation_scope scope)
{
  Storage s{scope};
  s.reserve(capacity);
  return s;
}

template <typename Storage>
static __device__ Storage make_storage_copy(void const* data, size_t size, allocation_scope scope)
{
  Storage s{scope};
  s.reserve(size);
  memcpy(s.memory(), data, size);
  return s;
}

template <typename Storage>
class storage_base {
 protected:
  Storage& super() { return static_cast<Storage&>(*this); }

  Storage const& super() const { return static_cast<Storage const&>(*this); }

  __device__ void reallocate(size_t capacity)
  {
    CUDF_EXPECTS(super().try_reallocate(capacity), "");
  }

  __device__ bool try_reserve(size_t capacity)
  {
    if (super().m_capacity >= capacity) { return true; }

    return super().try_reallocate(capacity);
  }

  __device__ void reserve(size_t capacity)
  {
    // TODO:
    CUDF_EXPECTS(try_reserve(capacity), "");
  }

  __device__ bool try_grow(size_t target_size)
  {
    if (super().m_capacity >= target_size) { return true; }

    return super().try_reallocate(target_size << 1);
  }

  __device__ void grow(size_t target_size) { CUDF_EXPECTS(try_grow(target_size), ""); }
};

template <storage_type Type>
class scoped_storage;

class heap_storage : protected storage_base<heap_storage> {
 protected:
  void* m_memory;

  size_t m_capacity;

 public:
  __device__ heap_storage(void* memory, size_t capacity, [[maybe_unused]] allocation_scope scope)
    : m_memory{memory}, m_capacity{capacity}
  {
  }

  __device__ heap_storage([[maybe_unused]] allocation_scope scope)
    : m_memory{nullptr}, m_capacity{0}
  {
  }

  __device__ heap_storage(heap_storage const&) = delete;

  __device__ heap_storage(heap_storage&& other)
    : m_memory{other.m_memory}, m_capacity{other.m_capacity}
  {
    other.m_memory   = nullptr;
    other.m_capacity = 0;
  }

  __device__ heap_storage& operator=(heap_storage const&) = delete;

  __device__ heap_storage& operator=(heap_storage&& other)
  {
    if (this == &other) { return *this; }

    m_memory         = other.m_memory;
    other.m_memory   = nullptr;
    m_capacity       = other.m_capacity;
    other.m_capacity = 0;

    return *this;
  }

  __device__ ~heap_storage() { reset(); }

  __device__ void reset()
  {
    heap_allocator::deallocate(m_memory, m_capacity);
    m_memory   = nullptr;
    m_capacity = 0;
  }

  __device__ void leak()
  {
    m_memory   = nullptr;
    m_capacity = 0;
  }

  __device__ void* memory() const { return m_memory; }

  __device__ size_t capacity() const { return m_capacity; }

  __device__ bool try_reallocate(size_t capacity)
  {
    void* alloc;
    if (alloc = heap_allocator::reallocate(m_memory, m_capacity, capacity); alloc == nullptr) {
      return false;
    }

    m_memory   = alloc;
    m_capacity = capacity;

    return true;
  }

  /// @brief Take ownership of the storage if the storage types and memory sources are compatible,
  /// otherwise copy the bytes
  template <storage_type SrcType>
  __device__ void receive(scoped_storage<SrcType> other, size_t max_copy_size = -1);

  __device__ void receive(heap_storage other, [[maybe_unused]] size_t max_copy_size = -1)
  {
    *this = std::move(other);
  }

  template <storage_type Type>
  friend class scoped_storage;
};

template <storage_type Type>
class scoped_storage : protected storage_base<scoped_storage<Type>> {
  using base = storage_base<scoped_storage<Type>>;

 protected:
  void* m_memory;

  size_t m_capacity;

  memory_source m_source;

  allocation_scope m_scope;

 public:
  static constexpr storage_type type = Type;

  __device__ scoped_storage(void* memory,
                            size_t capacity,
                            memory_source source,
                            allocation_scope scope)
    : m_memory{memory}, m_capacity{capacity}, m_source{source}, m_scope{scope}
  {
  }

  __device__ scoped_storage(allocation_scope scope = allocation_scope{})
    : scoped_storage{nullptr, 0, 0, memory_source::NONE, scope}
  {
  }

  __device__ scoped_storage(scoped_storage&& other)
    : m_memory{other.m_memory},
      m_capacity{other.m_capacity},
      m_source{other.m_source},
      m_scope{other.m_scope}
  {
    other.m_memory   = nullptr;
    other.m_capacity = 0;
    other.m_source   = memory_source::NONE;
    other.m_scope    = allocation_scope{};
  }

  template <storage_type SrcType, CUDF_ENABLE_IF(Type != SrcType)>
  __device__ scoped_storage(scoped_storage<SrcType>&& other)
    : m_memory{nullptr}, m_capacity{0}, m_source{}, m_scope{}
  {
    // the storage needs to be transferred to the heap since an allocation_scope is not yet
    // provided.
    if (other.is_heap_sourced()) {
      m_memory         = other.m_memory;
      other.m_memory   = nullptr;
      m_capacity       = other.m_capacity;
      other.m_capacity = 0;
      m_source         = other.m_source;
      other.m_source   = memory_source::NONE;
      other.m_scope    = allocation_scope{};
    } else {
      base::reserve(other.m_capacity);
      memcpy(m_memory, other.m_memory, other.m_capacity);
    }
  }

  __device__ scoped_storage(scoped_storage const&) = delete;

  __device__ scoped_storage(heap_storage&& other)
    : m_memory{other.m_memory},
      m_capacity{other.m_capacity},
      m_source{memory_source::HEAP},
      m_scope{}
  {
    other.m_memory   = nullptr;
    other.m_capacity = 0;
  }

  __device__ scoped_storage& operator=(scoped_storage const&) = delete;

  __device__ scoped_storage& operator=(scoped_storage&& other)
  {
    if (this == &other) { return *this; }
    reset();
    m_memory         = other.m_memory;
    other.m_memory   = nullptr;
    m_capacity       = other.m_capacity;
    other.m_capacity = 0;
    m_source         = other.m_source;
    other.m_source   = memory_source::NONE;
    m_scope          = other.m_scope;
    other.m_scope    = allocation_scope{};
    return *this;
  }

  __device__ scoped_storage& operator=(heap_storage&& other)
  {
    reset();
    m_memory         = other.m_memory;
    other.m_memory   = nullptr;
    m_capacity       = other.m_capacity;
    other.m_capacity = 0;
    m_source         = memory_source::HEAP;
    m_scope          = allocation_scope{};
    return *this;
  }

  __device__ ~scoped_storage() { reset(); }

  __device__ void reset()
  {
    m_scope.deallocate<Type>(allocation{m_memory, m_source}, m_capacity);
    m_memory   = nullptr;
    m_capacity = 0;
    m_source   = memory_source::NONE;
  }

  __device__ void leak()
  {
    m_memory   = nullptr;
    m_capacity = 0;
    m_source   = memory_source::NONE;
  }

  __device__ void* memory() const { return m_memory; }

  __device__ size_t capacity() const { return m_capacity; }

  __device__ bool try_reallocate(size_t capacity)
  {
    allocation alloc;
    if (alloc = m_scope.reallocate<Type>(allocation{m_memory, m_source}, m_capacity, capacity);
        alloc.memory == nullptr) {
      return false;
    }

    m_memory   = alloc.memory;
    m_source   = alloc.source;
    m_capacity = capacity;

    return true;
  }

  __device__ bool is_heap_sourced() const { return m_source == memory_source::HEAP; }

  /// @brief Take ownership of the storage if the storage types and memory sources are compatible
  /// with this storage type, otherwise copy the bytes
  template <storage_type SrcType>
  __device__ void receive(scoped_storage<SrcType> other, size_t max_copy_size = -1)
  {
    if constexpr (Type == SrcType) {
      *this = std::move(other);
    } else {
      if (other.is_heap_sourced()) {
        *this = heap_storage{other.m_memory, other.m_capacity};
        other.leak();
      } else {
        // the sources are incompatible, the bytes need to be transferred here
        auto copy_size = std::min(max_copy_size, other.capacity());
        reserve(copy_size);
        memcpy(m_memory, other.m_memory, copy_size);
      }
    }
  }

  __device__ void receive(heap_storage other, [[maybe_unused]] size_t max_copy_size = -1)
  {
    *this = std::move(other);
  }
};

template <storage_type SrcType>
__device__ void heap_storage::receive(scoped_storage<SrcType> other, size_t max_copy_size)
{
  if (other.is_heap_sourced()) {
    *this = heap_storage{other.m_memory, other.m_capacity, allocation_scope{}};
    other.leak();
  } else {
    auto copy_size = std::min(max_copy_size, other.capacity());
    reserve(copy_size);
    memcpy(m_memory, other.m_memory, copy_size);
  }
}

// TODO: utility to estimate memory required across all threads for intermediates

/**
 * @brief Device string class for use with user-defined functions
 *
 * This class manages a device buffer of UTF-8 encoded characters
 * for string manipulation in a device kernel.
 *
 * Its methods and behavior are modelled after std::string but
 * with special consideration for UTF-8 encoded strings and for
 * use within a cuDF UDF.
 */
template <typename Storage>
class string : protected Storage {
 public:
  using storage_type = Storage;

  /**
   * @brief Represents unknown character position or length
   */
  static constexpr cudf::size_type npos = static_cast<cudf::size_type>(-1);

  /**
   * @brief Cast to cudf::string_view operator
   */
  __device__ operator cudf::string_view() const
  {
    return cudf::string_view(Storage::m_data, Storage::m_size);
  }

  /**
   * @brief Create an empty string.
   */
  __device__ string(allocation_scope scope = {});

  /**
   * @brief Create a string using existing device memory
   *
   * The given memory is copied into the instance returned.
   *
   * @param data Device pointer to UTF-8 encoded string
   * @param bytes Number of bytes in `data`
   */
  __device__ string(char const* data, cudf::size_type bytes, allocation_scope scope = {});

  /**
   * @brief Create a string object from a null-terminated character array
   *
   * The given memory is copied into the instance returned.
   *
   * @param data Device pointer to UTF-8 encoded null-terminated
   *             character array.
   */
  __device__ string(char const* data, allocation_scope scope = {});

  /**
   * @brief Create a string object from a cudf::string_view
   *
   * The input string data is copied into the instance returned.
   *
   * @param str String to copy
   */
  __device__ string(cudf::string_view str, allocation_scope scope = {});

  /**
   * @brief Create a string object with `count` copies of character `chr`
   *
   * @param count Number of times to copy `chr`
   * @param chr Character from which to create the string
   */
  __device__ string(cudf::size_type count, cudf::char_utf8 chr, allocation_scope scope = {});

  /**
   * @brief Create a string object from another instance
   *
   * The string data is copied from the `src` into the instance returned.
   *
   * @param src String to copy
   */
  template <typename SrcStorage>
  __device__ string(string<SrcStorage> const& src);

  /**
   * @brief Move a string object from an rvalue reference
   *
   * The string data is moved from `src` into the instance returned.
   * The `src` will have no content.
   *
   * @param src String to copy
   */
  template <typename SrcStorage>
  __device__ string(string<SrcStorage>&& src) noexcept;

  __device__ ~string() = default;

  template <typename SrcStorage>
  __device__ string& operator=(string<SrcStorage> const& src);

  template <typename SrcStorage>
  __device__ string& operator=(string<SrcStorage>&& src) noexcept;

  __device__ string& operator=(cudf::string_view const);

  __device__ string& operator=(char const*);

  /**
   * @brief Return the number of bytes in this string
   */
  __device__ cudf::size_type size_bytes() const noexcept;

  /**
   * @brief Return the number of characters in this string
   */
  __device__ cudf::size_type length() const noexcept;

  /**
   * @brief Return the maximum number of bytes a string can hold
   */
  __device__ constexpr cudf::size_type max_size() const noexcept;

  /**
   * @brief Return the internal pointer to the character array for this object
   */
  __device__ char* data() noexcept;
  __device__ char const* data() const noexcept;

  /**
   * @brief Returns true if there are no characters in this string
   */
  __device__ bool is_empty() const noexcept;

  /**
   * @brief Returns an iterator that can be used to navigate through
   *        the UTF-8 characters in this string
   *
   * This returns a `cudf::string_view::const_iterator` which is read-only.
   */
  __device__ cudf::string_view::const_iterator begin() const noexcept;
  __device__ cudf::string_view::const_iterator end() const noexcept;

  /**
   * @brief Returns the character at the specified position
   *
   * This will return 0 if `pos >= length()`.
   *
   * @param pos Index position of character to return
   * @return Character at position `pos`
   */
  __device__ cudf::char_utf8 at(cudf::size_type pos) const;

  /**
   * @brief Returns the character at the specified index
   *
   * This will return 0 if `pos >= length()`.
   * Note this is read-only. Use replace() to modify a character.
   *
   * @param pos Index position of character to return
   * @return Character at position `pos`
   */
  __device__ cudf::char_utf8 operator[](cudf::size_type pos) const;

  /**
   * @brief Return the byte offset for a given character position
   *
   * The byte offset for the character at `pos` such that
   * `data() + byte_offset(pos)` points to the memory location
   * the character at position `pos`.
   *
   * The behavior is undefined if `pos < 0 or pos >= length()`
   *
   * @param pos Index position of character to return byte offset.
   * @return Byte offset for character at `pos`
   */
  __device__ cudf::size_type byte_offset(cudf::size_type pos) const;

  /**
   * @brief Comparing target string with this string
   *
   * @param str Target string to compare with this string
   * @return 0  If they compare equal
   *         <0 Either the value of the first character of this string that does
   *            not match is ordered before the corresponding character in `str`,
   *            or all compared characters match but the `str` string is shorter.
   *         >0 Either the value of the first character of this string that does
   *            not match is ordered after the corresponding character in `str`,
   *            or all compared characters match but the `str` string is longer.
   */
  __device__ int compare(cudf::string_view str) const noexcept;

  /**
   * @brief Comparing target character array with this string
   *
   * @param str Target array of UTF-8 characters.
   * @param bytes Number of bytes in `str`.
   * @return 0  If they compare equal
   *         <0 Either the value of the first character of this string that does
   *            not match is ordered before the corresponding character in `str`,
   *            or all compared characters match but `bytes < size_bytes()`.
   *         >0 Either the value of the first character of this string that does
   *            not match is ordered after the corresponding character in `str`,
   *            or all compared characters match but `bytes > size_bytes()`.
   */
  __device__ int compare(char const* str, cudf::size_type bytes) const;

  /**
   * @brief Returns true if `rhs` matches this string exactly
   */
  __device__ bool operator==(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if `rhs` does not match this string
   */
  __device__ bool operator!=(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if this string is ordered before `rhs`
   */
  __device__ bool operator<(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if `rhs` is ordered before this string
   */
  __device__ bool operator>(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if this string matches or is ordered before `rhs`
   */
  __device__ bool operator<=(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if `rhs` matches or is ordered before this string
   */
  __device__ bool operator>=(cudf::string_view rhs) const noexcept;

  /**
   * @brief Remove all bytes from this string without deallocating
   *
   * All pointers, references, and iterators are invalidated.
   */
  __device__ void clear() noexcept;

  /**
   * @brief Remove all bytes from this string
   *
   * All pointers, references, and iterators are invalidated.
   */
  __device__ void reset() noexcept;

  /**
   * @brief Resizes string to contain `count` bytes
   *
   * If `count > size_bytes()` then zero-padding is added.
   * If `count < size_bytes()` then the string is truncated to size `count`.
   *
   * All pointers, references, and iterators may be invalidated.
   *
   * The behavior is undefined if `count > max_size()`
   *
   * @param count Size in bytes of this string.
   */
  __device__ void resize(cudf::size_type count);

  /**
   * @brief Reserve `count` bytes in this string
   *
   * If `count > capacity()`, new memory is allocated and `capacity()` will
   * be greater than or equal to `count`.
   * There is no effect if `count <= capacity()`.
   *
   * @param count Total number of bytes to reserve for this string
   */
  __device__ void reserve(cudf::size_type count);

  /**
   * @brief Returns the number of bytes that the string has allocated
   */
  __device__ cudf::size_type capacity() const noexcept;

  /**
   * @brief Reduces internal allocation to just `size_bytes()`
   *
   * All pointers, references, and iterators may be invalidated.
   */
  __device__ void shrink_to_fit();

  /**
   * @brief Moves the contents of `str` into this string instance
   *
   * On return, the `str` will have no contents.
   *
   * @param str String to move
   * @return This string with new contents
   */
  template <typename SrcStorage>
  __device__ string& assign(string<SrcStorage>&& str) noexcept;

  /**
   * @brief Replaces the contents of this string with contents of `str`
   *
   * @param str String to copy
   * @return This string with new contents
   */
  __device__ string& assign(cudf::string_view str);

  /**
   * @brief Replaces the contents of this string with contents of `str`
   *
   * @param str Null-terminated UTF-8 character array
   * @return This string with new contents
   */
  __device__ string& assign(char const* str);

  /**
   * @brief Replaces the contents of this string with contents of `str`
   *
   * @param str UTF-8 character array
   * @param bytes Number of bytes to copy from `str`
   * @return This string with new contents
   */
  __device__ string& assign(char const* str, cudf::size_type bytes);

  /**
   * @brief Append a string to the end of this string
   *
   * @param str String to append
   * @return This string with the appended argument
   */
  __device__ string& operator+=(cudf::string_view str);

  /**
   * @brief Append a character to the end of this string
   *
   * @param str Character to append
   * @return This string with the appended argument
   */
  __device__ string& operator+=(cudf::char_utf8 chr);

  /**
   * @brief Append a null-terminated device memory character array
   * to the end of this string
   *
   * @param str String to append
   * @return This string with the appended argument
   */
  __device__ string& operator+=(char const* str);

  /**
   * @brief Append a null-terminated character array to the end of this string
   *
   * @param str String to append
   * @return This string with the appended argument
   */
  __device__ string& append(char const* str);

  /**
   * @brief Append a character array to the end of this string
   *
   * @param str Character array to append
   * @param bytes Number of bytes from `str` to append.
   * @return This string with the appended argument
   */
  __device__ string& append(char const* str, cudf::size_type bytes);

  /**
   * @brief Append a string to the end of this string
   *
   * @param str String to append
   * @return This string with the appended argument
   */
  __device__ string& append(cudf::string_view str);

  /**
   * @brief Append a character to the end of this string
   * a specified number of times.
   *
   * @param chr Character to append
   * @param count Number of times to append `chr`
   * @return This string with the append character(s)
   */
  __device__ string& append(cudf::char_utf8 chr, cudf::size_type count = 1);

  /**
   * @brief Insert a string into the character position specified
   *
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * @param pos Character position to begin insert
   * @param str String to insert into this one
   * @return This string with the inserted argument
   */
  __device__ string& insert(cudf::size_type pos, cudf::string_view str);

  /**
   * @brief Insert a null-terminated character array into the character position specified
   *
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * @param pos Character position to begin insert
   * @param data Null-terminated character array to insert
   * @return This string with the inserted argument
   */
  __device__ string& insert(cudf::size_type pos, char const* data);

  /**
   * @brief Insert a character array into the character position specified
   *
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * @param pos Character position to begin insert
   * @param data Character array to insert
   * @param bytes Number of bytes from `data` to insert
   * @return This string with the inserted argument
   */
  __device__ string& insert(cudf::size_type pos, char const* data, cudf::size_type bytes);

  /**
   * @brief Insert a character one or more times into the character position specified
   *
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * @param pos Character position to begin insert
   * @param count Number of times to insert `chr`
   * @param chr Character to insert
   * @return This string with the inserted argument
   */
  __device__ string& insert(cudf::size_type pos, cudf::size_type count, cudf::char_utf8 chr);

  /**
   * @brief Returns a substring of this string
   *
   * An empty string is returned if `pos < 0 or pos >= length()`.
   *
   * @param pos Character position to start the substring
   * @param count Number of characters for the substring;
   *              This can be greater than the number of available characters.
   *              Default npos returns characters in range `[pos, length())`.
   * @return New string with the specified characters
   */
  __device__ string substr(cudf::size_type pos, cudf::size_type count = npos) const;

  /**
   * @brief Replace a range of characters with a given string
   *
   * Replaces characters in range `[pos, pos + count]` with `str`.
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * If `count==0` then `str` is inserted starting at `pos`.
   * If `count==npos` then the replacement range is `[pos,length())`.
   *
   * @param pos Position of first character to replace
   * @param count Number of characters to replace
   * @param str String to replace the given range
   * @return This string modified with the replacement
   */
  __device__ string& replace(cudf::size_type pos, cudf::size_type count, cudf::string_view str);

  /**
   * @brief Replace a range of characters with a null-terminated character array
   *
   * Replaces characters in range `[pos, pos + count)` with `data`.
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * If `count==0` then `data` is inserted starting at `pos`.
   * If `count==npos` then the replacement range is `[pos,length())`.
   *
   * @param pos Position of first character to replace
   * @param count Number of characters to replace
   * @param data Null-terminated character array to replace the given range
   * @return This string modified with the replacement
   */
  __device__ string& replace(cudf::size_type pos, cudf::size_type count, char const* data);

  /**
   * @brief Replace a range of characters with a given character array
   *
   * Replaces characters in range `[pos, pos + count)` with `[data, data + bytes)`.
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * If `count==0` then `data` is inserted starting at `pos`.
   * If `count==npos` then the replacement range is `[pos,length())`.
   *
   * @param pos Position of first character to replace
   * @param count Number of characters to replace
   * @param data String to replace the given range
   * @param bytes Number of bytes from data to use for replacement
   * @return This string modified with the replacement
   */
  __device__ string& replace(cudf::size_type pos,
                             cudf::size_type count,
                             char const* data,
                             cudf::size_type bytes);

  /**
   * @brief Replace a range of characters with a character one or more times
   *
   * Replaces characters in range `[pos, pos + count)` with `chr` `chr_count` times.
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * If `count==0` then `chr` is inserted starting at `pos`.
   * If `count==npos` then the replacement range is `[pos,length())`.
   *
   * @param pos Position of first character to replace
   * @param count Number of characters to replace
   * @param chr_count Number of times `chr` will repeated
   * @param chr Character to use for replacement
   * @return This string modified with the replacement
   */
  __device__ string& replace(cudf::size_type pos,
                             cudf::size_type count,
                             cudf::size_type chr_count,
                             cudf::char_utf8 chr);

  /**
   * @brief Removes specified characters from this string
   *
   * Removes `min(count, length() - pos)` characters starting at `pos`.
   * There is no effect if `pos < 0 or pos >= length()`.
   *
   * @param pos Character position to begin insert
   * @param count Number of characters to remove starting at `pos`
   * @return This string with remove characters
   */
  __device__ string& erase(cudf::size_type pos, cudf::size_type count = npos);

  // TODO: add multi-append method

 private:
  __device__ cudf::size_type char_offset(cudf::size_type byte_pos) const;
  __device__ void shift_bytes(cudf::size_type start_pos,
                              cudf::size_type end_pos,
                              cudf::size_type nbytes);

  cudf::size_type m_size_bytes;
};

/// @brief General-purpose strings that can outlive their parent scope
using udf_string = string<heap_storage>;

/// @brief used for strings that are only valid within a specific CUDA thread scope
using tmp_string = string<scoped_storage<storage_type::TEMPORARY>>;

/// @brief used for strings that outlive the CUDA kernel but can be sub-allocated from an output
/// buffer.
using out_string = string<scoped_storage<storage_type::OUTPUT>>;

}  // namespace udf
}  // namespace strings
}  // namespace cudf
