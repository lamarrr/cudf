# Column-backed scalars

`cudf::scalar_column` owns a one-row `cudf::column`. It is a parallel API;
existing scalar types and factories keep their behavior. Include
`scalar_column_factories.hpp` to use factories in
`cudf::scalar_column_factories`. Factories return `unique_ptr<scalar_column>`
and accept the existing stream and device resource arguments.

```cpp
auto value = cudf::scalar_column_factories::make_fixed_width_scalar<int>(42);
auto scalar_view = value->view();
auto column_view = scalar_view.as_column_view();
```

A `scalar_column_view` borrows any one-row column view, including slices.
Conversion preserves offsets, children, data pointers, and masks. It performs
no device allocation or synchronization; copying host child metadata may
allocate host memory. Constructing an owner from a view makes a deep copy.
Constructing from `unique_ptr<column>` or calling `release()` transfers ownership
without copying device buffers. Released and moved-from owners must be assigned
a new owner before use.

Include `scalar_column_device_view.cuh` for immutable and mutable device views.
Their `create()` handles retain child metadata and must outlive GPU work using
the views; the data owner must also remain alive. Pass `*handle` by value to a
kernel. `element<T>()`, `is_null()`, and `is_valid()` address row zero and account
for slice offsets. Children remain ordinary column views.

Device null checks directly read column bitmasks and require no stream argument.
Host `null_count()` and `has_nulls()` inspect metadata, not device memory.
Mutable device views can write representable column elements and change mask
bits with `set_null()` and `set_valid()`. Validity writes require an allocated
mask. After GPU mask changes, update the owner's null count before creating
fresh immutable views or using host null metadata. As with columns, callers
coordinate stream ordering and ensure borrowed views remain valid.

Factories for type-only fixed-width scalars produce invalid values with
uninitialized payloads. Value factories produce valid values. Default-constructed
scalars are invalid with default payloads. Strings and lists distinguish empty
values from null values. Nested values use ordinary column child layouts.
Host-value factories complete their host-to-device copies before returning;
they may synchronize the supplied stream. Device-view construction may allocate
and copy child metadata, following column-device-view semantics.

The storage layout is shared with columns. Explicit conversions are supported;
`reinterpret_cast` between different wrapper objects is not supported.

Consumer migration and performance profiling belong to a subsequent branch.
Capture legacy baselines before migrating scalar-intensive operations, measuring
construction and reuse separately, with host latency, GPU time, allocations,
and synchronization costs on identical hardware and build configurations.
