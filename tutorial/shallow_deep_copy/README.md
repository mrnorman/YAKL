This exists as an example to show shallow and deep copies. When you assign one array to another, think of that as pointer assignment. The left-hand-side (LHS) will then share the same data pointer as the right-hand-side (RHS), even though they are distinct Array objects. Changes to one will affect the other.

To create a new copy with a distinct data buffer, use the `arr.createHostCopy()` and `arr.createDeviceCopy()` routines.

If you have two arrays with distinct data buffers already, and you want to copy the contents of `a` into `b`, use `a.deep_copy_to(b)`.

```
cd YAKL/tutorials/shallow_deep_copy
g++ -I../../src -I../../src/extensions -I../../external shallow_deep_copy.cpp && ./a.out
```

If you want more insight into what YAKL is doing, you can turn on the `YAKL_VERBOSE` flag:

```
cd YAKL/tutorials/shallow_deep_copy
g++ -DYAKL_VERBOSE -I../../src -I../../src/extensions -I../../external shallow_deep_copy.cpp && ./a.out
```


