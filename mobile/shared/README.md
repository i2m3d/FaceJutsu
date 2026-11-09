# shared

The engine of the iPhone app, in Rust: it finds and follows faces and swaps them on the GPU through
Metal (`objc2-metal`). Swift bindings are generated with UniFFI. Unsafe code stays inside the GPU
backend.

The crate builds as a static library for the app. The bindings come from the built library:

```
cargo run --features bindgen --bin uniffi-bindgen -- \
  generate --library <path to the built library> --language swift --out-dir <dir>
```

`cargo test` runs on any desktop system.
