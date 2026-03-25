# Rust

## Resources

- https://doc.rust-lang.org/book/
- https://doc.rust-lang.org/rust-by-example/
- https://tourofrust.com/
- https://patterns.contextgeneric.dev/

## Glossary

- unit
    - represented by empty tuple `()`
    - similar to `None` in Python
- module
- diverging function
    - A function that never returns
- move
    - refers to transferring ownership of a value
- borrow
- destructuring
    - (not related to destructor)
    - supports complicated assignments
- destructor
    - (not related to destructuring)
    - a function to be called when a value is freed
- slice
    - two-words
        - pointer

## Commands and tips

```sh
rustup show
rustup update stable
rustc --version
```

### Cargo

```sh
cargo new <project_name>
cargo build
cargo run
cargo test
cargo check

# delete all the compiled binaries
cargo clean

# format your code
cargo fmt

# lint your code
cargo clippy

# build documentation and open it
cargo doc --open
```

### Setting up build cache

#### Windows

- install scoop (Refer to https://scoop.sh/)

```sh
code C:\Users\<user_name>\.cargo\config.toml
```

`config.toml`:

```toml
[build]
rustc-wrapper = "C:\\Users\\hotohoto\\scoop\\apps\\sccache\\current\\sccache.exe"
```

## Debugging in VS Code

### cppvsdbg

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug executable 'try-bevy'",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/try-bevy.exe",
            "args": [],
            "cwd": "${workspaceFolder}",
            "environment": [
                {
                    "name": "CARGO_MANIFEST_DIR",
                    "value": "${workspaceFolder}"
                }
            ]
        }
    ]
}
```

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug executable 'try-bevy'",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/try-bevy.exe",
            "args": [],
            "cwd": "${workspaceFolder}",
            "environment": [
                {
                    "name": "CARGO_MANIFEST_DIR",
                    "value": "${workspaceFolder}"
                },
                {
                    "name": "PATH",
                    "value": "${env:PATH};${workspaceFolder}/target/debug;C:/Users/hotohoto/.rustup/toolchains/stable-x86_64-pc-windows-msvc/bin;${workspaceFolder}/target/debug/deps"
                }
            ],
            "externalConsole": true
        }
    ]
}
```

- run
    - `cargo run -vv --features bevy/dynamic_linking`
- and see what kind of settings are required

### lldb

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug try-bevy",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/try-bevy.exe",
            "args": [],
            "env": {
                "CARGO_MANIFEST_DIR": "${workspaceFolder}"
            }
        }
    ]
}
```

### Unsorted

- `trait`
    - it's like interfaces or mixins
    - can provide a default implementation
- `&` operator
    - (expression - shared borrow)
        - `let r = &x;`
        - creates a shared reference (borrows `x`)
    - (mutable borrow expression)
        - `let r = &mut x;`
        - creates a mutable reference (borrows mutably)
    - (type position)
        - `fn foo(x: &i32)`
        - declares a reference type
    - (pattern matching)
        - `match r { &v => println!("{}", v), }`
        - matches and dereferences a reference value
    - (with lifetime)
        - `&'a T`
        - reference with explicit lifetime
- `ref`
    - (pattern only)
            - `let Some(ref s) = value;`
            - binds by reference instead of moving
            - this _is borrowing_ (just done via pattern binding)
- `*` operator
    - (expression)
        - `*r`
        - dereferences a reference to access the value
    - (pattern)
        - implicit via &pattern (e.g., &v)
        - removes one layer of reference during matching
- reference types for function arguments
    - &T
        - read only
        - don't copy the original value but the reference
            - 👉 efficient
        - don't allow the value of the original caller's variable
            - 👉 safe
    - &mut T
        - don't copy the original value but the reference
            - 👉 efficient
        - allow the value of the original caller's variable
            - 👉 kind of safe since it allows only one mutable reference at a time 🤔
                - By doing this, Rust prevents
                    - use after free (UAF)
                    - data race
    - T
        - copy the value of the types with `Copy` trait
            - 👉 efficient since it's used for small data
                - e.g. i32, f32, bool, char, ...
            - 👉 safe
        - move the ownership of the types that doesn't implement `Copy` trait
            - e.g. String, Vec, Box, ...
            - 👉 efficient
            - 👉 safe
    - (mut T)
        - not allowed!!
- macros
    - (declarative - what)
        - `macro_rules! foo`
    - (procedural - how)
        - `#[proc_macro]`
            - 
        - `#[proc_macro_derive]`
            - `derive`
        - `#[proc_macro_attribute]`
