# Rust

## Resources

- https://doc.rust-lang.org/book/
- https://doc.rust-lang.org/rust-by-example/
- https://tourofrust.com/

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

## Rust by example

### 1 Hello World

https://doc.rust-lang.org/rust-by-example/hello.html
```rust
#[derive(Debug)]
struct Point { x: i32, y: i32, }

fn main() {
    let p = Point { x: 10, y: 20 };
    println!("{:#?}", p);
}
```

- `#[derive(Debug)]`
    - Debug
        - the name of `trait` to implement methods for.
            - (`fmt()` in this case)
    - `#[...]`
        - represents for an attribute
    - `derive`
        - a built-in macro only used within `#[...]`
- `println!`
    - `!`
        - indicates `println` is a macro
    - `{}`
        - requires `Display` trait
    - `{:?}`
        - requires `Debug` trait
    - `{:#?}`
        - requires `Debug` trait (?)
        - for pretty print

### 2 Primitives

https://doc.rust-lang.org/rust-by-example/primitives/literals.html
https://doc.rust-lang.org/rust-by-example/primitives/tuples.html (TODO)

### 5 Types

- `u8`
- `i8`
- `i16`
- `f32`
- `char`
- `usize`
    - 8 bytes
            - used for heap variables
        - ptr
        - len
        - capacity

#### 5.1 Casting

- 

### 15 Scoping rules

- local scope
- global scope 😮

#### 15.1 RAII

(Resource Acquisition Is Initialization)

```rust
// raii.rs
fn create_box() {
    // Allocate an integer on the heap
    let _box1 = Box::new(3i32);

    // `_box1` is destroyed here, and memory gets freed
}

fn main() {
    // Allocate an integer on the heap
    let _box2 = Box::new(5i32);

    // A nested scope:
    {
        // Allocate an integer on the heap
        let _box3 = Box::new(4i32);

        // `_box3` is destroyed here, and memory gets freed
    }

    // Creating lots of boxes just for fun
    // There's no need to manually free memory!
    for _ in 0u32..1_000 {
        create_box();
    }

    // `_box2` is destroyed here, and memory gets freed
}
```

Destructor:

```rust
struct ToDrop;

impl Drop for ToDrop {
    fn drop(&mut self) {
        println!("ToDrop is being dropped");
    }
}

fn main() {
    let x = ToDrop;
    println!("Made a ToDrop!");
}
```

#### 15.2 Ownership and moves

##### 15.2.1 Mutability

```rust
fn main() {
    let immutable_box = Box::new(5u32);

    println!("immutable_box contains {}", immutable_box);

    // Mutability error
    //*immutable_box = 4;

    // *Move* the box, changing the ownership (and mutability)
    let mut mutable_box = immutable_box;

    println!("mutable_box contains {}", mutable_box);

    // Modify the contents of the box
    *mutable_box = 4;

    println!("mutable_box now contains {}", mutable_box);
}
```

##### 15.2.2 Partial moves

```rust
fn main() {
    #[derive(Debug)]
    struct Person {
        name: String,
        age: Box<u8>,
    }

    // Error! cannot move out of a type which implements the `Drop` trait
    //impl Drop for Person {
    //    fn drop(&mut self) {
    //        println!("Dropping the person struct {:?}", self)
    //    }
    //}
    // TODO ^ Try uncommenting these lines

    let person = Person {
        name: String::from("Alice"),
        age: Box::new(20),
    };

    // `name` is moved out of person, but `age` is referenced
    let Person { name, ref age } = person;

    println!("The person's age is {}", age);

    println!("The person's name is {}", name);

    // Error! borrow of partially moved value: `person` partial move occurs
    //println!("The person struct is {:?}", person);

    // `person` cannot be used but `person.age` can be used as it is not moved
    println!("The person's age from person struct is {}", person.age);
}
```

#### 15.3 Borrowing

#### 15.4 Lifetimes

### Unsorted

- `trait`
    - it's like interfaces or mixins
    - can provide a default implementation
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
        - move the ownership of the types without `Copy` trait
            - e.g. String, Vec, Box, ...
            - 👉 efficient
            - 👉 safe
    - mut T
        - not allowed
- macros
    - (declarative - what)
        - `macro_rules! foo`
    - (procedural - how)
        - `#[proc_macro]`
            - 
        - `#[proc_macro_derive]`
            - `derive`
        - `#[proc_macro_attribute]`
            - 

## Tour of Rust

TODO:

https://tourofrust.com/29_en.html
https://tourofrust.com/30_en.html

```rust
#![allow(dead_code)] // this line prevents compiler warnings

enum Species { Crab, Octopus, Fish, Clam }
enum PoisonType { Acidic, Painful, Lethal }
enum Size { Big, Small }
enum Weapon {
    Claw(i32, Size),
    Poison(PoisonType),
    None
}

struct SeaCreature {
    species: Species,
    name: String,
    arms: i32,
    legs: i32,
    weapon: Weapon,
}

fn main() {
    let ferris = SeaCreature {
        // String struct is also on stack,
        // but holds a reference to data on heap
        species: Species::Crab,
        name: String::from("Ferris"),
        arms: 2,
        legs: 4,
        weapon: Weapon::Claw(2, Size::Small),
    };

    match ferris.species {
        Species::Crab => {
            match ferris.weapon {
                Weapon::Claw(num_claws,size) => {
                    let size_description = match size {
                        Size::Big => "big",
                        Size::Small => "small"
                    };
                    println!("ferris is a crab with {} {} claws", num_claws, size_description)
                },
                _ => println!("ferris is a crab with some other weapon")
            }
        },
        _ => println!("ferris is some other animal"),
    }
}
```
