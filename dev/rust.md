# Rust

## Resources

- https://doc.rust-lang.org/book/
- https://doc.rust-lang.org/rust-by-example/
- https://tourofrust.com/

## Glossary

- unit
- module

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
- 

### 2 Primitives

https://doc.rust-lang.org/rust-by-example/primitives/literals.html
https://doc.rust-lang.org/rust-by-example/primitives/tuples.html (TODO)

### Unsorted

- `trait`
    - it's like interfaces or mixins
    - can provide a default implementation
- reference types for function arguments
    - &T
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
