# Rust

## Glossary

- unit
  - `()`
- 



## Online book

https://doc.rust-lang.org/book/



## Online tutorial

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





