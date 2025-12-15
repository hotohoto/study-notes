# A rust study note

Study guide:

- Try "Rust By Practice 💪" with the help of any AI chatbot.
    - https://practice.course.rs/
    - https://youtu.be/BpPEoZW5IiY?si=pArhPWoSP5vxLP-Z
- Summarize things in the structure of "Rust By Example 🔎"
    - https://doc.rust-lang.org/rust-by-example/
- Also refer to "Tour of Rust 🎒" if needed
    - https://tourofrust.com

## 1 Hello World

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
        - requires `Debug` trait
        - for pretty print

## 2 Primitives

### 2.1 Literals and operators

https://doc.rust-lang.org/rust-by-example/primitives/literals.html

### 2.2 Tuples

- https://doc.rust-lang.org/rust-by-example/primitives/tuples.html
- https://practice.course.rs/compound-types/tuple.html

### 2.3 Arrays and slices

https://doc.rust-lang.org/rust-by-example/primitives/array.html

## 3 Custom types

### 3.1 Structures

- https://doc.rust-lang.org/rust-by-example/custom_types/structs.html
- https://practice.course.rs/compound-types/struct.html
- It's not allowed to mark only certain fields as mutable.

(struct update syntax)

```rust

// Fill the blank to make the code work
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}
fn main() {
    let u1 = User {
        email: String::from("someone@example.com"),
        username: String::from("sunface"),
        active: true,
        sign_in_count: 1,
    };

    let u2 = set_email(u1);

    println!("Success!");
} 

fn set_email(u: User) -> User {
    User {
        email: String::from("contact@im.dev"),
        ..u  // struct update syntax
    }
}
```

### 3.2 Enums

- https://tourofrust.com/29_en.html
- https://tourofrust.com/30_en.html
- https://practice.course.rs/compound-types/enum.html
    - TODO

#### 3.2.1 use

#### 3.2.2 C-like

#### 3.2.3 Testcase: linked-list

### 3.3 Constants

## 4 Variable bindings

### 4.1 Mutability

### 4.2 Scope and shadowing

### 4.3 Declare first

### 4.4 Freezing

## 5 Types

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

### 5.1 Casting

### 5.2 Literals

### 5.3 Inference

### 5.4 Aliasing

## 15 Scoping rules

- local scope
- global scope 😮

### 15.1 RAII

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

### 15.2 Ownership and moves

#### 15.2.1 Mutability

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

#### 15.2.2 Partial moves

"Within the destructuring of a single variable, both by-move and by-reference pattern bindings can be used at the same time. Doing this will result in a partial move of the variable, which means that parts of the variable will be moved while other parts stay. In such a case, the parent variable cannot be used afterwards as a whole, however the parts that are only referenced (and not moved) can still be used."

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

- Default destructuring behavior:
    - if a field type implements the `Copy` trait
        - values are copied
        - e.g., `i32`, `bool`, `char`
    - if not
        - ownership is moved
        - e.g., `String`, `Vec`, `Box<T>`
- `ref`
    - only for destructuring
    - changes the ownership/type of the variable created during destructuring
    - Non-Copy types → creates a read-only reference (`&T`) instead of moving
    - Copy types → can create a reference (`&T`) instead of copying the value
- Heap variables (`Box`, `Vec`, `String`, etc.)
    - stack stores a smart pointer containing the address
    - The actual data is on the heap
        - It can be thought of as accessing via its address
- `&`
    - takes the address of a stack or heap value
- `*`
    - dereference to access the actual value

### 15.3 Borrowing

### 15.4 Lifetimes

## 16 Traits

### Unsorted

(type converting between tuple structs)

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

impl From<Point> for Color {
    fn from(p: Point) -> Self {
        Color(p.0, p.1, p.2)
    }
}

fn main() {
    let v = Point(0, 127, 255);
    check_color(v.into());

    println!("Success!");
}   

fn check_color(p: Color) {
    let Color(x, y, z) = p;
    assert_eq!(x, 0);
    assert_eq!(p.1, 127);
    assert_eq!(z, 255);
}
```

- Implementing `From<Point>` for `Color` automatically provides `Into<Color>` for `Point` via the blanket implementation of `Into`.
