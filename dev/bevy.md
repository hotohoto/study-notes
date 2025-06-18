# Bevy

- https://bevyengine.org/
- https://github.com/bevyengine/bevy
- y-up, right handed
- written in [Rust](rust.md).

## Getting started

- entity
- `Component`
- system
- `Plugin`
    - a module that modifies App
    - e.g.
        - `UiPlugin`
        - `RenderPlugin`
- `Resource`
    - global data
- `Commands`
    - represents mutation to be applied to `World`

### ECS

https://bevyengine.org/learn/quick-start/getting-started/ecs/

```rust
use bevy::prelude::*;

#[derive(Component)]
struct Person;

#[derive(Component)]
struct Name(String);

#[derive(Resource)]
struct GreetTimer(Timer);

pub struct HelloPlugin;


fn add_people(mut commands: Commands) {
    commands.spawn((Person, Name("Elaina Proctor".to_string())));
    commands.spawn((Person, Name("Renzo Hume".to_string())));
    commands.spawn((Person, Name("Zayna Nieves".to_string())));
}

fn update_people(mut query: Query<&mut Name, With<Person>>) {
    for mut name in &mut query {
        if name.0 == "Elaina Proctor" {
            name.0 = "Elaina Hume".to_string();
            break; // We don't need to change any other names.
        }
    }
}

fn greet_people(time: Res<Time>, mut timer: ResMut<GreetTimer>, query: Query<&Name, With<Person>>) {
    // update our timer with the time elapsed since the last update
    // if that caused the timer to finish, we say hello to everyone
    if timer.0.tick(time.delta()).just_finished() {
        for name in &query {
            println!("hello {}!", name.0);
        }
    }
}

impl Plugin for HelloPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GreetTimer(Timer::from_seconds(2.0, TimerMode::Repeating)));
        app.add_systems(Startup, add_people);
        app.add_systems(Update, (update_people, greet_people).chain());
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(HelloPlugin)
        .run();
}

```

## 3D Rendering

### 3D Scene

https://bevyengine.org/examples/3d-rendering/3d-scene/

## Physics engine

https://github.com/Jondolf/avian
